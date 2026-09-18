"""Model-free tests. Generated color frames are fixtures, never inference evidence."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from sn_gamestate.custom_video.runner import build_config, summarize_state, working_directory
from sn_gamestate.custom_video.video import extracted_frame_paths, inspect_video
from sn_gamestate.custom_video.export import export_tracking


@pytest.fixture
def video(tmp_path):
    path = tmp_path / 'clip with spaces.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (64, 48))
    assert writer.isOpened(), 'Test host needs an MP4 encoder'
    try:
        for value in [20, 80, 140, 200]:
            writer.write(np.full((48, 64, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_decoded_metadata_and_json(video):
    result = inspect_video(video)
    assert (result['width'], result['height'], result['frame_count']) == (64, 48, 4)
    assert result['fps'] == pytest.approx(25)
    assert result['duration_seconds'] == pytest.approx(4 / 25)
    assert json.loads(json.dumps(result, allow_nan=False)) == result


@pytest.mark.parametrize('kind', ['missing', 'empty', 'corrupt', 'wrong_extension'])
def test_invalid_video(tmp_path, kind):
    path = tmp_path / ('clip.avi' if kind == 'wrong_extension' else 'clip.mp4')
    if kind != 'missing':
        path.write_bytes(b'not a video' if kind == 'corrupt' else b'')
    with pytest.raises(ValueError):
        inspect_video(path)


def test_invalid_fps_releases_capture(monkeypatch, tmp_path):
    path = tmp_path / 'clip.mp4'
    path.touch()
    class Capture:
        released = False
        def isOpened(self):
            return True
        def get(self, key):
            return float('nan')
        def release(self):
            self.released = True
    capture = Capture()
    monkeypatch.setattr(cv2, 'VideoCapture', lambda _: capture)
    with pytest.raises(ValueError, match='FPS'):
        inspect_video(path)
    assert capture.released


def test_extracted_frames_keep_zero_based_order(tmp_path):
    for frame in range(4):
        assert cv2.imwrite(str(tmp_path / f'clip_{frame:06d}.jpg'),
                           np.full((4, 4, 3), frame * 60, dtype=np.uint8))
    paths = extracted_frame_paths(tmp_path, 'clip', [0, 1, 2, 3])
    assert [round(float(cv2.imread(p).mean())) for p in paths] == [0, 60, 120, 180]


@pytest.mark.parametrize('frame', [-1, 0.5, 10])
def test_missing_or_invalid_extracted_frame(tmp_path, frame):
    with pytest.raises(ValueError):
        extracted_frame_paths(tmp_path, 'clip', [frame])


def test_empty_tracking():
    result = summarize_state(pd.DataFrame(), pd.DataFrame(index=[0, 1]))
    assert result['detections'] == result['track_count'] == 0
    assert result['finite_pitch_positions'] == 0
    assert result['frames'] == 2
    assert not result['quality_verified']


@pytest.mark.parametrize('point', [None, np.nan, {},
    {'x_bottom_middle': float('nan'), 'y_bottom_middle': 0},
    {'x_bottom_middle': 0, 'y_bottom_middle': float('inf')}])
def test_missing_or_nonfinite_calibration(point):
    rows = pd.DataFrame([{'track_id': 7, 'bbox_pitch': point}])
    result = summarize_state(rows, pd.DataFrame(index=[0]))
    assert result['tracked_detections'] == 1
    assert result['finite_pitch_positions'] == 0
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_finite_projection_counts_do_not_claim_quality():
    rows = pd.DataFrame([
        {'track_id': 7, 'bbox_pitch': {'x_bottom_middle': 1, 'y_bottom_middle': 2}},
        {'track_id': np.nan, 'bbox_pitch': {'x_bottom_middle': 4, 'y_bottom_middle': 5}},
    ])
    result = summarize_state(rows, pd.DataFrame(index=[0, 1]))
    assert result['finite_pitch_positions'] == 2
    assert result['tracked_finite_pitch_positions'] == 1
    assert not result['quality_verified']


def test_export_tracking_orders_positions_and_preserves_missing_pitch(tmp_path):
    detections = pd.DataFrame([
        {'track_id': 3, 'image_id': 'image-1', 'bbox_ltwh': [10, 20, 4, 8],
         'bbox_conf': np.float32(0.8),
         'bbox_pitch': {'x_bottom_middle': 2, 'y_bottom_middle': 4}},
        {'track_id': 3, 'image_id': 'image-0', 'bbox_ltwh': [1, 2, 3, 4],
         'bbox_pitch': None},
    ])
    images = pd.DataFrame([
        {'image_id': 'image-0', 'frame': 0},
        {'image_id': 'image-1', 'frame': 1},
    ]).set_index('image_id')
    video = {'fps': 25.0, 'width': 64, 'height': 48, 'frame_count': 2}

    tracking, metadata = export_tracking(detections, images, video, tmp_path)

    assert [p['frame'] for p in tracking['players'][0]['positions']] == [0, 1]
    assert tracking['players'][0]['positions'][0]['pitch_x'] is None
    assert tracking['players'][0]['positions'][1]['pitch_y'] == 4
    assert metadata['finite_pitch_position_count'] == 1
    assert json.loads((tmp_path / 'tracking.json').read_text()) == tracking
    assert json.loads((tmp_path / 'metadata.json').read_text()) == metadata


@pytest.mark.parametrize('stage', ['mvp', 'detect', 'track', 'calibrate', 'full'])
def test_compose_real_tracklab_configuration(video, tmp_path, stage):
    cfg = build_config(video, tmp_path / 'out', tmp_path / 'weights',
                       inspect_video(video), stage)
    assert cfg.dataset._target_ == 'tracklab.wrappers.ExternalVideo'
    assert cfg.dataset.eval_set == 'val'
    assert cfg.dataset.video_path == str(video)
    assert cfg.modules.calibration.image_width == 64
    assert cfg.modules.calibration.image_height == 48
    assert cfg.modules.calibration.use_prev_homography is False
    assert cfg.modules.calibration.batch_size == 1
    assert cfg.modules.bbox_detector.cfg.path_to_checkpoint.endswith('yolo11m.pt')
    assert cfg.eval_tracking is False
    if stage == 'mvp':
        assert list(cfg.pipeline) == ['bbox_detector', 'reid', 'track', 'pitch', 'calibration']
        assert cfg.modules.track.cfg.ecc is True
        assert cfg.modules.track.cfg.max_age == 75
        assert cfg.modules.track.cfg.max_kalman_prediction_without_update == 20
    if stage == 'calibrate':
        assert list(cfg.pipeline) == ['bbox_detector', 'reid', 'track', 'pitch', 'calibration']


def test_working_directory_restored_on_failure(tmp_path):
    previous = Path.cwd()
    with pytest.raises(RuntimeError):
        with working_directory(tmp_path):
            assert Path.cwd() == tmp_path
            raise RuntimeError('test fixture')
    assert Path.cwd() == previous


def test_numpy_bbox_and_partial_video_export(tmp_path):
    detections = pd.DataFrame([{
        'track_id': 1, 'image_id': 10,
        'bbox_ltwh': np.array([10., 20., 4., 8.]), 'bbox_pitch': None,
    }])
    # Include a processed frame without detections in the coverage metadata.
    images = pd.DataFrame({'frame': [0, 1]}, index=[10, 11])
    tracking, metadata = export_tracking(
        detections, images, {'fps': 25., 'frame_count': 750}, tmp_path)
    position = tracking['players'][0]['positions'][0]
    assert (position['image_x'], position['image_y']) == (12., 28.)
    assert metadata['frame_count'] == 750
    assert metadata['frames_processed'] == 2
    assert metadata['last_frame'] == 1
    assert metadata['processed_duration_seconds'] == pytest.approx(0.08)
    assert tracking['processing']['quality_verified'] is False


@pytest.mark.parametrize('count, expected', [(1, 0), (0, 3)])
def test_mvp_cli_proof_gate(monkeypatch, count, expected):
    from tools import analyze_video
    monkeypatch.setattr('sys.argv', ['analyze_video', '--stage', 'mvp',
                                    '--video', 'clip.mp4'])
    monkeypatch.setattr(analyze_video, 'run_video', lambda *args: {
        'tracked_finite_pitch_positions': count})
    assert analyze_video.main() == expected


def test_mvp_environment_check_skips_optional_stack(monkeypatch):
    from tools import analyze_video
    monkeypatch.setattr('sys.argv', ['analyze_video', '--stage', 'mvp',
                                    '--check-environment'])
    def check(require_optional):
        assert require_optional is False
        return {'ready_for_import_check': True}
    monkeypatch.setattr(analyze_video, 'check_environment', check)
    assert analyze_video.main() == 0


def test_export_outside_pitch_and_role_without_clipping(tmp_path):
    detections = pd.DataFrame([{
        'track_id': 7, 'image_id': 0, 'bbox_ltwh': np.array([10., 90., 20., 10.]),
        'bbox_pitch': {'x_bottom_middle': -5., 'y_bottom_middle': 36.},
        'role': 'other', 'team': 'nan', 'jersey_number': np.nan,
    }, {
        'track_id': 8, 'image_id': 0, 'bbox_ltwh': [20., 20., 10., 20.],
        'bbox_pitch': None, 'role': 'player', 'team': 'left', 'jersey_number': 9,
    }])
    tracking, metadata = export_tracking(detections, pd.DataFrame({'frame': [0]}),
        {'fps': 25., 'width': 100, 'height': 100, 'frame_count': 1}, tmp_path)
    outside = tracking['players'][0]['positions'][0]
    missing = tracking['players'][1]['positions'][0]
    assert outside['pitch_y'] == 36.
    assert outside['pitch_valid'] is True and outside['inside_pitch'] is False
    assert outside['role'] == 'other' and outside['team'] is None
    assert outside['bbox_truncated'] is True
    assert missing['pitch_valid'] is False and missing['inside_pitch'] is None
    assert missing['role'] == 'player' and missing['jersey_number'] == 9
    assert metadata['outside_pitch_position_count'] == 1
