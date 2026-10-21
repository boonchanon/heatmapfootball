"""Frame sampling (analyse every Nth frame) keeps original frame ids and real time."""

from collections import deque

import cv2
import numpy as np
import pandas as pd
import pytest

from sn_gamestate.custom_video.review import trail_points
from sn_gamestate.custom_video.runner import configure_tracking_set_segment, select_frame_segment
from sn_gamestate.custom_video.video import resolve_frame_step, stream_resized_frames


@pytest.fixture
def numbered_video(tmp_path):
    """30 frames whose brightness encodes the frame index (index * 8)."""
    path = tmp_path / 'numbered.mp4'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (64, 48))
    assert writer.isOpened(), 'Test host needs an MP4 encoder'
    try:
        for index in range(30):
            writer.write(np.full((48, 64, 3), index * 8, dtype=np.uint8))
    finally:
        writer.release()
    return path


@pytest.mark.parametrize('fps,sample_fps,step', [(50, None, 1), (50, 0, 1), (50, 5, 10), (25, 5, 5),
                                                 (29.97, 5, 6), (25, 60, 1)])
def test_resolve_frame_step(fps, sample_fps, step):
    assert resolve_frame_step(fps, sample_fps) == step


@pytest.mark.parametrize('sample_fps', [-1, float('nan'), float('inf')])
def test_resolve_frame_step_rejects_invalid(sample_fps):
    with pytest.raises(ValueError):
        resolve_frame_step(50, sample_fps)


def test_stream_keeps_every_nth_original_frame(numbered_video, tmp_path):
    result = stream_resized_frames(numbered_video, tmp_path / 'cache', max_frames=4, start_frame=3, frame_step=5)
    names = sorted(path.name for path in (tmp_path / 'cache').iterdir())
    assert names == [f'numbered_{index:06d}.jpg' for index in (3, 8, 13, 18)]
    assert (result['start_frame'], result['end_frame'], result['frame_count'], result['frame_step']) == (3, 19, 4, 5)
    # Brightness proves the right source frame was written, not just the right name.
    frame = cv2.imread(str(tmp_path / 'cache' / 'numbered_000013.jpg'))
    assert abs(float(frame.mean()) - 13 * 8) < 6


def test_stream_stops_at_end_of_video(numbered_video, tmp_path):
    result = stream_resized_frames(numbered_video, tmp_path / 'cache', max_frames=100, frame_step=10)
    assert result['frame_count'] == 3 and result['end_frame'] == 21


def test_select_frame_segment_with_step_matches_stream():
    images = pd.DataFrame({'frame': range(1000)})
    selected, start_frame, end_frame = select_frame_segment(
        images, {'fps': 50, 'frame_count': 1000}, start_seconds=10, max_frames=4, frame_step=10)
    assert selected.frame.tolist() == [500, 510, 520, 530]
    assert (start_frame, end_frame) == (500, 531)


def test_select_frame_segment_with_step_is_clipped_to_video():
    images = pd.DataFrame({'frame': range(100)})
    selected, _, end_frame = select_frame_segment(images, {'fps': 50, 'frame_count': 100}, max_frames=50, frame_step=30)
    assert selected.frame.tolist() == [0, 30, 60, 90]
    assert end_frame == 91


def test_configure_tracking_set_segment_with_step(tmp_path):
    from types import SimpleNamespace

    for frame in (500, 510, 520):
        (tmp_path / f'clip_{frame:06d}.jpg').write_bytes(b'x')
    images = pd.DataFrame({'id': range(1000), 'video_id': [7] * 1000, 'frame': range(1000)})
    tracking_set = SimpleNamespace(image_metadatas=images.copy(), image_gt=pd.DataFrame())
    segment = configure_tracking_set_segment(tracking_set, {'fps': 50, 'frame_count': 1000}, tmp_path, 'clip', 10, 3, 10)
    assert segment == (500, 521)
    assert tracking_set.image_gt.frame.tolist() == [500, 510, 520]


def test_trails_continue_across_sampled_frames_but_reset_on_real_gaps():
    history = deque()
    assert trail_points(history, 0, (0, 0), 5, frame_step=10) == [(0, 0)]
    assert trail_points(history, 10, (.5, 0), 5, frame_step=10) == [(0, 0), (.5, 0)]
    # A missed sampled frame (20 skipped) is a real gap: do not bridge it.
    assert trail_points(history, 30, (1, 0), 5, frame_step=10) == [(1, 0)]
