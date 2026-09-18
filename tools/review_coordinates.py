"""Render a coordinate audit from a trusted, locally generated TrackLab state.

The cyan field overlay reconstructs the stored projection from its input/output
pairs; it is a visual diagnostic, not an independent calibration accuracy score.
"""

import argparse
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sn_gamestate.custom_video.export import export_tracking


def review(run, output):
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(run / 'tracker_state.pklz') as archive:
        detections = pd.read_pickle(archive.open('0.pkl'))
        images = pd.read_pickle(archive.open('0_image.pkl'))
    evidence = json.loads((run / 'inference_evidence.json').read_text())
    video = evidence['video']
    export_tracking(detections, images, video, output)
    writer = cv2.VideoWriter(str(output / 'review.mp4'),
                             cv2.VideoWriter_fourcc(*'mp4v'), video['fps'], (1500, 720))
    if not writer.isOpened():
        raise RuntimeError('Cannot create review MP4')
    outside = []
    samples = {0, len(images) // 2, len(images) - 1}
    try:
        for sequence, (image_id, metadata) in enumerate(images.iterrows()):
            frame = cv2.imread(metadata.file_path)
            if frame is None:
                raise ValueError(f'Cannot read frame {metadata.file_path}')
            rows = detections[detections.image_id == image_id]
            pixels, ground = [], []
            for _, row in rows.iterrows():
                point = row.get('bbox_pitch')
                if not isinstance(point, dict):
                    continue
                x, y = point['x_bottom_middle'], point['y_bottom_middle']
                if not np.isfinite([x, y]).all():
                    continue
                l, t, w, h = row.bbox_ltwh
                pixels.append([l + w / 2, t + h])
                ground.append([x, y])
            if len(pixels) >= 4:
                projection, _ = cv2.findHomography(np.array(ground), np.array(pixels), 0)
                if projection is not None:
                    lines = [np.array([[-52.5, -34], [52.5, -34], [52.5, 34],
                                       [-52.5, 34], [-52.5, -34]], dtype=float),
                             np.array([[0., -34.], [0., 34.]])]
                    angles = np.linspace(0, 2 * np.pi, 100)
                    lines.append(np.column_stack([9.15 * np.cos(angles), 9.15 * np.sin(angles)]))
                    for line in lines:
                        projected = cv2.perspectiveTransform(line[None], projection)[0]
                        if np.isfinite(projected).all() and np.abs(projected).max() < 1e6:
                            cv2.polylines(frame, [projected.astype(np.int32)], False, (255, 255, 0), 2)
            panel = np.full((720, 540, 3), (36, 72, 36), dtype=np.uint8)
            def field_pixel(x, y):
                return int(270 + x * 4), int(350 + y * 4)
            cv2.rectangle(panel, field_pixel(-52.5, -34), field_pixel(52.5, 34), (255, 255, 255), 2)
            cv2.line(panel, field_pixel(0, -34), field_pixel(0, 34), (255, 255, 255), 1)
            cv2.circle(panel, field_pixel(0, 0), round(9.15 * 4), (255, 255, 255), 1)
            for _, row in rows.iterrows():
                point = row.get('bbox_pitch')
                if not isinstance(point, dict):
                    continue
                x, y = float(point['x_bottom_middle']), float(point['y_bottom_middle'])
                if not np.isfinite([x, y]).all():
                    continue
                is_outside = abs(x) > 52.5 or abs(y) > 34
                track = None if pd.isna(row.track_id) else int(row.track_id)
                color = (0, 0, 255) if is_outside else ((0, 255, 0) if track is not None else (0, 180, 255))
                l, t, w, h = map(float, row.bbox_ltwh)
                if is_outside:
                    outside.append({'frame': int(metadata.frame), 'track_id': track,
                                    'pitch_x': x, 'pitch_y': y,
                                    'bbox_truncated': bool(l <= 0 or t <= 0 or l+w >= video['width']-1 or t+h >= video['height']-1)})
                cv2.rectangle(frame, (int(l), int(t)), (int(l+w), int(t+h)), color, 2)
                label = f'{track if track is not None else "?"}: {x:.1f},{y:.1f}'
                cv2.putText(frame, label, (int(l), max(18, int(t)-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
                cv2.circle(frame, (int(l+w/2), int(t+h)), 4, color, -1)
                if abs(x) < 65 and abs(y) < 75:
                    cv2.circle(panel, field_pixel(x, y), 4, color, -1)
                    cv2.putText(panel, str(track if track is not None else '?'), field_pixel(x+1, y-1), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1)
            for line, label in enumerate([
                f'Frame {int(metadata.frame)} / {video["frame_count"]}',
                'Red: outside canonical pitch', 'Green: tracked; orange: untracked',
                'Cyan: stored projection overlay', 'Coordinates are not quality certified']):
                cv2.putText(panel, label, (12, 35+line*27), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
            resized = cv2.resize(frame, (960, 540))
            left = np.zeros((720, 960, 3), dtype=np.uint8)
            left[90:630] = resized
            combined = np.hstack([left, panel])
            writer.write(combined)
            if sequence in samples:
                cv2.imwrite(str(output / f'frame_{int(metadata.frame):03d}.jpg'), combined)
    finally:
        writer.release()
    report = {'source_run': str(run.resolve()), 'frames_reviewed': len(images),
              'outside_detections': len(outside),
              'outside_tracked_detections': sum(p['track_id'] is not None for p in outside),
              'outside_track_ids': sorted({p['track_id'] for p in outside if p['track_id'] is not None}),
              'quality_verified': False, 'outside_points': outside}
    (output / 'coordinate_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'outside_points'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--video', type=Path,
                        help='Original source video; decoded sequentially for streaming review')
    parser.add_argument('--clean', action='store_true', help='Larger video, colored IDs and short observed trails')
    args = parser.parse_args()
    if args.clean:
        from sn_gamestate.custom_video.review import clean_review
        clean_review(args.run, args.output, args.video or args.run.parent / 'upload.mp4')
    else:
        review(args.run, args.output)
