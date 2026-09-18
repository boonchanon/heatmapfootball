"""Decode metadata without importing models or downloading checkpoints."""

import math
from pathlib import Path


def inspect_video(video):
    """Return decoded dimensions and nominal timing for a nonempty local MP4.

    Counts decoded frames instead of trusting the container frame-count estimate.
    Timing is nominal frame/FPS, not variable-frame-rate presentation timestamps.
    """
    import cv2

    video = Path(video).expanduser().resolve()
    if not video.is_file() or video.suffix.lower() != '.mp4':
        raise ValueError(f'Expected an existing local MP4 file: {video}')
    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open MP4: {video}')
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Video has no usable FPS; timestamps cannot be inferred')
        count = 0
        shape = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if shape is None:
                shape = frame.shape[:2]
            elif frame.shape[:2] != shape:
                raise ValueError('Variable frame dimensions are not supported')
            count += 1
        if not count:
            raise ValueError(f'MP4 contains no decodable frames: {video}')
        return {
            'name': video.name,
            'fps': fps,
            'width': int(shape[1]),
            'height': int(shape[0]),
            'frame_count': count,
            'duration_seconds': count / fps,
            'frame_origin': 0,
            'time_basis': 'nominal_frame_divided_by_fps',
        }
    finally:
        capture.release()


def extracted_frame_paths(folder, stem, frames):
    """Map ExternalVideo's zero-based frames to its existing JPEG helper output."""
    paths = []
    for frame in frames:
        if int(frame) != frame or frame < 0:
            raise ValueError(f'Invalid zero-based frame: {frame}')
        path = Path(folder).resolve() / f'{stem}_{int(frame):06d}.jpg'
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'Missing extracted frame: {path}')
        paths.append(str(path))
    return paths
