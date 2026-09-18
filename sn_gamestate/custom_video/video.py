"""Decode metadata without importing models or downloading checkpoints."""

import math
import shutil
from pathlib import Path


DEFAULT_MAX_FRAMES = 250
MAX_FRAMES = 250


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


def validate_max_frames(value):
    """Apply the web-service safety limit in one place."""
    if value is None:
        return DEFAULT_MAX_FRAMES
    value = int(value)
    if not 1 <= value <= MAX_FRAMES:
        raise ValueError(f'max_frames must be between 1 and {MAX_FRAMES}')
    return value


def stream_resized_frames(video, folder, max_frames=DEFAULT_MAX_FRAMES, max_width=1280):
    """Decode one frame at a time into a *temporary* TrackLab input cache.

    The TrackLab offline adapter requires paths, so the cache is on disk only and
    is removed by the caller after inference.  At no point is a video-wide frame
    list retained in RAM.
    """
    import cv2

    video = Path(video).resolve()
    folder = Path(folder).resolve()
    max_frames = validate_max_frames(max_frames)
    folder.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video))
    count = 0
    shape = None
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        while count < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, round(frame.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
            shape = frame.shape[:2]
            target = folder / f'{video.stem}_{count:06d}.jpg'
            if not cv2.imwrite(str(target), frame):
                raise RuntimeError(f'Cannot write temporary frame {target}')
            del frame
            count += 1
        if not count:
            raise ValueError(f'Video contains no decodable frames: {video}')
        return {'frame_count': count, 'width': int(shape[1]), 'height': int(shape[0]),
                'frame_cache': str(folder)}
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise
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
