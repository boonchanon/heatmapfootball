"""Decode metadata without importing models or downloading checkpoints."""

import math
import os
import shutil
from pathlib import Path


# Shared with tools/web_app.py's own MAX_ANALYSIS_FRAMES-derived limit so the
# CLI and the web service can never accept a request the other one rejects.
# Counts *analysed* frames: 15000 covers a 50-minute half at 5 sampled frames/s.
DEFAULT_MAX_FRAMES = MAX_FRAMES = int(os.environ.get('MAX_ANALYSIS_FRAMES', '15000'))
if MAX_FRAMES <= 0:
    raise ValueError('MAX_ANALYSIS_FRAMES must be greater than zero')


# Frames wider than this are downscaled before detection to save RAM. 0 keeps full
# resolution (best detection of small, distant players); small hosts set 1280.
ANALYSIS_MAX_WIDTH = int(os.environ.get('ANALYSIS_MAX_WIDTH', '0'))


def resolve_frame_step(fps, sample_fps=None):
    """Original frames to advance per analysed frame (1 = analyse every frame)."""
    if sample_fps in (None, 0, '0', ''):
        return 1
    sample_fps = float(sample_fps)
    if not math.isfinite(sample_fps) or sample_fps <= 0:
        raise ValueError('sample_fps must be a positive number')
    return max(1, round(float(fps) / sample_fps))


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


def stream_resized_frames(video, folder, max_frames=DEFAULT_MAX_FRAMES, max_width=ANALYSIS_MAX_WIDTH,
                          start_frame=0, frame_step=1):
    """Decode one frame at a time into a *temporary* TrackLab input cache.

    With ``frame_step`` > 1 only every Nth original frame is kept (frame sampling);
    ``max_frames`` counts kept frames.

    The TrackLab offline adapter requires paths, so the cache is on disk only and
    is removed by the caller after inference.  At no point is a video-wide frame
    list retained in RAM.  Frames before ``start_frame`` are decoded and discarded
    (not seeked to) so the kept frame count matches what ``inspect_video`` counted.
    Output files are named by their absolute original-frame index so segment
    selection can keep using original frame identifiers.
    """
    import cv2

    video = Path(video).resolve()
    folder = Path(folder).resolve()
    max_frames = validate_max_frames(max_frames)
    start_frame = int(start_frame)
    frame_step = int(frame_step)
    if start_frame < 0:
        raise ValueError('start_frame must be greater than or equal to zero')
    if frame_step < 1:
        raise ValueError('frame_step must be at least 1')
    folder.mkdir(parents=True, exist_ok=False)
    capture = cv2.VideoCapture(str(video))
    count = 0
    shape = None
    try:
        if not capture.isOpened():
            raise ValueError(f'Cannot open video: {video}')
        for _ in range(start_frame):
            ok, _ = capture.read()
            if not ok:
                raise ValueError('start_frame is outside the video duration')
        index = start_frame
        last_kept = start_frame
        while count < max_frames:
            if (index - start_frame) % frame_step:
                # grab() advances without converting the skipped frame to an image.
                if not capture.grab():
                    break
                index += 1
                continue
            ok, frame = capture.read()
            if not ok:
                break
            if max_width and frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, round(frame.shape[0] * scale)),
                                   interpolation=cv2.INTER_AREA)
            shape = frame.shape[:2]
            target = folder / f'{video.stem}_{index:06d}.jpg'
            if not cv2.imwrite(str(target), frame):
                raise RuntimeError(f'Cannot write temporary frame {target}')
            del frame
            last_kept = index
            count += 1
            index += 1
        if not count:
            raise ValueError(f'Video contains no decodable frames: {video}')
        return {'frame_count': count, 'width': int(shape[1]), 'height': int(shape[0]),
                'frame_cache': str(folder), 'start_frame': start_frame,
                'end_frame': last_kept + 1, 'frame_step': frame_step}
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
