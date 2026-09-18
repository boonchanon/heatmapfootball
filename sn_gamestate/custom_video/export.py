"""Export TrackLab predictions to the frontend trajectory contract."""

import json
import math
from pathlib import Path

import numpy as np


def _value(value):
    """Convert pandas/numpy scalars to JSON-safe primitive values."""
    if value is None:
        return None
    try:
        if bool(value != value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, 'item'):
        value = value.item()
    return value


def _finite_pitch(value):
    if not isinstance(value, dict):
        return None
    try:
        x = float(value['x_bottom_middle'])
        y = float(value['y_bottom_middle'])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    return {'pitch_x': x, 'pitch_y': y}


def _image_frame(image_by_id, image_id):
    image = image_by_id.get(image_id)
    if image is None:
        return None
    frame = image.get('frame', image_id)
    return int(frame) if frame is not None else None


def export_tracking(detections, images, video, output):
    """Write honest player trajectories and metadata from TrackLab DataFrames."""
    image_by_id = {}
    if hasattr(images, 'iterrows'):
        for index, image in images.iterrows():
            key = _value(image.get('image_id', index))
            image_by_id[key] = image.to_dict()
    grouped = {}
    for _, row in detections.iterrows():
        track_id = _value(row.get('track_id'))
        if track_id is None:
            continue
        frame = _image_frame(image_by_id, _value(row.get('image_id')))
        if frame is None:
            continue
        position = {'frame': frame, 'time': frame / float(video['fps'])}
        bbox_pitch = _finite_pitch(row.get('bbox_pitch'))
        position.update(bbox_pitch or {'pitch_x': None, 'pitch_y': None})
        position['pitch_valid'] = bbox_pitch is not None
        position['inside_pitch'] = (abs(bbox_pitch['pitch_x']) <= 52.5
                                    and abs(bbox_pitch['pitch_y']) <= 34) if bbox_pitch else None
        for attribute in ['role', 'team', 'jersey_number']:
            value = _value(row.get(attribute))
            position[attribute] = (None if isinstance(value, str)
                                   and value.lower() in {'nan', 'none', 'null'} else value)
        bbox = row.get('bbox_ltwh')
        if (isinstance(bbox, (list, tuple, np.ndarray))
                and np.shape(bbox) == (4,) and np.isfinite(bbox).all()):
            position['image_x'] = float(bbox[0]) + float(bbox[2]) / 2
            position['image_y'] = float(bbox[1]) + float(bbox[3])
            position['bbox_truncated'] = bool(
                bbox[0] <= 0 or bbox[1] <= 0
                or bbox[0] + bbox[2] >= video.get('width', float('inf')) - 1
                or bbox[1] + bbox[3] >= video.get('height', float('inf')) - 1)
        confidence = _value(row.get('bbox_conf'))
        if confidence is not None:
            position['detection_confidence'] = float(confidence)
        grouped.setdefault(track_id, []).append(position)

    players = []
    for track_id, positions in grouped.items():
        positions.sort(key=lambda item: item['frame'])
        players.append({'track_id': track_id, 'positions': positions})
    players.sort(key=lambda player: str(player['track_id']))
    frames = sorted({_image_frame(image_by_id, key) for key in image_by_id}
                    - {None})
    processing = {
        'frames_processed': len(frames),
        'first_frame': frames[0] if frames else None,
        'last_frame': frames[-1] if frames else None,
        'processed_duration_seconds': len(frames) / float(video['fps']),
        'quality_verified': False,
    }
    tracking = {'video': video, 'processing': processing, 'players': players}
    metadata = {
        **video,
        **processing,
        'track_count': len(players),
        'position_count': sum(len(player['positions']) for player in players),
        'finite_pitch_position_count': sum(
            position['pitch_x'] is not None
            for player in players for position in player['positions']),
        'outside_pitch_position_count': sum(
            position['inside_pitch'] is False
            for player in players for position in player['positions']),
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'tracking.json').write_text(
        json.dumps(tracking, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    return tracking, metadata
