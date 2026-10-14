"""Export TrackLab predictions to the frontend trajectory contract."""

import json
import math
from pathlib import Path

import numpy as np
from .scene_validation import build_frame_decisions, usable_pitch
from .identity import stabilize_identities


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


def export_tracking(detections, images, video, output, start_seconds=0, start_frame=0, end_frame=None):
    """Write honest player trajectories and metadata from TrackLab DataFrames."""
    image_by_id = {}
    if hasattr(images, 'iterrows'):
        for index, image in images.iterrows():
            key = _value(image.get('image_id', index))
            image_by_id[key] = image.to_dict()
    decisions = build_frame_decisions(images)
    stabilized, identity_summary = stabilize_identities(
        detections, images, video['fps'],
        {_value(image_id): decision.get('segment_id') for image_id, decision in decisions.items()})
    grouped, rejected = {}, []
    for _, row in stabilized.iterrows():
        track_id = _value(row.get('identity_id'))
        if track_id is None:
            continue
        frame = _image_frame(image_by_id, _value(row.get('image_id')))
        if frame is None:
            continue
        decision = decisions.get(_value(row.get('image_id')), {})
        position = {'frame': frame, 'time': frame / float(video['fps']),
                    'segment_id': decision.get('segment_id'),
                    'scene_status': decision.get('scene_status', 'UNCERTAIN'),
                    'calibration_status': decision.get('calibration_status', 'invalid'),
                    'reject_reason': decision.get('reject_reason'),
                    'camera_cut': decision.get('camera_cut', False),
                    'camera_cut_score': decision.get('camera_cut_score')}
        source_track_id = _value(row.get('track_id'))
        position['source_track_id'] = source_track_id
        position['identity_status'] = row.get('identity_status') or 'untracked'
        confidence = _value(row.get('identity_confidence'))
        position['identity_confidence'] = float(confidence) if confidence is not None else None
        bbox_pitch = _finite_pitch(row.get('bbox_pitch'))
        # Preserve a finite but off-pitch projection for diagnostics; it is
        # never admitted to a heatmap. Invalid scenes get no coordinates at all.
        raw_pitch = bbox_pitch if decision.get('scene_valid') else None
        valid_pitch = usable_pitch(row.get('bbox_pitch')) if raw_pitch else None
        position.update(raw_pitch or {'pitch_x': None, 'pitch_y': None})
        position['pitch_valid'] = raw_pitch is not None
        position['usable_for_heatmap'] = valid_pitch is not None
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
        identity = (position['segment_id'], track_id)
        grouped.setdefault(identity, []).append(position)
        if not position['usable_for_heatmap']:
            rejected.append({'frame': frame, 'segment_id': position['segment_id'],
                             'track_id': track_id, 'reason': position['reject_reason'] or 'outside_pitch_or_projection_failed'})

    players = []
    for (segment_id, track_id), positions in grouped.items():
        positions.sort(key=lambda item: item['frame'])
        players.append({'segment_id': segment_id, 'track_id': track_id,
                        'identity_statuses': sorted({position['identity_status'] for position in positions}),
                        'positions': positions})
    players.sort(key=lambda player: (str(player['segment_id']), str(player['track_id'])))
    frames = sorted({_image_frame(image_by_id, key) for key in image_by_id}
                    - {None})
    processing = {
        'frames_processed': len(frames),
        'first_frame': frames[0] if frames else None,
        'last_frame': frames[-1] if frames else None,
        'processed_duration_seconds': len(frames) / float(video['fps']),
        'start_seconds': float(start_seconds),
        'start_frame': int(start_frame),
        'end_frame': int(end_frame) if end_frame is not None else (frames[-1] + 1 if frames else int(start_frame)),
        'quality_verified': False,
    }
    scenes = []
    for segment_id in sorted({d['segment_id'] for d in decisions.values() if d['segment_id']}):
        segment_frames = [d['frame'] for d in decisions.values() if d['segment_id'] == segment_id]
        start_frame, last_frame = min(segment_frames), max(segment_frames)
        first = next(d for d in decisions.values() if d['frame'] == start_frame)
        scenes.append({'segment_id': segment_id, 'start_frame': start_frame,
                       'end_frame': last_frame + 1,
                       'start_time': start_frame / float(video['fps']),
                       'end_time': (last_frame + 1) / float(video['fps']),
                       'start_reason': first.get('segment_start_reason')})
    summary = {'total_frames': len(frames), 'valid_gameplay_frames': sum(d['scene_status'] == 'VALID_GAMEPLAY' for d in decisions.values()),
        'invalid_scene_frames': sum(d['scene_status'] == 'INVALID_SCENE' for d in decisions.values()),
        'invalid_scenes': sum(d['scene_status'] == 'INVALID_SCENE' for d in decisions.values()),
        'uncertain_frames': sum(d['scene_status'] == 'UNCERTAIN' for d in decisions.values()),
        'fresh_valid_calibration_frames': sum(d['calibration_status'] == 'fresh_valid' for d in decisions.values()),
        'fallback_frames': 0, 'invalid_calibration_frames': sum(d['calibration_status'] == 'invalid' for d in decisions.values()),
        'scene_cuts': sum(d['scene_cut'] for d in decisions.values()),
        'camera_cuts': sum(d['camera_cut'] for d in decisions.values()),
        'segments': len({d['segment_id'] for d in decisions.values() if d['segment_id']}),
        'heatmap_usable_records': sum(p['usable_for_heatmap'] for x in players for p in x['positions']),
        'usable_heatmap_records': sum(p['usable_for_heatmap'] for x in players for p in x['positions']),
        'rejected_records': len(rejected), 'identity_summary': identity_summary}
    tracking = {'video': video, 'processing': processing, 'quality_summary': summary,
                'scenes': scenes, 'players': players}
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
        **summary,
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'tracking.json').write_text(
        json.dumps(tracking, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'metadata.json').write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'rejected_records.json').write_text(json.dumps(rejected, indent=2, allow_nan=False), encoding='utf-8')
    return tracking, metadata
