"""Conservative post-tracking identity safeguards.

This layer never replaces StrongSORT.  It only makes exported/reviewed IDs
safer: short, well-supported gaps can reconnect, while implausible pitch jumps
split instead of silently assigning one player another player's identity.
"""

import math

import numpy as np


MAX_GAP_SECONDS = 1.0
MAX_SPEED_METERS_PER_SECOND = 9.0
JUMP_BASE_METERS = 3.0
MIN_REID_COSINE = 0.70


def _scalar(value):
    return value.item() if hasattr(value, 'item') else value


def _point(value):
    if not isinstance(value, dict):
        return None
    try:
        point = np.array([float(value['x_bottom_middle']), float(value['y_bottom_middle'])])
    except (KeyError, TypeError, ValueError):
        return None
    return point if np.isfinite(point).all() else None


def _embedding(value):
    if value is None:
        return None
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return vector if vector.size and np.isfinite(vector).all() else None


def _cosine(left, right):
    if left is None or right is None or left.shape != right.shape:
        return None
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator else None


def _mean_embedding(records):
    vectors = [record['embedding'] for record in records if record['embedding'] is not None]
    return np.mean(vectors, axis=0) if vectors else None


def _tracklets(detections, frame_by_image, segment_by_image, fps):
    """Split raw StrongSORT IDs only at physically implausible pitch jumps."""
    grouped = {}
    for index, row in detections.iterrows():
        raw_id = row.get('track_id')
        if raw_id is None or (isinstance(raw_id, float) and math.isnan(raw_id)):
            continue
        image_id = _scalar(row.get('image_id'))
        if image_id not in frame_by_image:
            continue
        record = {'index': index, 'frame': int(frame_by_image[image_id]),
                  'point': _point(row.get('bbox_pitch')),
                  'embedding': _embedding(row.get('embeddings'))}
        key = (segment_by_image.get(image_id), _scalar(raw_id))
        grouped.setdefault(key, []).append(record)

    result = []
    for (segment_id, raw_id), records in grouped.items():
        records.sort(key=lambda record: record['frame'])
        chunks = [[]]
        previous = None
        for record in records:
            split = False
            if previous and record['point'] is not None and previous['point'] is not None:
                elapsed = max(1, record['frame'] - previous['frame']) / fps
                allowed = JUMP_BASE_METERS + MAX_SPEED_METERS_PER_SECOND * elapsed
                split = np.linalg.norm(record['point'] - previous['point']) > allowed
            if split:
                chunks.append([])
            chunks[-1].append(record)
            previous = record
        for ordinal, chunk in enumerate(chunks):
            points = [record for record in chunk if record['point'] is not None]
            result.append({'segment_id': segment_id, 'raw_id': raw_id, 'ordinal': ordinal,
                           'records': chunk, 'start': chunk[0]['frame'], 'end': chunk[-1]['frame'],
                           'first_point': points[0]['point'] if points else None,
                           'last_point': points[-1]['point'] if points else None,
                           'embedding': _mean_embedding(chunk), 'identity_id': None,
                           'identity_status': None, 'identity_confidence': None})
    return sorted(result, key=lambda item: (str(item['segment_id']), item['start'], str(item['raw_id']), item['ordinal']))


def _predicted_distance(previous, current, fps):
    if previous['last_point'] is None or current['first_point'] is None:
        return None, None
    gap_frames = current['start'] - previous['end']
    if gap_frames <= 0:
        return None, None
    prior_points = [record for record in previous['records'] if record['point'] is not None]
    predicted = previous['last_point']
    if len(prior_points) >= 2:
        before, last = prior_points[-2], prior_points[-1]
        delta = max(1, last['frame'] - before['frame'])
        predicted = last['point'] + (last['point'] - before['point']) * (gap_frames / delta)
    distance = float(np.linalg.norm(current['first_point'] - predicted))
    allowed = JUMP_BASE_METERS + MAX_SPEED_METERS_PER_SECOND * (gap_frames / fps)
    return distance, allowed


def stabilize_identities(detections, images, fps, segment_by_image=None):
    """Return a copy with ``identity_*`` columns and an auditable summary.

    A re-link needs both a ReID cosine score and a feasible pitch-motion score;
    therefore missing evidence deliberately creates a new identity instead of a
    speculative merge.
    """
    result = detections.copy()
    result['identity_id'] = np.nan
    result['identity_status'] = None
    result['identity_confidence'] = np.nan
    frame_by_image = {_scalar(index): int(row.frame) for index, row in images.iterrows()}
    segment_by_image = segment_by_image or {}
    tracklets = _tracklets(result, frame_by_image, segment_by_image, float(fps))
    raw_values = [int(item['raw_id']) for item in tracklets
                  if isinstance(item['raw_id'], (int, np.integer, float, np.floating))]
    next_identity = max(raw_values, default=0) + 1
    by_segment = {}
    for item in tracklets:
        by_segment.setdefault(item['segment_id'], []).append(item)
    counts = {'stable': 0, 'reconnected': 0, 'new_after_gap': 0, 'split_on_jump': 0}
    for segment_id, items in by_segment.items():
        # Keep only the latest tracklet for each emitted identity.  Retaining
        # every historical root would let two later, overlapping raw tracklets
        # reconnect to the same old identity and double-count a frame.
        latest_by_identity = {}
        segment_start = min(item['start'] for item in items)
        for item in items:
            candidates = []
            for previous in latest_by_identity.values():
                gap = item['start'] - previous['end']
                if not 0 < gap <= round(MAX_GAP_SECONDS * fps):
                    continue
                similarity = _cosine(previous['embedding'], item['embedding'])
                distance, allowed = _predicted_distance(previous, item, fps)
                if similarity is not None and similarity >= MIN_REID_COSINE and distance is not None and distance <= allowed:
                    score = similarity - distance / max(allowed, 1e-6) * 0.15
                    candidates.append((score, similarity, distance, allowed, previous))
            if candidates:
                _, similarity, distance, allowed, previous = max(candidates, key=lambda value: value[0])
                item['identity_id'] = previous['identity_id']
                item['identity_status'] = 'reconnected'
                item['identity_confidence'] = min(0.98, 0.65 + 0.25 * similarity + 0.10 * (1 - distance / allowed))
            elif item['ordinal']:
                item['identity_id'] = next_identity; next_identity += 1
                item['identity_status'] = 'split_on_jump'; item['identity_confidence'] = 0.35
            else:
                item['identity_id'] = item['raw_id']
                item['identity_status'] = 'stable' if item['start'] == segment_start else 'new_after_gap'
                item['identity_confidence'] = 0.80 if item['identity_status'] == 'stable' else 0.55
            counts[item['identity_status']] += 1
            for record in item['records']:
                result.at[record['index'], 'identity_id'] = item['identity_id']
                result.at[record['index'], 'identity_status'] = item['identity_status']
                result.at[record['index'], 'identity_confidence'] = item['identity_confidence']
            latest_by_identity[item['identity_id']] = item
    counts['identity_tracklets'] = len(tracklets)
    counts['stable_identities'] = len({item['identity_id'] for item in tracklets})
    return result, counts
