"""Conservative, model-free admission rules for custom-video heatmaps.

This consumes the existing pitch keypoints, lines and calibration output.  It
does not attempt to recognise television scenes: lack of fresh geometric
evidence is sufficient to reject a frame.
"""

import math

import cv2
import numpy as np


VALID_GAMEPLAY = 'VALID_GAMEPLAY'
INVALID_SCENE = 'INVALID_SCENE'
UNCERTAIN = 'UNCERTAIN'


def _count_mapping(value):
    return len(value) if isinstance(value, dict) else 0


def frame_quality(image, min_keypoints=4, min_lines=2):
    """Return scene/calibration status and an auditable rejection reason."""
    keypoints = _count_mapping(image.get('keypoints'))
    lines = _count_mapping(image.get('lines'))
    parameters = image.get('parameters')
    if keypoints < min_keypoints:
        return INVALID_SCENE, 'invalid', 'insufficient_keypoints'
    if lines < min_lines:
        return UNCERTAIN, 'invalid', 'insufficient_pitch_lines'
    # Empty parameters are what the current calibration modules emit when no
    # homography/camera fit was obtained.  Previous-frame reuse is disabled.
    if not isinstance(parameters, dict) or not parameters:
        return UNCERTAIN, 'invalid', 'homography_failed'
    return VALID_GAMEPLAY, 'fresh_valid', None


def camera_cut_scores(images, bins=32):
    """Return lightweight visual discontinuity scores keyed by image id.

    This is *shot* segmentation only.  It neither accepts nor rejects a
    frame for calibration/heatmap use.  Histogram distance is deliberately
    used instead of another learned model so the custom-video pipeline stays
    deterministic and has no new runtime dependency.
    """
    scores, previous = {}, None
    for image_id, image in images.sort_values('frame').iterrows():
        path = image.get('file_path')
        frame = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path else None
        if frame is None:
            scores[image_id] = None
            continue
        frame = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        histogram = cv2.calcHist([frame], [0], None, [bins], [0, 256])
        cv2.normalize(histogram, histogram)
        scores[image_id] = (None if previous is None else float(
            cv2.compareHist(previous, histogram, cv2.HISTCMP_BHATTACHARYYA)))
        previous = histogram
    return scores


def build_frame_decisions(images, invalid_gap=3, camera_cut_threshold=.10,
                          min_camera_cut_gap=15):
    """Build per-frame decisions and segment ids from existing image metadata.

    A new segment starts when valid gameplay resumes after an invalid gap, or
    at a confirmed visual camera cut. Track IDs are scoped to that segment by
    the exporter. A camera cut does not make an otherwise valid frame invalid.
    """
    decisions, invalid_run, segment = {}, 0, 0
    previous_valid = False
    last_camera_cut_frame = None
    cut_scores = camera_cut_scores(images)
    # Old callers which provide only frame numbers are test/legacy exports,
    # not calibrated pipeline output. Keep their contract while every actual
    # custom-video pipeline image has the three evidence columns.
    has_evidence = {'keypoints', 'lines', 'parameters'}.issubset(images.columns)
    for image_id, image in images.sort_values('frame').iterrows():
        frame = int(image.get('frame', image_id))
        scene, calibration, reason = (frame_quality(image) if has_evidence
                                      else (VALID_GAMEPLAY, 'fresh_valid', None))
        valid = scene == VALID_GAMEPLAY and calibration == 'fresh_valid'
        cut_score = cut_scores.get(image_id)
        visual_cut = (valid and cut_score is not None
                      and cut_score >= camera_cut_threshold
                      and (last_camera_cut_frame is None
                           or frame - last_camera_cut_frame >= min_camera_cut_gap))
        resumes_after_gap = valid and (not previous_valid or invalid_run >= invalid_gap)
        starts_segment = resumes_after_gap or visual_cut
        if not valid:
            invalid_run += 1
        else:
            if starts_segment:
                segment += 1
            if visual_cut:
                last_camera_cut_frame = frame
            invalid_run = 0
        if resumes_after_gap:
            segment_reason = 'gameplay_resume'
        elif visual_cut:
            segment_reason = 'camera_cut'
        else:
            segment_reason = None
        decisions[image_id] = {'frame': frame, 'scene_status': scene,
            'scene_valid': valid, 'calibration_status': calibration,
            'reject_reason': reason, 'segment_id': (f'segment_{segment:03d}' if valid else None),
            'scene_cut': bool(starts_segment), 'camera_cut': bool(visual_cut),
            'camera_cut_score': cut_score, 'segment_start_reason': segment_reason}
        previous_valid = valid
    return decisions


def usable_pitch(value):
    if not isinstance(value, dict):
        return None
    try:
        x, y = float(value['x_bottom_middle']), float(value['y_bottom_middle'])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y) or abs(x) > 52.5 or abs(y) > 34:
        return None
    return x, y
