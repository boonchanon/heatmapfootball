"""Re-run ByteTrack on trusted local predictions without repeating neural inference."""

import argparse
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import torch
from byte_track.byte_tracker import BYTETracker
from byte_track.basetrack import BaseTrack
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sn_gamestate.custom_video.export import export_tracking
from sn_gamestate.custom_video.runner import summarize_state


def coverage(detections):
    counts = detections.groupby('track_id').size()
    return {'detections': len(detections),
            'tracked_detections': int(detections.track_id.notna().sum()),
            'coverage_percent': round(100 * detections.track_id.notna().mean(), 2),
            'track_count': len(counts), 'tracks_with_at_most_5_detections': int((counts <= 5).sum())}


def retrack(detections, images, fps, threshold):
    BaseTrack._count = 0
    tracker = BYTETracker(track_thresh=threshold, match_thresh=.8,
                          track_buffer=30, frame_rate=fps)
    result = detections.copy(deep=True)
    result['track_id'] = np.nan
    result['track_bbox_conf'] = np.nan
    result['track_bbox_ltwh'] = pd.Series(None, index=result.index, dtype=object)
    previous = None
    for image_id, image in images.sort_values('frame').iterrows():
        frame = int(image.frame)
        if previous is not None and frame != previous + 1:
            raise ValueError('Input frames must be contiguous; segment videos before tracking')
        previous = frame
        rows = detections[detections.image_id == image_id]
        inputs = []
        for index, row in rows.iterrows():
            if float(row.bbox_conf) <= .4:
                continue
            l, t, w, h = map(float, row.bbox_ltwh)
            inputs.append([l, t, l+w, t+h, float(row.bbox_conf), float(row.category_id), int(index)])
        predictions = tracker.update(torch.tensor(inputs, dtype=torch.float64).reshape(-1, 7), None)
        for l, t, r, b, track_id, _, confidence, index in predictions:
            index = int(index)
            if index not in rows.index:
                raise ValueError('Tracker returned a detection from a different frame')
            result.at[index, 'track_id'] = track_id
            result.at[index, 'track_bbox_conf'] = confidence
            result.at[index, 'track_bbox_ltwh'] = np.array([l, t, r-l, b-t])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=.5)
    args = parser.parse_args()
    if not .1 <= args.threshold <= .9:
        parser.error('--threshold must be between .1 and .9')
    if args.output.exists():
        parser.error('Use a new output directory')
    with zipfile.ZipFile(args.run / 'tracker_state.pklz') as archive:
        detections = pd.read_pickle(archive.open('0.pkl'))
        images = pd.read_pickle(archive.open('0_image.pkl'))
        summary = json.loads(archive.read('summary.json'))
    evidence = json.loads((args.run / 'inference_evidence.json').read_text())
    if evidence.get('stage') != 'mvp':
        parser.error('Replay supports MVP states only; full-pipeline attributes depend on the original track IDs')
    updated = retrack(detections, images, evidence['video']['fps'], args.threshold)
    args.output.mkdir(parents=True)
    with zipfile.ZipFile(args.output / 'tracker_state.pklz', 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('summary.json', json.dumps(summary))
        with archive.open('0.pkl', 'w') as file:
            updated.to_pickle(file)
        with archive.open('0_image.pkl', 'w') as file:
            images.to_pickle(file)
    config = OmegaConf.load(args.run / 'resolved_config.yaml')
    config.modules.track.cfg.hyperparams.track_thresh = args.threshold
    config.modules.track.cfg.hyperparams.frame_rate = evidence['video']['fps']
    # Record the configuration whose detections/calibration were reused.
    OmegaConf.save(config, args.output / 'source_config_with_tracking_overrides.yaml')
    evidence.update(summarize_state(updated, images))
    evidence.update(source_run=str(args.run.resolve()), geometry_reused=True,
                    tracking_overrides={'track_thresh': args.threshold,
                                        'frame_rate': evidence['video']['fps']})
    (args.output / 'inference_evidence.json').write_text(json.dumps(evidence, indent=2))
    comparison = {'before': coverage(detections), 'after': coverage(updated),
                  'identity_accuracy_measured': False, 'geometry_changed': False}
    (args.output / 'comparison.json').write_text(json.dumps(comparison, indent=2))
    export_tracking(updated, images, evidence['video'], args.output)
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
