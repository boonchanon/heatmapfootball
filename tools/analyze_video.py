"""Run the audited custom-MP4 proof pipeline using the original CV components."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sn_gamestate.custom_video.runner import STAGES, check_environment, run_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--model-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'pretrained_models')
    parser.add_argument('--stage', choices=STAGES, default='calibrate')
    parser.add_argument('--max-frames', type=int,
                        help='Process only the first contiguous real frames')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--check-environment', action='store_true')
    args = parser.parse_args()
    if args.check_environment:
        report = check_environment(require_optional=args.stage != 'mvp')
        print(json.dumps(report, indent=2))
        return 0 if report['ready_for_import_check'] else 2
    if args.video is None:
        parser.error('--video is required unless --check-environment is used')
    output = args.output or Path('outputs') / args.video.stem
    try:
        evidence = run_video(args.video, output, args.model_dir, args.stage, args.device,
                     args.max_frames)
    except Exception as error:
        print(f'Inference failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 1
    required = {'detect': 'detections', 'track': 'tracked_detections',
                'mvp': 'tracked_finite_pitch_positions',
                'calibrate': 'tracked_finite_pitch_positions',
                'full': 'tracked_finite_pitch_positions'}[args.stage]
    if not evidence[required]:
        print(f'Pipeline finished without {required}; proof gate not met.', file=sys.stderr)
        return 3
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
