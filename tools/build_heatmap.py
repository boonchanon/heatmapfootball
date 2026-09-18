"""Build a standalone offline heatmap page from exported tracking JSON."""

import argparse
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


def prepare_data(tracking):
    fps = float(tracking['video']['fps'])
    duration = float(tracking['processing']['processed_duration_seconds'])
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError('Expected a positive finite FPS and analyzed duration')
    samples, tracks, seen = [], [], set()
    excluded = 0
    for player in tracking['players']:
        value = player['track_id']
        track = str(int(value)) if isinstance(value, (float, int)) and int(value) == value else str(value)
        tracks.append(track)
        for point in player['positions']:
            x, y = point.get('pitch_x'), point.get('pitch_y')
            if x is None or y is None or not all(math.isfinite(v) for v in (x,y)) or abs(x)>52.5 or abs(y)>34:
                excluded += 1
                continue
            frame, time = int(point['frame']), float(point['time'])
            if frame < 0 or frame != point['frame'] or not math.isfinite(time) or abs(time-frame/fps)>1e-6:
                raise ValueError('Invalid nominal frame timestamp')
            if not 0 <= time < duration:
                raise ValueError('Heatmap currently requires a segment starting at frame zero')
            if (track,frame) in seen:
                raise ValueError('Duplicate track/frame would double-count occupancy')
            seen.add((track,frame))
            samples.append([x,y,time,frame,track,point.get('bbox_truncated') is not False])
    return {'fps':fps,'duration':duration,'name':tracking['video']['name'],
            'tracks':sorted(set(tracks),key=lambda v:(not v.isdigit(),int(v) if v.isdigit() else v)),
            'samples':samples,'excluded_invalid_or_outside':excluded}


def build(source, output):
    data = prepare_data(json.loads(source.read_text(encoding='utf-8')))
    root = Path(__file__).resolve().parents[1] / 'sn_gamestate/custom_video'
    html = (root/'heatmap.html').read_text(encoding='utf-8')
    payload = json.dumps(data,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    html = html.replace('__MATH__',(root/'heatmap.js').read_text(encoding='utf-8')).replace('__DATA__',payload)
    review_video = source.parent / 'review.mp4'
    if not review_video.exists():
        review_video = Path(__file__).resolve().parents[1] / 'outputs' / 'match_20s_improved_final' / 'review.mp4'
    if not review_video.exists():
        raise FileNotFoundError(f'Review video not found: {review_video}')
    video_url = quote(Path(os.path.relpath(review_video.resolve(), output.resolve())).as_posix(),safe='/')
    html = html.replace('__VIDEO_URL__',video_url)
    # Regeneration is intentional: the page is a derived artifact and may be
    # refreshed after the renderer is improved.
    output.mkdir(parents=True,exist_ok=True)
    (output/'index.html').write_text(html,encoding='utf-8')
    report={'source':str(source.resolve()),'analyzed_seconds':data['duration'],
            'in_field_samples':len(data['samples']),'excluded_invalid_or_outside':data['excluded_invalid_or_outside'],
            'default_samples':sum(not p[5] for p in data['samples']),
            'weight_seconds_per_sample':1/data['fps'],'quality_verified':False}
    (output/'heatmap_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracking',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    build(args.tracking,args.output)
