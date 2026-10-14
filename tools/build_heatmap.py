"""Build a standalone offline heatmap page from exported tracking JSON."""

import argparse
import json
import math
import os
from pathlib import Path
from urllib.parse import quote


def track_option_id(value):
    """Use the same canonical ID in selector options and heatmap samples."""
    if isinstance(value, (float, int)) and int(value) == value:
        return str(int(value))
    return str(value)


def prepare_data(tracking):
    fps = float(tracking['video']['fps'])
    processing = tracking['processing']
    start_frame = int(processing.get('start_frame', processing.get('first_frame', 0) or 0))
    processed_duration = float(processing['processed_duration_seconds'])
    fallback_end = (processing.get('last_frame') + 1 if processing.get('last_frame') is not None
                    else start_frame + round(processed_duration * fps))
    end_frame = int(processing.get('end_frame', fallback_end))
    segment_start = start_frame / fps
    segment_end = end_frame / fps
    duration = segment_end
    if not math.isfinite(fps) or fps <= 0 or not math.isfinite(duration) or segment_end <= segment_start:
        raise ValueError('Expected a positive finite FPS and analyzed duration')
    samples, tracks, seen = [], [], set()
    excluded = 0
    for player in tracking['players']:
        value = player['track_id']
        segment = player.get('segment_id')
        track_value = track_option_id(value)
        track = f'{segment}/{track_value}' if segment else track_value
        tracks.append(track)
        for point in player['positions']:
            x, y = point.get('pitch_x'), point.get('pitch_y')
            if not point.get('usable_for_heatmap', True) or x is None or y is None or not all(math.isfinite(v) for v in (x,y)) or abs(x)>52.5 or abs(y)>34:
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
            samples.append([x,y,time,frame,track_value,point.get('bbox_truncated') is not False,segment or 'segment_001'])
    scenes = {}
    track_options = []
    for player in tracking['players']:
        scene = player.get('segment_id') or 'segment_001'
        positions = player.get('positions', [])
        times = [float(point['time']) for point in positions if isinstance(point.get('time'), (int, float)) and math.isfinite(point['time'])]
        if not times:
            continue
        scenes.setdefault(scene, {'id': scene, 'start': min(times), 'end': max(times)})
        track_options.append({'scene': scene, 'id': track_option_id(player['track_id']),
                              'start': min(times), 'end': max(times), 'frames': len(positions)})
    # New exports include all segments, including a shot where no player was
    # tracked.  Prefer that authoritative timeline over inferring it from
    # player positions, which can truncate a scene label in the UI.
    exported_scenes = tracking.get('scenes')
    if isinstance(exported_scenes, list):
        scene_options = [
            {'id': scene['segment_id'], 'start': float(scene['start_time']),
             'end': float(scene['end_time']), 'start_reason': scene.get('start_reason')}
            for scene in exported_scenes
            if isinstance(scene, dict) and scene.get('segment_id')
        ]
    else:
        scene_options = list(scenes.values())
    scene_options.sort(key=lambda item: (item['start'], item['id']))
    return {'fps':fps,'duration':duration,'start_time':segment_start,'name':tracking['video']['name'],
            'tracks':sorted(set(tracks),key=lambda v:(not v.isdigit(),int(v) if v.isdigit() else v)),
            'scenes':scene_options,'track_options':track_options,
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
