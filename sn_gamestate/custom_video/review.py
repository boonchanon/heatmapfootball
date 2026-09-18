"""Readable video review. Trails use observed points only, never gap filling."""

from collections import deque
import colorsys
import json
from pathlib import Path
import subprocess
import zipfile

import cv2
import numpy as np
import pandas as pd

from .export import export_tracking, _finite_pitch


def make_browser_video(raw_video, output_video):
    """Encode the OpenCV intermediate as universally playable H.264 MP4."""
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError('imageio-ffmpeg is required to create a browser-playable review video') from error
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(raw_video), '-map', '0:v:0',
               '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '23', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output_video)]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f'H.264 encoding failed: {(result.stderr or result.stdout)[-2000:]}')
    if not output_video.is_file() or output_video.stat().st_size == 0:
        raise RuntimeError('H.264 encoder did not produce a review video')


def track_color(track):
    rgb = colorsys.hsv_to_rgb((track * .61803398875) % 1, .65, 1.)
    return tuple(round(v * 255) for v in rgb[::-1])


def trail_points(history, frame, point, max_points):
    """Clear at gaps or very large jumps; never draw a bridge across missing data."""
    if history and (history[-1][0] != frame - 1
                    or np.linalg.norm(np.array(point) - history[-1][1]) > 3.):
        history.clear()
    history.append((frame, point))
    while len(history) > max_points:
        history.popleft()
    return [p for _, p in history]


def clean_review(run, output, video_path):
    run, output = Path(run), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(run / 'tracker_state.pklz') as archive:
        detections = pd.read_pickle(archive.open('0.pkl'))
        images = pd.read_pickle(archive.open('0_image.pkl')).sort_values('frame')
    evidence = json.loads((run / 'inference_evidence.json').read_text())
    video = evidence['video']
    _, metadata = export_tracking(detections, images, video, output)
    fps = video['fps']
    duration = len(images) / fps
    raw_video = output / 'review_raw.mp4'
    browser_video = output / 'review.mp4'
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f'Cannot open source video for review: {video_path}')
    writer = cv2.VideoWriter(str(raw_video), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 900))
    if not writer.isOpened():
        raise RuntimeError('Cannot open MP4 encoder')
    histories = {}
    def text(canvas, value, point, scale=.65, color=(220, 225, 230), thickness=1):
        cv2.putText(canvas, value, point, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
    def pitch_point(x, y):
        return round(1598 + x * 5), round(410 + y * 5)
    try:
        for sequence, (image_id, image) in enumerate(images.iterrows()):
            ok, raw = capture.read()
            if not ok:
                raise ValueError(f'Cannot decode source frame {sequence}')
            if raw.shape[1] > video['width']:
                raw = cv2.resize(raw, (video['width'], video['height']), interpolation=cv2.INTER_AREA)
            canvas = np.full((900, 1920, 3), (23, 21, 18), dtype=np.uint8)
            # Fit without distortion, including videos that are not 16:9.
            scale = min(1280/raw.shape[1], 720/raw.shape[0])
            w, h = round(raw.shape[1]*scale), round(raw.shape[0]*scale)
            ox, oy = (1280-w)//2, 80+(720-h)//2
            canvas[oy:oy+h, ox:ox+w] = cv2.resize(raw, (w, h))
            text(canvas, 'MATCH REVIEW', (24, 43), 1., thickness=2)
            text(canvas, f'{sequence/fps:05.2f} / {duration:.2f} s', (1010, 43), .8)
            text(canvas, 'FIELD VIEW', (1338, 110), .95, thickness=2)
            text(canvas, 'Colors identify tracks, not teams', (1338, 145), .55)
            cv2.rectangle(canvas, pitch_point(-52.5, -34), pitch_point(52.5, 34), (43, 76, 42), -1)
            white = (185, 203, 191)
            for xa, ya, xb, yb in [(-52.5,-34,52.5,34),(-52.5,-20.16,-36,20.16),
                                   (36,-20.16,52.5,20.16),(-52.5,-9.16,-47,9.16),(47,-9.16,52.5,9.16)]:
                cv2.rectangle(canvas, pitch_point(xa,ya), pitch_point(xb,yb), white, 1)
            cv2.line(canvas,pitch_point(0,-34),pitch_point(0,34),white,1)
            cv2.circle(canvas,pitch_point(0,0),round(9.15*5),white,1)
            rows = detections[detections.image_id == image_id]
            outside, mapped, untracked, unavailable = 0, 0, 0, 0
            occupied, image_labels = [], []
            # Reserve the actual markers before placing labels so later points
            # cannot cover an earlier ID. Marker locations are never displaced.
            for _, row in rows.iterrows():
                p = _finite_pitch(row.get('bbox_pitch'))
                if pd.notna(row.track_id) and p and abs(p['pitch_x']) <= 52.5 and abs(p['pitch_y']) <= 34:
                    px, py = pitch_point(p['pitch_x'], p['pitch_y'])
                    occupied.append((px-6, py-6, px+6, py+6))
            for _, row in rows.iterrows():
                track = None if pd.isna(row.track_id) else int(row.track_id)
                point = _finite_pitch(row.get('bbox_pitch'))
                inside = point is not None and abs(point['pitch_x']) <= 52.5 and abs(point['pitch_y']) <= 34
                is_outside = point is not None and not inside
                outside += int(is_outside)
                untracked += int(track is None)
                unavailable += int(point is None)
                color = track_color(track) if track is not None and inside else (150, 150, 150)
                l,t,bw,bh = map(float,row.bbox_ltwh)
                a = (round(ox+l*scale),round(oy+t*scale))
                b = (round(ox+(l+bw)*scale),round(oy+(t+bh)*scale))
                cv2.rectangle(canvas, a,b,color,1)
                # Border-truncated boxes remain visible; they are not reliable foot positions.
                truncated = l <= 0 or t <= 0 or l+bw >= raw.shape[1]-1 or t+bh >= raw.shape[0]-1
                label = str(track) if track is not None else '?'
                if is_outside:
                    label = 'OUT '+label
                if truncated:
                    label += '*'
                size = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.58,2)[0]
                for lx,ly in [(a[0],a[1]-5),(b[0]+4,a[1]+17),
                              (a[0]-size[0]-11,a[1]+17),(a[0],b[1]+21)]:
                    lx = max(0,min(lx,1270-size[0]))
                    ly = max(oy+22,min(ly,oy+h-5))
                    rect = (lx,ly-20,lx+size[0]+7,ly+4)
                    if not any(rect[0]<r[2] and rect[2]>r[0] and rect[1]<r[3] and rect[3]>r[1] for r in image_labels):
                        break
                image_labels.append(rect)
                cv2.rectangle(canvas,(lx,ly-20),(lx+size[0]+7,ly+4),(25,25,25),-1)
                text(canvas,label,(lx+3,ly),.58,color,2)
                if not inside or track is None:
                    if track is not None:
                        histories.pop(track,None)
                    continue
                mapped += 1
                native = (point['pitch_x'],point['pitch_y'])
                history = histories.setdefault(track,deque())
                trail = trail_points(history,int(image.frame),native,max(2,round(fps*.4)))
                if len(trail)>1:
                    cv2.polylines(canvas,[np.array([pitch_point(*p) for p in trail],dtype=np.int32)],False,tuple(v//2 for v in color),1,cv2.LINE_AA)
                pos = pitch_point(*native)
                cv2.circle(canvas,pos,6,(15,20,15),-1,cv2.LINE_AA)
                cv2.circle(canvas,pos,4,color,-1,cv2.LINE_AA)
                # Separate neighboring ID labels without moving the actual pitch points.
                for dx,dy in [(8,-6),(8,20),(-32,-6),(-32,20),(8,-28),(8,42),(-32,-28),(-32,42)]:
                    rect=(pos[0]+dx,pos[1]+dy-14,pos[0]+dx+28,pos[1]+dy+4)
                    if not any(rect[0]<r[2] and rect[2]>r[0] and rect[1]<r[3] and rect[3]>r[1] for r in occupied):
                        break
                occupied.append(rect)
                cv2.rectangle(canvas,(rect[0],rect[1]),(rect[2],rect[3]),(25,35,25),-1)
                text(canvas,str(track),(rect[0]+2,rect[3]-3),.5,color,1)
            for i,line in enumerate([f'{mapped} tracked points on field',
                                     f'{outside} outside field | {untracked} without ID',
                                     f'{unavailable} missing projections',
                                     'Outside points remain in the JSON.',
                                     'Trails show 0.4 s of observed positions.',
                                     'Identity and calibration are unverified.']):
                text(canvas,line,(1338,640+i*32),.55)
            text(canvas, f'Frame {sequence+1} / {len(images)}   |   * Box touches image border', (24,835),.65)
            text(canvas, 'Track IDs can change after occlusion or leaving the image. No positions are filled across gaps.', (24,871),.52)
            writer.write(canvas)
            if sequence in {0,len(images)//4,len(images)//2,3*len(images)//4,len(images)-1}:
                cv2.imwrite(str(output/f'frame_{int(image.frame):03d}.jpg'),canvas)
    finally:
        writer.release()
        capture.release()
    make_browser_video(raw_video, browser_video)
    raw_video.unlink()
    metadata.update(review_layout='clean',review_width=1920,review_height=900,
                    minimap_filter='tracked, finite, inside canonical pitch; raw JSON retained',
                    trails='0.4 s, reset on gaps or >3 m consecutive displacement')
    (output/'review_summary.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))
