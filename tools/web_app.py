# -*- coding: utf-8 -*-
"""Local web app: upload one football video and receive tracking + heatmap results.

Run with the project's inference Python, e.g.:
    .venv\Scripts\python.exe tools\web_app.py
Then open http://127.0.0.1:8001 .  This server intentionally handles one job at a
time because the upstream TrackLab adapter changes its process working directory.
"""

import cgi
import argparse
import gc
import json
import math
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import gc
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
JOBS = Path(os.environ.get('WEB_JOBS_DIR', ROOT / 'outputs' / 'web_jobs')).resolve()
MODEL_DIR = Path(os.environ.get('MODEL_DIR', ROOT / 'pretrained_models')).resolve()
# MAX_JOB_MB is the deployment-facing upload limit.  Keep MAX_UPLOAD_BYTES for
# existing deployments, but prefer the simpler MiB setting when both exist.
if 'MAX_JOB_MB' in os.environ:
    MAX_UPLOAD_BYTES = int(os.environ['MAX_JOB_MB']) * 1024 * 1024
else:
    MAX_UPLOAD_BYTES = int(os.environ.get('MAX_UPLOAD_BYTES', str(2 * 1024 * 1024 * 1024)))
if MAX_UPLOAD_BYTES <= 0:
    raise ValueError('MAX_JOB_MB or MAX_UPLOAD_BYTES must be greater than zero')
# A one/two-minute verification run at common broadcast frame rates needs more
# than the old smoke-test ceiling of 250 frames. Operators can lower this for
# constrained deployments without changing code.
MAX_FRAMES = int(os.environ.get('MAX_ANALYSIS_FRAMES', '7200'))
if MAX_FRAMES <= 0:
    raise ValueError('MAX_ANALYSIS_FRAMES must be greater than zero')
MAX_JOB_RSS_BYTES = int(os.environ.get('MAX_JOB_RSS_MB', '380')) * 1024 * 1024
ALLOWED_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv'}
# Comma-separated browser origins that may call this API.  Set this to the
# production Vercel URL before publishing the tunnel.  It intentionally has no
# wildcard default: this endpoint accepts large, expensive video jobs.
ALLOWED_ORIGINS = {origin.strip().rstrip('/') for origin in
                   os.environ.get('ALLOWED_ORIGINS', 'http://localhost:3000,http://127.0.0.1:3000').split(',')
                   if origin.strip()}
LOCK = threading.Lock()
PROCESS_LOCK = threading.Lock()
STATE = {}


def job_paths(job_id):
    folder = JOBS / job_id
    return {'folder': folder, 'upload': folder / 'upload.mp4', 'run': folder / 'run',
            'review': folder / 'review', 'heatmap': folder / 'heatmap'}


def workspace_scenes(job_id):
    """Convert the pipeline's tracking export to the publishing UI contract."""
    tracking_path = job_paths(job_id)['review'] / 'tracking.json'
    try:
        tracking = json.loads(tracking_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return []
    source_scenes = tracking.get('scenes') if isinstance(tracking, dict) else None
    players = tracking.get('players') if isinstance(tracking, dict) else None
    if not isinstance(source_scenes, list) or not isinstance(players, list):
        return []

    tracks_by_scene = {}
    for player in players:
        if not isinstance(player, dict):
            continue
        scene_id = player.get('segment_id')
        track_id = player.get('track_id')
        if scene_id is None or track_id is None:
            continue
        points = []
        for position in player.get('positions', []):
            if not isinstance(position, dict) or not position.get('usable_for_heatmap'):
                continue
            try:
                x, y, timestamp = (float(position['pitch_x']), float(position['pitch_y']),
                                   float(position['time']))
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(value) for value in (x, y, timestamp)):
                points.append({'x': x, 'y': y, 'timestamp': timestamp})
        if points:
            points.sort(key=lambda point: point['timestamp'])
            tracks_by_scene.setdefault(str(scene_id), []).append({
                'trackId': str(track_id), 'from': points[0]['timestamp'],
                'to': points[-1]['timestamp'], 'points': points,
            })

    scenes = []
    for number, scene in enumerate(source_scenes, start=1):
        if not isinstance(scene, dict):
            continue
        scene_id = scene.get('segment_id')
        if scene_id is None:
            continue
        tracks = tracks_by_scene.get(str(scene_id), [])
        if not tracks:
            continue
        try:
            start, end = float(scene['start_time']), float(scene['end_time'])
        except (KeyError, TypeError, ValueError):
            start = min(track['from'] for track in tracks)
            end = max(track['to'] for track in tracks)
        scenes.append({'sceneId': str(scene_id), 'sceneNumber': number,
                       'from': start, 'to': end, 'tracks': tracks})
    return scenes


def public_state(job_id):
    with LOCK:
        state = dict(STATE.get(job_id, {}))
    if not state:
        return None
    state.pop('log_path', None)
    if state.get('status') == 'complete':
        state['scenes'] = workspace_scenes(job_id)
    return state


def process_rss_bytes(pid):
    """Read a process's RSS without requiring an additional deployment dependency.

    Falls back to psutil when /proc is unavailable, e.g. on a Windows host.
    """
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        return psutil.Process(pid).memory_info().rss
    except Exception:
        return None


def log_rss(log, label, pid):
    rss = process_rss_bytes(pid)
    if rss is not None:
        log.write(f'[memory] {label}: RSS={rss / 1024 / 1024:.1f} MiB\n')
    return rss


def log_memory(state, phase):
    """Log this web server's own RSS without adding a heavyweight runtime dependency."""
    rss = process_rss_bytes(os.getpid())
    message = (f'[RAM] phase={phase} rss_mb={rss / 1048576:.1f}' if rss is not None
               else f'[RAM] phase={phase} rss=unavailable')
    with Path(state['log_path']).open('a', encoding='utf-8') as log:
        log.write(message + '\n')


def run_command(command, state, label):
    state.update(phase=label, detail='กำลังประมวลผล อาจใช้เวลาหลายนาที…')
    log_path = Path(state['log_path'])
    exceeded = False
    log_memory(state, f'before:{label}')
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n$ {subprocess.list2cmdline(command)}\n')
        child = subprocess.Popen(command, cwd=ROOT, text=True, stdout=log,
                                 stderr=subprocess.STDOUT)
        log_rss(log, f'{label} started', child.pid)
        last_log = time.monotonic()
        while child.poll() is None:
            rss = process_rss_bytes(child.pid)
            if rss is not None and rss > MAX_JOB_RSS_BYTES:
                exceeded = True
                log.write(f'[memory] RSS={rss / 1024 / 1024:.1f} MiB exceeds '
                          f'limit={MAX_JOB_RSS_BYTES / 1024 / 1024:.1f} MiB; terminating job\n')
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                break
            if time.monotonic() - last_log >= 5:
                log_rss(log, label, child.pid)
                last_log = time.monotonic()
            time.sleep(.25)
        returncode = child.wait()
        log_rss(log, f'{label} ended', child.pid)
    log_memory(state, f'after:{label}')
    if returncode:
        if exceeded:
            detail = f'Process RSS exceeded the {MAX_JOB_RSS_BYTES // 1024 // 1024} MiB safety limit'
        else:
            detail = (log_path.read_text(encoding='utf-8')[-4000:] or
                      f'คำสั่งจบด้วย exit code {returncode}: {subprocess.list2cmdline(command)}')
        state.update(status='failed', phase='ไม่สำเร็จ', detail='การประมวลผลหยุดด้วยข้อผิดพลาด',
                     error_detail=detail[-4000:])
        return False
    return True


def process(job_id, stage, device, max_frames, start_seconds):
    paths = job_paths(job_id)
    with LOCK:
        state = STATE[job_id]
    with PROCESS_LOCK:
        try:
            state.update(status='running', phase='กำลังเริ่มงาน', detail='เตรียมโมเดลและวิดีโอ…')
            command = [sys.executable, 'tools/analyze_video.py', '--video', str(paths['upload']),
                       '--output', str(paths['run']), '--model-dir', str(MODEL_DIR),
                       '--stage', stage, '--device', device, '--start-seconds', str(start_seconds)]
            if max_frames:
                command += ['--max-frames', str(max_frames)]
            if not run_command(command, state, 'ตรวจจับและติดตามผู้เล่น'):
                return
            evidence_path = paths['run'] / 'inference_evidence.json'
            if evidence_path.exists():
                evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
                state.update(start_seconds=evidence['start_seconds'], start_frame=evidence['start_frame'],
                             end_frame=evidence['end_frame'], frames_processed=evidence['frames_processed'],
                             quality_summary=evidence.get('quality_summary', {}))
            if not run_command([sys.executable, 'tools/review_coordinates.py', '--run', str(paths['run']),
                                '--output', str(paths['review']), '--clean'], state, 'สร้างวิดีโอผลลัพธ์'):
                return
            if not run_command([sys.executable, 'tools/build_heatmap.py', '--tracking',
                                str(paths['review'] / 'tracking.json'), '--output', str(paths['heatmap'])],
                               state, 'สร้าง heatmap'):
                return
            state.update(status='complete', phase='เสร็จแล้ว', detail='เปิดดูวิดีโอและ heatmap ได้ด้านล่าง',
                         result=f'/jobs/{job_id}/heatmap/index.html', review=f'/jobs/{job_id}/review/review.mp4',
                         log=f'/jobs/{job_id}/pipeline.log')
        except Exception as error:
            with Path(state['log_path']).open('a', encoding='utf-8') as log:
                log.write(f'\nWeb app exception: {error!r}\n')
            state.update(status='failed', phase='ไม่สำเร็จ', detail='เกิดข้อผิดพลาดในเว็บแอป', error_detail=repr(error))
        finally:
            gc.collect()


PAGE = '''<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pitchside · วิเคราะห์วิดีโอ</title><style>
*{box-sizing:border-box}body{margin:0;background:#0d181a;color:#ecf5ef;font:16px system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:42px 22px}h1{font-size:34px;margin:0}p{color:#b9cbc4;line-height:1.6}.box{background:#172728;border:1px solid #294340;border-radius:14px;padding:24px;margin:22px 0}.row{display:flex;gap:18px;align-items:end;flex-wrap:wrap}label{display:grid;gap:7px;color:#cfe2d9;font-size:14px}input,select,button{font:inherit;color:inherit;background:#102022;border:1px solid #54746a;border-radius:7px;padding:10px}input[type=file]{max-width:420px}button{background:#21805e;border:0;font-weight:700;cursor:pointer;padding:11px 20px}.status{border-left:4px solid #67dcb0}.failed{border-color:#ff8d76}iframe{width:100%;height:920px;border:0;border-radius:12px;background:#10191c}pre{white-space:pre-wrap;background:#0c1415;padding:12px;border-radius:7px;max-height:220px;overflow:auto}.hidden{display:none}@media(max-width:650px){main{padding:24px 14px}h1{font-size:27px}iframe{height:1050px}}</style><main><header><h1>Pitchside วิเคราะห์วิดีโอฟุตบอล</h1><p>อัปโหลดคลิป → ตรวจจับ/ติดตาม → สร้างวิดีโอพร้อมพิกัดและ heatmap ในหน้าเดียว</p></header><section class="box"><form id="upload"><div class="row"><label>วิดีโอฟุตบอล<input name="video" type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,.mp4,.mov,.avi,.mkv" required></label><label>โหมดประมวลผล<select name="stage"><option value="mvp">MVP (แนะนำ)</option><option value="calibrate">Calibrate</option><option value="full">Full</option></select></label><label>อุปกรณ์<select name="device"><option value="auto">Auto</option><option value="cuda">CUDA</option><option value="cpu">CPU</option></select></label><label>จำกัดเฟรม (ว่าง=ทั้งคลิป)<input name="max_frames" type="number" min="1" placeholder="เช่น 750"></label><button>อัปโหลดและเริ่ม</button></div></form><p>รองรับ MP4, MOV, AVI, MKV สูงสุด 2 GB · งานจะเข้าคิวทีละรายการบนเครื่องนี้</p></section><section id="progress" class="box status hidden"><b id="phase"></b><p id="detail"></p><pre id="error" class="hidden"></pre></section><section id="results" class="hidden"><h2>ผลลัพธ์</h2><p><a id="review" target="_blank">เปิดวิดีโอผลลัพธ์</a></p><iframe id="heatmap" title="Heatmap result"></iframe></section></main><script>
const form=document.querySelector('#upload'),progress=document.querySelector('#progress'),phase=document.querySelector('#phase'),detail=document.querySelector('#detail'),error=document.querySelector('#error'),results=document.querySelector('#results');let timer;
form.onsubmit=async e=>{e.preventDefault();const b=form.querySelector('button');b.disabled=true;progress.className='box status';detail.textContent='กำลังอัปโหลด…';try{const r=await fetch('/api/jobs',{method:'POST',body:new FormData(form)}),d=await r.json();if(!r.ok)throw new Error(d.error||'อัปโหลดไม่สำเร็จ');poll(d.id)}catch(x){detail.textContent=x.message;b.disabled=false}};
function poll(id){timer=setInterval(async()=>{try{const r=await fetch('/api/jobs/'+id),d=await r.json();if(!r.ok)throw new Error(d.error||'อ่านสถานะงานไม่สำเร็จ');phase.textContent=d.phase;detail.textContent=d.detail;if(d.status==='failed'){clearInterval(timer);progress.classList.add('failed');error.textContent=(d.error_detail||'ไม่มีรายละเอียด')+'\\n\\nดู log: '+(d.log||'ไม่มี log');error.classList.remove('hidden');form.querySelector('button').disabled=false}if(d.status==='complete'){clearInterval(timer);results.classList.remove('hidden');document.querySelector('#review').href=d.review;document.querySelector('#heatmap').src=d.result;form.querySelector('button').disabled=false}}catch(x){clearInterval(timer);progress.classList.add('failed');error.textContent=x.message;error.classList.remove('hidden');form.querySelector('button').disabled=false}},1500)}
</script></html>'''


class App(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(fmt % args)

    def end_headers(self):
        origin = self.headers.get('Origin', '').rstrip('/')
        if origin in ALLOWED_ORIGINS:
            self.send_header('Access-Control-Allow-Origin', origin)
            self.send_header('Vary', 'Origin')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        if self.headers.get('Origin', '').rstrip('/') not in ALLOWED_ORIGINS:
            self.send_error(403, 'Origin is not allowed')
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.end_headers()

    def send_json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/healthz':
            self.send_json({'status': 'ok'})
            return
        if parsed.path == '/':
            body = PAGE.encode(); self.send_response(200); self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body); return
        if parsed.path.startswith('/api/jobs/'):
            state = public_state(parsed.path.rsplit('/', 1)[-1])
            self.send_json(state or {'error': 'ไม่พบงาน'}, 200 if state else 404); return
        if parsed.path.startswith('/jobs/'):
            target = (JOBS / parsed.path.removeprefix('/jobs/')).resolve()
            if JOBS.resolve() not in target.parents or not target.is_file():
                self.send_error(404); return
            content_type = mimetypes.guess_type(target.name)[0] or 'application/octet-stream'
            self.send_response(200); self.send_header('Content-Type', content_type); self.send_header('Content-Length', str(target.stat().st_size)); self.end_headers()
            with target.open('rb') as source: shutil.copyfileobj(source, self.wfile)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != '/api/jobs': self.send_error(404); return
        length = int(self.headers.get('Content-Length', '0'))
        if not 0 < length <= MAX_UPLOAD_BYTES:
            limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
            self.send_json({'error': f'ไฟล์ว่างหรือใหญ่เกิน {limit_mb} MB'}, 413); return
        form = cgi.FieldStorage(fp=self.rfile, headers=self.headers, environ={'REQUEST_METHOD': 'POST', 'CONTENT_TYPE': self.headers['Content-Type']})
        if 'video' not in form or not getattr(form['video'], 'file', None): self.send_json({'error': 'กรุณาเลือกวิดีโอ'}, 400); return
        upload = form['video']; suffix = Path(upload.filename or '').suffix.lower()
        if suffix not in ALLOWED_SUFFIXES: self.send_json({'error': 'รองรับเฉพาะ MP4, MOV, AVI, MKV'}, 400); return
        stage = form.getfirst('stage', 'mvp'); device = form.getfirst('device', 'auto')
        if stage not in {'mvp', 'calibrate', 'full'} or device not in {'auto', 'cpu', 'cuda'}: self.send_json({'error': 'ตัวเลือกไม่ถูกต้อง'}, 400); return
        try: start_seconds = float(form.getfirst('start_seconds') or 0)
        except ValueError: self.send_json({'error': 'start_seconds must be a number'}, 400); return
        if not math.isfinite(start_seconds) or start_seconds < 0:
            self.send_json({'error': 'start_seconds must be finite and greater than or equal to zero'}, 400); return
        try: max_frames = int(form.getfirst('max_frames') or MAX_FRAMES)
        except ValueError: self.send_json({'error': 'จำนวนเฟรมต้องเป็นตัวเลข'}, 400); return
        if not 1 <= max_frames <= MAX_FRAMES:
            self.send_json({'error': f'จำกัดการประมวลผลไว้ 1-{MAX_FRAMES} เฟรม'}, 400); return
        job_id = uuid.uuid4().hex[:12]; paths = job_paths(job_id); paths['folder'].mkdir(parents=True)
        with paths['upload'].open('wb') as destination: shutil.copyfileobj(upload.file, destination)
        with LOCK:
            STATE[job_id] = {'id': job_id, 'status': 'queued', 'phase': 'รอคิว',
                             'detail': 'กำลังเตรียมงาน…',
                             'log_path': str(paths['folder'] / 'pipeline.log'),
                             'log': f'/jobs/{job_id}/pipeline.log', 'start_seconds': start_seconds,
                             'max_frames': max_frames}
        threading.Thread(target=process, args=(job_id, stage, device, max_frames, start_seconds), daemon=True).start()
        self.send_json({'id': job_id}, HTTPStatus.ACCEPTED)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8001')),
                        help='HTTP port (default: PORT environment variable or 8001)')
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('--port must be between 1024 and 65535')
    JOBS.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Serving on http://0.0.0.0:{args.port}')
    ThreadingHTTPServer(('0.0.0.0', args.port), App).serve_forever()
