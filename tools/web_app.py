# -*- coding: utf-8 -*-
"""Local web app: upload one football video and receive tracking + heatmap results.

Run with the project's inference Python, e.g.:
    .venv\Scripts\python.exe tools\web_app.py
Then open http://127.0.0.1:8001 .  This server intentionally handles one job at a
time because the upstream TrackLab adapter changes its process working directory.
"""

import cgi
import argparse
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
JOBS = Path(os.environ.get('WEB_JOBS_DIR', ROOT / 'outputs' / 'web_jobs')).resolve()
MODEL_DIR = Path(os.environ.get('MODEL_DIR', ROOT / 'pretrained_models')).resolve()
MAX_UPLOAD_BYTES = int(os.environ.get('MAX_UPLOAD_BYTES', str(2 * 1024 * 1024 * 1024)))
ALLOWED_SUFFIXES = {'.mp4', '.mov', '.avi', '.mkv'}
LOCK = threading.Lock()
PROCESS_LOCK = threading.Lock()
STATE = {}


def job_paths(job_id):
    folder = JOBS / job_id
    return {'folder': folder, 'upload': folder / 'upload.mp4', 'run': folder / 'run',
            'review': folder / 'review', 'heatmap': folder / 'heatmap'}


def public_state(job_id):
    with LOCK:
        state = dict(STATE.get(job_id, {}))
    if not state:
        return None
    state.pop('log_path', None)
    return state


def run_command(command, state, label):
    state.update(phase=label, detail='กำลังประมวลผล อาจใช้เวลาหลายนาที…')
    log_path = Path(state['log_path'])
    with log_path.open('a', encoding='utf-8') as log:
        log.write(f'\n$ {subprocess.list2cmdline(command)}\n')
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    with log_path.open('a', encoding='utf-8') as log:
        log.write(result.stdout or '')
        log.write(result.stderr or '')
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        if not detail:
            detail = f'คำสั่งจบด้วย exit code {result.returncode}: {subprocess.list2cmdline(command)}'
        state.update(status='failed', phase='ไม่สำเร็จ', detail='การประมวลผลหยุดด้วยข้อผิดพลาด',
                     error_detail=detail[-4000:])
        return False
    return True


def process(job_id, stage, device, max_frames):
    paths = job_paths(job_id)
    with LOCK:
        state = STATE[job_id]
    with PROCESS_LOCK:
        try:
            state.update(status='running', phase='กำลังเริ่มงาน', detail='เตรียมโมเดลและวิดีโอ…')
            command = [sys.executable, 'tools/analyze_video.py', '--video', str(paths['upload']),
                       '--output', str(paths['run']), '--model-dir', str(MODEL_DIR),
                       '--stage', stage, '--device', device]
            if max_frames:
                command += ['--max-frames', str(max_frames)]
            if not run_command(command, state, 'ตรวจจับและติดตามผู้เล่น'):
                return
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


PAGE = '''<!doctype html><html lang="th"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pitchside · วิเคราะห์วิดีโอ</title><style>
*{box-sizing:border-box}body{margin:0;background:#0d181a;color:#ecf5ef;font:16px system-ui,sans-serif}main{max-width:1050px;margin:auto;padding:42px 22px}h1{font-size:34px;margin:0}p{color:#b9cbc4;line-height:1.6}.box{background:#172728;border:1px solid #294340;border-radius:14px;padding:24px;margin:22px 0}.row{display:flex;gap:18px;align-items:end;flex-wrap:wrap}label{display:grid;gap:7px;color:#cfe2d9;font-size:14px}input,select,button{font:inherit;color:inherit;background:#102022;border:1px solid #54746a;border-radius:7px;padding:10px}input[type=file]{max-width:420px}button{background:#21805e;border:0;font-weight:700;cursor:pointer;padding:11px 20px}.status{border-left:4px solid #67dcb0}.failed{border-color:#ff8d76}iframe{width:100%;height:920px;border:0;border-radius:12px;background:#10191c}pre{white-space:pre-wrap;background:#0c1415;padding:12px;border-radius:7px;max-height:220px;overflow:auto}.hidden{display:none}@media(max-width:650px){main{padding:24px 14px}h1{font-size:27px}iframe{height:1050px}}</style><main><header><h1>Pitchside วิเคราะห์วิดีโอฟุตบอล</h1><p>อัปโหลดคลิป → ตรวจจับ/ติดตาม → สร้างวิดีโอพร้อมพิกัดและ heatmap ในหน้าเดียว</p></header><section class="box"><form id="upload"><div class="row"><label>วิดีโอฟุตบอล<input name="video" type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/x-matroska,.mp4,.mov,.avi,.mkv" required></label><label>โหมดประมวลผล<select name="stage"><option value="mvp">MVP (แนะนำ)</option><option value="calibrate">Calibrate</option><option value="full">Full</option></select></label><label>อุปกรณ์<select name="device"><option value="auto">Auto</option><option value="cuda">CUDA</option><option value="cpu">CPU</option></select></label><label>จำกัดเฟรม (ว่าง=ทั้งคลิป)<input name="max_frames" type="number" min="1" placeholder="เช่น 750"></label><button>อัปโหลดและเริ่ม</button></div></form><p>รองรับ MP4, MOV, AVI, MKV สูงสุด 2 GB · งานจะเข้าคิวทีละรายการบนเครื่องนี้</p></section><section id="progress" class="box status hidden"><b id="phase"></b><p id="detail"></p><pre id="error" class="hidden"></pre></section><section id="results" class="hidden"><h2>ผลลัพธ์</h2><p><a id="review" target="_blank">เปิดวิดีโอผลลัพธ์</a></p><iframe id="heatmap" title="Heatmap result"></iframe></section></main><script>
const form=document.querySelector('#upload'),progress=document.querySelector('#progress'),phase=document.querySelector('#phase'),detail=document.querySelector('#detail'),error=document.querySelector('#error'),results=document.querySelector('#results');let timer;
form.onsubmit=async e=>{e.preventDefault();const b=form.querySelector('button');b.disabled=true;progress.className='box status';detail.textContent='กำลังอัปโหลด…';try{const r=await fetch('/api/jobs',{method:'POST',body:new FormData(form)}),d=await r.json();if(!r.ok)throw new Error(d.error||'อัปโหลดไม่สำเร็จ');poll(d.id)}catch(x){detail.textContent=x.message;b.disabled=false}};
function poll(id){timer=setInterval(async()=>{try{const r=await fetch('/api/jobs/'+id),d=await r.json();if(!r.ok)throw new Error(d.error||'อ่านสถานะงานไม่สำเร็จ');phase.textContent=d.phase;detail.textContent=d.detail;if(d.status==='failed'){clearInterval(timer);progress.classList.add('failed');error.textContent=(d.error_detail||'ไม่มีรายละเอียด')+'\\n\\nดู log: '+(d.log||'ไม่มี log');error.classList.remove('hidden');form.querySelector('button').disabled=false}if(d.status==='complete'){clearInterval(timer);results.classList.remove('hidden');document.querySelector('#review').href=d.review;document.querySelector('#heatmap').src=d.result;form.querySelector('button').disabled=false}}catch(x){clearInterval(timer);progress.classList.add('failed');error.textContent=x.message;error.classList.remove('hidden');form.querySelector('button').disabled=false}},1500)}
</script></html>'''


class App(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(fmt % args)

    def send_json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status); self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
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
        try: max_frames = int(form.getfirst('max_frames') or 0)
        except ValueError: self.send_json({'error': 'จำนวนเฟรมต้องเป็นตัวเลข'}, 400); return
        job_id = uuid.uuid4().hex[:12]; paths = job_paths(job_id); paths['folder'].mkdir(parents=True)
        with paths['upload'].open('wb') as destination: shutil.copyfileobj(upload.file, destination)
        with LOCK:
            STATE[job_id] = {'id': job_id, 'status': 'queued', 'phase': 'รอคิว',
                             'detail': 'กำลังเตรียมงาน…',
                             'log_path': str(paths['folder'] / 'pipeline.log'),
                             'log': f'/jobs/{job_id}/pipeline.log'}
        threading.Thread(target=process, args=(job_id, stage, device, max_frames), daemon=True).start()
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
