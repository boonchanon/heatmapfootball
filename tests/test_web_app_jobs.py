"""Job persistence for tools/web_app.py: jobs survive a restart and can be listed/deleted."""

import importlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class WebAppJobsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.jobs = Path(self.temp.name)
        os.environ['WEB_JOBS_DIR'] = str(self.jobs)
        os.environ.pop('SERVICE_API_TOKEN', None)
        sys.path.insert(0, str(ROOT / 'tools'))
        self.web_app = importlib.reload(importlib.import_module('web_app'))

    def tearDown(self):
        if hasattr(self, 'server'):
            self.server.shutdown()
            self.server.server_close()
        sys.path.remove(str(ROOT / 'tools'))
        self.temp.cleanup()

    def start_server(self):
        self.web_app.load_jobs()
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), self.web_app.App)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f'http://127.0.0.1:{self.server.server_address[1]}'

    def request(self, url, method='GET', headers=None):
        req = urllib.request.Request(url, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(req) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read() or b'{}')

    def make_job(self, job_id, state=None, complete=False):
        folder = self.jobs / job_id
        (folder / 'heatmap').mkdir(parents=True)
        (folder / 'review').mkdir()
        (folder / 'upload.mp4').write_bytes(b'video')
        if complete:
            (folder / 'heatmap' / 'index.html').write_text('<html></html>')
            (folder / 'review' / 'review.mp4').write_bytes(b'review')
        if state is not None:
            (folder / 'state.json').write_text(json.dumps(state), encoding='utf-8')
        return folder

    def test_state_is_written_on_every_update(self):
        folder = self.jobs / 'aaaaaaaaaaaa'
        folder.mkdir()
        state = self.web_app.JobState(folder, {'id': 'aaaaaaaaaaaa', 'status': 'queued'})
        state.update(status='running', phase='x')
        saved = json.loads((folder / 'state.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'running')
        self.assertIn('updated_at', saved)

    def test_restart_restores_jobs_and_fails_interrupted_ones(self):
        self.make_job('111111111111', {'id': '111111111111', 'status': 'complete', 'created_at': 2, 'filename': 'a.mp4'}, complete=True)
        self.make_job('222222222222', {'id': '222222222222', 'status': 'running', 'created_at': 3})
        self.make_job('333333333333', complete=True)  # legacy: no state.json
        base = self.start_server()

        status, body = self.request(f'{base}/api/jobs')
        self.assertEqual(status, 200)
        by_id = {item['id']: item for item in body['items']}
        self.assertEqual(by_id['111111111111']['status'], 'complete')
        self.assertEqual(by_id['222222222222']['status'], 'failed')
        self.assertEqual(by_id['333333333333']['status'], 'complete')
        self.assertTrue(all('log_path' not in item for item in body['items']))
        order = [item['id'] for item in body['items']]
        self.assertLess(order.index('222222222222'), order.index('111111111111'))  # newest first

        status, job = self.request(f'{base}/api/jobs/111111111111')
        self.assertEqual((status, job['status']), (200, 'complete'))

    def test_delete_removes_finished_job_but_not_running_one(self):
        finished = self.make_job('444444444444', {'id': '444444444444', 'status': 'complete'}, complete=True)
        base = self.start_server()
        self.web_app.STATE['555555555555'] = self.web_app.JobState(self.make_job('555555555555'), {'id': '555555555555', 'status': 'running'})

        self.assertEqual(self.request(f'{base}/api/jobs/555555555555', 'DELETE')[0], 409)
        self.assertEqual(self.request(f'{base}/api/jobs/444444444444', 'DELETE')[0], 200)
        self.assertFalse(finished.exists())
        self.assertEqual(self.request(f'{base}/api/jobs/444444444444', 'DELETE')[0], 404)
        self.assertEqual(self.request(f'{base}/api/jobs/..%2F..%2Fetc', 'DELETE')[0], 404)

    def test_progress_percent_follows_pipeline_files(self):
        folder = self.make_job('777777777777')
        state = {'id': '777777777777', 'status': 'running'}
        self.assertEqual(self.web_app.job_progress(state)['percent'], 0)
        (folder / 'run').mkdir()
        (folder / 'run' / 'progress.json').write_text(json.dumps(
            {'stage': 'inference', 'module': 'reid', 'module_index': 1, 'module_count': 5, 'step': 5, 'total': 10}))
        progress = self.web_app.job_progress(state)
        self.assertAlmostEqual(progress['percent'], 6 + 82 * (1.5 / 5), places=1)
        self.assertEqual(progress['module'], 'reid')
        (folder / 'review' / 'progress.json').write_text(json.dumps({'stage': 'review', 'step': 50, 'total': 100}))
        self.assertEqual(self.web_app.job_progress(state)['percent'], 93)
        self.assertEqual(self.web_app.job_progress({'id': '777777777777', 'status': 'complete'})['percent'], 100)

    def test_service_token_protects_list_and_delete(self):
        os.environ['SERVICE_API_TOKEN'] = 'secret'
        self.web_app = importlib.reload(self.web_app)
        self.make_job('666666666666', {'id': '666666666666', 'status': 'complete'}, complete=True)
        base = self.start_server()
        self.assertEqual(self.request(f'{base}/api/jobs')[0], 401)
        self.assertEqual(self.request(f'{base}/api/jobs/666666666666', 'DELETE')[0], 401)
        self.assertEqual(self.request(f'{base}/api/jobs', headers={'X-Service-Token': 'secret'})[0], 200)
        os.environ.pop('SERVICE_API_TOKEN')


if __name__ == '__main__':
    unittest.main()
