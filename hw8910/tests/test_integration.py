"""Real subprocesses, isolated directories and localhost HTTP; no model/network.
PROTOTYPE_ROOT selects original commit with identical launch interface.
"""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(os.environ.get('PROTOTYPE_ROOT', Path(__file__).resolve().parents[1])).resolve()


def http(port, route, body=None):
    request = Request(f'http://127.0.0.1:{port}{route}',
                      data=None if body is None else json.dumps(body).encode(),
                      headers={'Content-Type': 'application/json'})
    try:
        with urlopen(request, timeout=30) as r:
            result = r.status, json.load(r)
            record_http(route, body, result)
            return result
    except HTTPError as r:
        result = r.code, json.load(r)
        record_http(route, body, result)
        return result

def record_http(route, body, result):
    destination = os.environ.get('HTTP_EVIDENCE')
    if destination:
        # Polling while warming up can be frequent; record diagnostics and generations.
        if route not in ('/generate','/debug'): return
        with Path(destination).open('a') as f:
            f.write(json.dumps({'route':route,'request':body,'status':result[0],'response':result[1]},ensure_ascii=False)+'\n')


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name) / 'workspace'
        self.cli('init', success=True)

    def cli(self, *args, success=None):
        r = subprocess.run([sys.executable, str(ROOT/'lab.py'), '--workspace', str(self.work), *args],
                           cwd=self.tmp.name, capture_output=True, text=True, timeout=40)
        if success is True:
            self.assertEqual(r.returncode, 0, r.stdout+r.stderr)
        elif success is False:
            self.assertNotEqual(r.returncode, 0, r.stdout+r.stderr)
        return r

    @contextlib.contextmanager
    def server(self, warmup=0):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
        with tempfile.TemporaryFile(mode='w+') as log:
            p = subprocess.Popen([sys.executable, str(ROOT/'service.py'), '--workspace', str(self.work),
                                  '--port', str(port), '--warmup-seconds', str(warmup)],
                                 cwd=self.tmp.name, stdout=log, stderr=log)
            try:
                deadline = time.monotonic()+20
                while True:
                    try:
                        if http(port, '/health')[0] == 200:
                            break
                    except (URLError, OSError):
                        pass
                    if p.poll() is not None or time.monotonic() > deadline:
                        log.seek(0); self.fail('Server failed: '+log.read())
                    time.sleep(.03)
                yield port
            finally:
                p.terminate()
                try: p.wait(timeout=8)
                except subprocess.TimeoutExpired: p.kill(); p.wait()

    def test_cache_recovery_and_changed_bytes(self):
        self.cli('run', '--fail-after-prepare', success=False)
        r = self.cli('run', success=True)
        self.assertIn('CACHE_HIT', r.stdout)
        (self.work/'input.txt').write_text('Changed real input\n')
        r = self.cli('run', success=True)
        self.assertIn('PREPARED', r.stdout)
        old = json.loads((self.work/'config.json').read_text())
        for key, value in [('max_length',256),('tokenizer_revision','r2'),('code_revision','v2'),('template_sha256','f'*64)]:
            old[key] = value
            (self.work/'config.json').write_text(json.dumps(old))
            self.assertIn('PREPARED', self.cli('run',success=True).stdout)

    def test_reports_fail_closed_and_preserve_bytes(self):
        before = (self.work/'accepted.json').read_bytes()
        for report in ['report-reject.json', 'report-incomplete.json', 'report-A.json']:
            with self.subTest(report=report):
                r = self.cli('run', '--report', report)
                self.assertNotEqual(r.returncode, 0, 'Bad report accepted: '+report)
                after = (self.work/'accepted.json').read_bytes()
                self.assertEqual(hashlib.sha256(before).hexdigest(), hashlib.sha256(after).hexdigest())
                self.assertEqual(before, after)

    def test_repeat_across_processes_one_event(self):
        self.cli('run', success=True)
        before = (self.work/'accepted.json').read_bytes()
        self.cli('run', success=True)
        self.assertEqual((self.work/'accepted.json').read_bytes(), before)
        self.assertEqual(len((self.work/'publications.jsonl').read_text().splitlines()), 1)

    def test_http_readiness_blocks_backend(self):
        with self.server(warmup=3) as port:
            self.assertEqual(http(port,'/health')[0], 200)
            self.assertEqual(http(port,'/ready')[0], 503)
            self.assertEqual(http(port,'/generate',{'prompt':'test'})[0],503)
            self.assertEqual(http(port,'/debug')[1]['calls'],0)
            deadline=time.monotonic()+8
            while http(port,'/ready')[0] != 200 and time.monotonic()<deadline: time.sleep(.05)
            self.assertEqual(http(port,'/ready')[0],200)
            self.assertTrue(http(port,'/debug')[1]['warmup_finished'])
            self.assertEqual(http(port,'/generate',{'prompt':'test'})[0],200)
            self.assertEqual(http(port,'/debug')[1]['calls'],1)

    def test_http_loaded_version_restart_and_rollback(self):
        with self.server() as port:
            self.assertEqual(http(port,'/generate',{'prompt':'test'})[1]['model_id'],'fixture-A')
            self.cli('run', success=True)
            answer=http(port,'/generate',{'prompt':'test'})[1]
            self.assertIn('fixture-A',answer['answer'])
            self.assertEqual(answer['model_id'],'fixture-A')
        with self.server() as port:
            answer=http(port,'/generate',{'prompt':'test'})[1]
            self.assertIn('fixture-B',answer['answer']);self.assertEqual(answer['model_id'],'fixture-B')
        self.cli('run','--manifest','manifest-A.json','--report','report-A.json',success=True)
        with self.server() as port:
            answer=http(port,'/generate',{'prompt':'test'})[1]
            self.assertIn('fixture-A',answer['answer']);self.assertEqual(answer['model_id'],'fixture-A')

    def test_http_invalid_requests_never_call_backend(self):
        with self.server() as port:
            before=http(port,'/debug')[1]['calls']
            for body in [{'prompt':''},{'prompt':'x','max_new_tokens':0},{'prompt':'x','max_new_tokens':100000},
                         {'prompt':'x','max_new_tokens':'16'},{'prompt':'x','max_new_tokens':True},[]]:
                with self.subTest(body=body):
                    self.assertIn(http(port,'/generate',body)[0],range(400,500))
                    self.assertEqual(http(port,'/debug')[1]['calls'],before)
            status,answer=http(port,'/generate',{'prompt':'valid','max_new_tokens':32})
            self.assertEqual(status,200);self.assertIn('STUB ONLY',answer['answer'])
            self.assertEqual(http(port,'/debug')[1]['calls'],before+1)

    def test_atomic_disk_failure_keeps_old_pointer(self):
        # Execute mock in an independent interpreter, against selected source.
        code='''from pathlib import Path
import json, release
from unittest.mock import patch
r=Path.cwd(); before=(r/'accepted.json').read_bytes()
m=json.loads((r/'manifest-B.json').read_text()); e=json.loads((r/'report-B.json').read_text())
with patch('os.replace',side_effect=OSError('injected disk error')):
    try: release.publish(m,e,r/'accepted.json')
    except OSError: pass
    else: raise AssertionError('Expected atomic replace to fail')
assert (r/'accepted.json').read_bytes()==before
assert not list(r.glob('*.tmp'))
'''
        r=subprocess.run([sys.executable,'-c',code],cwd=self.work,
                         env={**os.environ,'PYTHONPATH':str(ROOT)},capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)


if __name__=='__main__': unittest.main(verbosity=2)
