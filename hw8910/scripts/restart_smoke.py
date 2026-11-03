"""Confirm the real accepted model is actually loaded again after process restart."""
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark import get, ROOT
from urllib.error import URLError


def main():
    expected=json.loads((ROOT/'accepted.json').read_text())
    with socket.socket() as s: s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    with (ROOT/'docs/real-restart-server.log').open('w') as log:
        p=subprocess.Popen([sys.executable,str(ROOT/'service.py'),'--workspace',str(ROOT),'--backend','transformers',
                            '--warmup-seconds','0','--debug','--port',str(port)],stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+180
            while True:
                try:
                    ready=get(port,'/ready')
                    if ready[0]==200: break
                    d=get(port,'/debug')[1]
                    if d['error']:raise RuntimeError(d['error'])
                except URLError:pass
                if p.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Load failed')
                time.sleep(.2)
            prompt=json.loads((ROOT/'evidence/golden.jsonl').read_text().splitlines()[0])['input']
            status,result=get(port,'/generate',{'prompt':prompt})
            assert status==200 and result['model_id']==expected['model_id'] and result['runtime_sha256']==expected['runtime_sha256']
            previous=json.loads((ROOT/'docs/real-api.json').read_text())
            before=next(row['result'][1] for row in previous if row['step']=='real_smoke')
            assert before['model_id']==result['model_id'] and before['runtime_sha256']==result['runtime_sha256']
            (ROOT/'docs/real-restart.json').write_text(json.dumps({'ready':ready,'response':result,
                'same_loaded_model_as_previous_process':True},ensure_ascii=False,indent=2))
        finally:
            p.terminate()
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()


if __name__=='__main__':main()
