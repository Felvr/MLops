"""Reuse one local orchestration server across independent flow processes."""
from contextlib import contextmanager
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen
ROOT=Path(__file__).resolve().parents[1]


@contextmanager
def local_server():
    with socket.socket() as s: s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    url=f'http://127.0.0.1:{port}/api'
    previous={name:os.environ.get(name) for name in ('PREFECT_API_URL','PREFECT_HOME','PREFECT_SERVER_ANALYTICS_ENABLED')}
    os.environ.update(PREFECT_API_URL=url,PREFECT_HOME=str(ROOT/'.prefect'),PREFECT_SERVER_ANALYTICS_ENABLED='false')
    with (ROOT/'docs/prefect-server.log').open('a') as log:
        proc=subprocess.Popen([sys.executable,'-m','prefect','server','start','--host','127.0.0.1','--port',str(port)],
                              cwd=ROOT,stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+120
            while True:
                try:
                    with urlopen(url+'/health',timeout=2) as r:
                        if r.status==200:break
                except (URLError,OSError):pass
                if proc.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Local Prefect startup failed; inspect prefect-server.log')
                time.sleep(.2)
            yield
        finally:
            proc.terminate()
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            for name,value in previous.items():
                if value is None:os.environ.pop(name,None)
                else:os.environ[name]=value
