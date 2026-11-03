"""Real HTTP smoke + 8 equal prompts per concurrency; retains every attempt."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT=Path(__file__).resolve().parents[1]


def get(port,path,body=None,timeout=120):
    req=Request(f'http://127.0.0.1:{port}{path}',data=json.dumps(body).encode() if body is not None else None,
                headers={'Content-Type':'application/json'})
    try:
        with urlopen(req,timeout=timeout) as r: return r.status,json.load(r)
    except HTTPError as e: return e.code,json.load(e)


def main():
    with socket.socket() as s: s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    log=(ROOT/'docs/real-api-server.log').open('w')
    proc=subprocess.Popen([sys.executable,str(ROOT/'service.py'),'--workspace',str(ROOT),
                           '--backend','transformers','--warmup-seconds','0','--debug','--port',str(port)],
                          cwd=ROOT,stdout=log,stderr=log)
    transcript=[]
    def record(label,value):
        transcript.append({'step':label,'result':value})
        (ROOT/'docs/real-api.json').write_text(json.dumps(transcript,ensure_ascii=False,indent=2))
    try:
        deadline=time.monotonic()+180
        while True:
            try:
                health=get(port,'/health');ready=get(port,'/ready')
                if ready[0]==200: break
                if not transcript: record('health_during_load',health);record('ready_during_load',ready)
                debug=get(port,'/debug')[1]
                if debug['error']: raise RuntimeError(debug['error'])
            except URLError: pass
            if proc.poll() is not None or time.monotonic()>deadline: raise RuntimeError('Load timeout; inspect server log')
            time.sleep(.2)
        record('ready_after_warmup',ready)
        before=get(port,'/debug')[1]
        invalid=get(port,'/generate',{'prompt':'test','max_new_tokens':100000})
        after=get(port,'/debug')[1]
        assert invalid[0]==422 and before['calls']==after['calls']
        record('invalid_without_compute',{'response':invalid,'calls_before':before['calls'],'calls_after':after['calls']})
        prompts=[json.loads(line)['input'] for line in (ROOT/'evidence/golden.jsonl').read_text().splitlines()][:8]
        limit=json.loads((ROOT/'accepted.json').read_text())['runtime']['generation']['max_new_tokens']
        record('real_smoke',get(port,'/generate',{'prompt':prompts[0],'max_new_tokens':limit}))
        all_rows=[];summaries=[]
        attempts=ROOT/'docs/requests.jsonl'
        attempts.write_text('')
        for concurrency in (1,2):
            def one(item):
                index,prompt=item
                start=time.perf_counter()
                row={'id':f'c{concurrency}-{index+1}','prompt_id':index+1,'concurrency':concurrency,
                     'max_new_tokens':limit,'error':None}
                try:
                    status,result=get(port,'/generate',{'prompt':prompt,'max_new_tokens':limit})
                    row.update(status=status,client_seconds=time.perf_counter()-start,
                               error=None if status==200 else str(result),**(result if status==200 else {}))
                except Exception as e:
                    row.update(status=None,error=f'{type(e).__name__}: {e}',client_seconds=time.perf_counter()-start)
                return row
            begin=time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                for row in pool.map(one,enumerate(prompts)):
                    all_rows.append(row)
                    with attempts.open('a') as f: f.write(json.dumps(row,ensure_ascii=False)+'\n')
            seconds=time.perf_counter()-begin
            batch=[r for r in all_rows if r['concurrency']==concurrency]
            good=[r for r in batch if r['status']==200]
            errors=[r for r in batch if r['status']!=200]
            summaries.append({'concurrency':concurrency,'n':len(batch),'successes':len(good),'errors':len(errors),
                'batch_seconds':seconds,'success_req_per_s':len(good)/seconds,
                'mean_success_seconds':sum(r['client_seconds'] for r in good)/len(good) if good else None,
                'max_success_seconds':max((r['client_seconds'] for r in good),default=None),
                'mean_error_seconds':sum(r['client_seconds'] for r in errors)/len(errors) if errors else None,
                'max_error_seconds':max((r['client_seconds'] for r in errors),default=None),
                'output_tokens':[r['output_tokens'] for r in good]})
            print(json.dumps(summaries[-1]),flush=True)
            (ROOT/'docs/benchmark.json').write_text(json.dumps(summaries,indent=2))
        # Restart same actual model; readiness metadata must remain tied to same manifest.
        record('final_debug',get(port,'/debug'))
    finally:
        proc.terminate()
        try: proc.wait(timeout=15)
        except subprocess.TimeoutExpired: proc.kill();proc.wait()
        log.close()


if __name__=='__main__':main()
