"""Capture real Prefect runs in independent processes, including failure states."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from prefect_local import local_server

ROOT=Path(__file__).resolve().parents[1]


def run(args,log,expected):
    p=subprocess.run([sys.executable,*args],cwd=ROOT,capture_output=True,text=True,timeout=180)
    (ROOT/'docs'/log).write_text(p.stdout+p.stderr+f'\nEXIT_CODE={p.returncode}\n')
    assert (p.returncode==0)==expected,(log,p.returncode)
    return p


def main():
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)/'flow'
        run(['lab.py','--workspace',str(work),'init'],'flow-stub-init.log',True)
        cmd=['pipeline.py','--workspace',str(work),'--synthetic']
        run(cmd+['--fail-after-prepare'],'flow-stub-failure.log',False)
        run(cmd,'flow-stub-recovery.log',True)
        run(cmd,'flow-stub-repeat.log',True)
        (work/'input.txt').write_text('Changed bytes at the SAME path\n')
        run(cmd,'flow-stub-invalidation.log',True)
        assert len((work/'publications.jsonl').read_text().splitlines())==1
        # New report identity -> new release. Inject crash in the pointer/event gap.
        report=json.loads((work/'report-B.json').read_text());report['evaluation_run_id']='fixture-eval-B-second'
        (work/'report-B.json').write_text(json.dumps(report))
        run(cmd+['--fail-after-pointer'],'flow-stub-pointer-crash.log',False)
        assert (work/'pending.json').exists()
        run(cmd,'flow-stub-pointer-recovery.log',True)
        run(cmd,'flow-stub-pointer-repeat.log',True)
        assert len((work/'publications.jsonl').read_text().splitlines())==2
        before=(work/'accepted.json').read_bytes()
        run(cmd+['--report','report-reject.json'],'flow-stub-rejection.log',False)
        after=(work/'accepted.json').read_bytes()
        assert before==after
        (ROOT/'docs/flow-evidence.json').write_text(json.dumps({
            'accepted_before_reject_sha256':hashlib.sha256(before).hexdigest(),
            'accepted_after_reject_sha256':hashlib.sha256(after).hexdigest(),
            'publications':[json.loads(line) for line in (work/'publications.jsonl').read_text().splitlines()],
            'attempts':[json.loads(line) for line in (work/'attempts.jsonl').read_text().splitlines()]},indent=2))
    before=(ROOT/'accepted.json').read_bytes()
    run(['pipeline.py','--fail-after-prepare'],'flow-real-failure.log',False)
    run(['pipeline.py'],'flow-real-recovery-rejection.log',False)
    after=(ROOT/'accepted.json').read_bytes()
    assert before==after
    (ROOT/'docs/real-rejection.json').write_text(json.dumps({'before_sha256':hashlib.sha256(before).hexdigest(),
        'after_sha256':hashlib.sha256(after).hexdigest(),'unchanged':before==after,'decision':json.loads((ROOT/'release-report.json').read_text())},indent=2))


if __name__=='__main__':
    with local_server():
        main()
        print('All synthetic and real Prefect experiments verified; exit 0')
