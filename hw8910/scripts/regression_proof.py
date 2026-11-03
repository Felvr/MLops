"""Run the SAME seven integration tests against the preserved original commit."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
ORIGINAL='38c2458'


def main():
    with tempfile.TemporaryDirectory() as tmp:
        original=Path(tmp)
        for name in ('release.py','lab.py','service.py'):
            source=subprocess.check_output(['git','show',f'{ORIGINAL}:hw8910/{name}'],cwd=ROOT)
            (original/name).write_bytes(source)
        p=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-p','test_integration.py','-v'],
                         cwd=ROOT,env={**os.environ,'PROTOTYPE_ROOT':str(original)},capture_output=True,text=True)
        (ROOT/'docs/red-original.log').write_text(p.stdout+p.stderr+f'\nEXIT_CODE={p.returncode}\n')
        assert p.returncode!=0,'Original defects unexpectedly passed'
        assert 'Ran 7 tests' in p.stderr and 'ERROR' not in p.stderr,'Test must fail on behavior, not import/launch'
        names=['test_cache_recovery_and_changed_bytes','test_reports_fail_closed_and_preserve_bytes',
               'test_repeat_across_processes_one_event','test_http_readiness_blocks_backend',
               'test_http_loaded_version_restart_and_rollback','test_http_invalid_requests_never_call_backend',
               'test_atomic_disk_failure_keeps_old_pointer']
        for name in names: assert f'FAIL: {name}' in p.stderr,name
        diff=subprocess.check_output(['git','diff','--no-ext-diff','--unified=0',ORIGINAL,'--','hw8910/release.py','hw8910/lab.py','hw8910/service.py'],cwd=ROOT.parent)
        (ROOT/'docs/fixes.diff').write_bytes(diff)
        print('All seven tests failed on original behavior; saved exit code',p.returncode)


if __name__=='__main__':main()
