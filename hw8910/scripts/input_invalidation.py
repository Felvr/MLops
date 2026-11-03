"""Mutate real evaluation input at same path, prove re-preparation and fail closed.
Restores exact bytes even on failure. No training or generation is performed.
"""
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from release import file_digest
from prefect_local import local_server
ROOT=Path(__file__).resolve().parents[1]


def main():
    source=ROOT/'evidence/golden.jsonl'
    original=source.read_bytes()
    before=file_digest(source)
    pointer_before=file_digest(ROOT/'accepted.json')
    try:
        # Same logical records, changed bytes at the SAME path.
        source.write_bytes(b'\n'.join(b' '+line for line in original.splitlines())+b'\n')
        changed=file_digest(source)
        p=subprocess.run([sys.executable,'pipeline.py'],cwd=ROOT,capture_output=True,text=True,timeout=180)
        transcript=p.stdout+p.stderr
        (ROOT/'docs/flow-real-invalidation.log').write_text(transcript+f'\nEXIT_CODE={p.returncode}\n')
        assert p.returncode!=0
        assert 'PREPARATION_STATE Completed' in transcript and 'PREPARED' in transcript
        assert 'Evaluation source changed' in transcript
        assert file_digest(ROOT/'accepted.json')==pointer_before
    finally:
        source.write_bytes(original)
    assert file_digest(source)==before
    (ROOT/'docs/real-invalidation.json').write_text(json.dumps({'input_before':before,'input_changed':changed,
        'input_after_restore':file_digest(source),'pointer_before':pointer_before,
        'pointer_after':file_digest(ROOT/'accepted.json'),'restored':True,'exit_code':p.returncode},indent=2))


if __name__=='__main__':
    with local_server():
        main()
        print('Real input invalidated cache; gate blocked stale evidence; exact bytes restored')
