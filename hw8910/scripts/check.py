"""One fast no-weights suite; --mutation proves a real regression returns nonzero."""
import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--mutation',action='store_true');a=p.parse_args()
    if not a.mutation:
        original=subprocess.check_output(['git','show','38c2458:hw8910/tests/test_release.py'],cwd=ROOT)
        current=(ROOT/'tests/test_release.py').read_bytes()
        if original!=current: raise SystemExit('Instructor tests changed')
        print('Instructor fingerprint:',hashlib.sha256(b'test_release.py'+current).hexdigest()[:12],flush=True)
        raise SystemExit(subprocess.call([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT))
    with tempfile.TemporaryDirectory() as tmp:
        work=Path(tmp)
        shutil.copytree(ROOT/'tests',work/'tests')
        for name in ['release.py','lab.py','service.py','backends.py']:
            shutil.copy2(ROOT/name,work/name)
        source=(work/'release.py').read_text()
        source=source.replace("return digest(canonical({'data_sha256': file_digest(data_path), 'config': config}))",
                              "return digest(str(data_path).encode())  # deliberate defect in isolated copy")
        (work/'release.py').write_text(source)
        print('INTENTIONAL MUTATION: input bytes/config omitted from cache key',flush=True)
        raise SystemExit(subprocess.call([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=work))


if __name__=='__main__':main()
