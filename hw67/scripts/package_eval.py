"""Provide evaluation data without a private DVC remote."""
import sys
import zipfile
from pathlib import Path

path = Path('docs/evaluation-data.zip')
if '--restore' in sys.argv:
    with zipfile.ZipFile(path) as archive:
        for item in archive.infolist():
            target = (Path('.') / item.filename).resolve()
            if not target.is_relative_to(Path('data/eval').resolve()):
                raise ValueError('Unsafe archive path')
        archive.extractall('.')
    print('Restored evaluation data; verify with uv run dvc status data/eval.dvc')
else:
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(Path('data/eval').glob('*.jsonl')):
            archive.write(source, source.as_posix())
    print(path)
