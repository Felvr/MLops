"""Побайтово перенести сплиты ДЗ3, не пересоздавая разбиение."""
import hashlib
import json
from pathlib import Path
import shutil

from src.config import load_params


def main():
    params = load_params()
    sources = [Path(params['data'][f'source_{name}']) for name in ('train', 'val')]
    for source in sources:
        if not source.is_file():
            raise SystemExit(f'Нет {source}. Сначала выполните: cd ../hw3 && uv sync --locked && make repro')
    provenance = {}
    for name, source in zip(('train', 'val'), sources):
        target = Path(params['data'][f'{name}_jsonl'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        raw = target.read_bytes()
        provenance[name] = {'source': str(source), 'sha256': hashlib.sha256(raw).hexdigest(),
                            'rows': len(raw.splitlines())}
    output = Path('metrics/input.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + '\n')
    print('Сплиты ДЗ3 скопированы без изменений:', ', '.join(str(p) for p in sources))


if __name__ == '__main__':
    main()
