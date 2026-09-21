#!/usr/bin/env python3
"""Переключить версию, сохраняя формат и кавычки остальных параметров."""
import re
import sys
from pathlib import Path
import yaml

PARAMS = Path(__file__).resolve().parents[1] / "params.yaml"


def main():
    if len(sys.argv) != 2:
        print("использование: set_version.py v1|v2", file=sys.stderr)
        return 2
    version = sys.argv[1]
    text = PARAMS.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    if version not in cfg["collect"]["documents"]:
        print(f"неизвестная версия {version!r}", file=sys.stderr)
        return 1
    new, count = re.subn(r'^  version:.*$', f'  version: "{version}"', text, count=1, flags=re.M)
    if count != 1:
        raise ValueError("не найдена строка collect.version")
    PARAMS.write_text(new, encoding="utf-8")
    print(f"collect.version = {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
