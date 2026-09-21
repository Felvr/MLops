"""Скачать закреплённый снимок MedQuAD и проверить SHA-256."""
import hashlib
import os
from pathlib import Path
import ssl
import urllib.request

import certifi

from src.config import load_params


def main():
    cfg = load_params()["source"]
    target = Path(cfg["path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == cfg["sha256"]:
        print("download: SHA-256 исходника подтверждён")
        return
    temporary = target.with_suffix(".download")
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(cfg["url"], context=context, timeout=120) as response:
            with temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    output.write(block)
        actual = hashlib.sha256(temporary.read_bytes()).hexdigest()
        if actual != cfg["sha256"]:
            raise ValueError(f"SHA-256 не совпадает: {actual}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"download: {target}, SHA-256 {cfg['sha256']}")


if __name__ == "__main__":
    main()
