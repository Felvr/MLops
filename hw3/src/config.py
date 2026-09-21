"""Чтение params.yaml — единственная точка правды о конфигурации."""

from pathlib import Path

import yaml


def load_params(path: str = "params.yaml") -> dict:
    """Загрузить параметры запуска."""
    params = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if params["clean"]["near_dup"]["threshold"] != params["contamination"]["threshold"]:
        raise ValueError("Пороги clean.near_dup и contamination должны совпадать")
    return params
