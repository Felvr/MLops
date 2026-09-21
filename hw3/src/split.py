"""Стадия split: разбиение на train/val/test."""

import json
import hashlib
import time
from pathlib import Path

from src.config import load_params
from src.contamination import is_clean, report
from src.schema import Example, dump, iter_examples
from src.textnorm import normalize_group


def group_split(groups: list[str], ratios: dict[str, float], seed: int) -> dict[str, str]:
    """Стабильное назначение группы: новые данные не меняют старые сплиты."""
    if set(ratios) != {"train", "val", "test"} or any(v <= 0 for v in ratios.values()):
        raise ValueError("Нужны положительные доли train, val, test")
    if abs(sum(ratios.values()) - 1) > 1e-9:
        raise ValueError("Сумма долей должна быть 1")
    labels = {}
    for group in sorted(groups):
        value = int(hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(), 16) / 2**256
        cumulative = 0.0
        for name, ratio in ratios.items():
            cumulative += ratio
            if value < cumulative:
                labels[group] = name
                break
    return labels


def main() -> None:
    params = load_params()
    paths = params["paths"]
    cfg = params["split"]
    started = time.perf_counter()

    examples: list[Example] = list(iter_examples(paths["clean"]))
    if cfg["group_key"] != "topic":
        raise SystemExit(f"неизвестный split.group_key: {cfg['group_key']!r}")

    sizes: dict[str, int] = {}
    for ex in examples:
        key = normalize_group(ex.topic)
        sizes[key] = sizes.get(key, 0) + 1

    labels = group_split(list(sizes), cfg["ratios"], cfg["seed"])
    buckets: dict[str, list[Example]] = {name: [] for name in cfg["ratios"]}
    for ex in examples:
        buckets[labels[normalize_group(ex.topic)]].append(ex)
    if any(not rows for rows in buckets.values()):
        raise SystemExit("split: один из сплитов пуст — недостаточно групп")

    for name, rows in buckets.items():
        out = Path(paths[name])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in rows:
                fh.write(dump(ex) + "\n")

    nd = params["clean"]["near_dup"]
    rep = report(
        buckets["train"],
        buckets["test"],
        shingle_words=nd["shingle_words"],
        num_perm=nd["num_perm"],
        threshold=params["contamination"]["threshold"],
    )

    if not is_clean(rep):
        raise SystemExit(f"split: КОНТАМИНАЦИЯ: {rep}")

    metrics = {
        "version": params["collect"]["version"],
        "seed": cfg["seed"],
        "group_key": cfg["group_key"],
        "groups_total": len(sizes),
        "sizes": {name: len(rows) for name, rows in buckets.items()},
        "groups": {
            name: len({normalize_group(ex.topic) for ex in rows}) for name, rows in buckets.items()
        },
        "ratios_actual": {
            name: round(len(rows) / len(examples), 4) for name, rows in buckets.items()
        },
        "contamination": rep,
        "seconds": round(time.perf_counter() - started, 2),
    }
    mpath = Path(paths["metrics_split"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        "split: "
        + ", ".join(f"{name} {len(rows)}" for name, rows in buckets.items())
        + f" (групп {len(sizes)}, {metrics['seconds']} с)"
    )


if __name__ == "__main__":
    main()
