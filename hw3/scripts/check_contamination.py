#!/usr/bin/env python3
"""Отдельный гейт для ВСЕХ пар train/val/test с машинным отчётом."""
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_params
from src.contamination import is_clean, report
from src.schema import iter_examples


def main():
    params = load_params()
    paths, nd = params["paths"], params["clean"]["near_dup"]
    buckets = {name: list(iter_examples(paths[name])) for name in ("train", "val", "test")}
    results = {}
    for left, right in itertools.combinations(buckets, 2):
        rep = report(buckets[left], buckets[right], shingle_words=nd["shingle_words"],
                     num_perm=nd["num_perm"], threshold=params["contamination"]["threshold"])
        results[f"{left}_{right}"] = rep
        print(f"{left}↔{right}: пересечение по id {rep['id_overlap']}, "
              f"по тексту {rep['text_overlap']}, по группам {rep['group_overlap']}, "
              f"near-dup {rep['near_dup_pairs']}")
    passed = all(is_clean(rep) for rep in results.values()) and all(buckets.values())
    output = Path(paths["metrics_contamination"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"passed": passed, "pairs": results}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("контаминации нет" if passed else "КОНТАМИНАЦИЯ или пустой сплит")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
