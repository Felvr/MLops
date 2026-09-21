#!/usr/bin/env python3
"""Измерить эффект старого случайного сплита и нарушения вырожденного набора."""
import json
from pathlib import Path
import random
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_params
from src.contamination import report
from src.diversity import measure, violations
from src.schema import iter_examples


def main():
    params = load_params()
    rows = list(iter_examples(params["paths"]["clean"]))
    order = list(range(len(rows)))
    random.Random(params["split"]["seed"]).shuffle(order)
    train_stop = round(len(rows) * params["split"]["ratios"]["train"])
    val_stop = train_stop + round(len(rows) * params["split"]["ratios"]["val"])
    nd = params["clean"]["near_dup"]
    old = report([rows[i] for i in order[:train_stop]], [rows[i] for i in order[val_stop:]],
                 shingle_words=nd["shingle_words"], num_perm=nd["num_perm"], threshold=nd["threshold"])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "degenerate.jsonl"
        with path.open("w") as output:
            for i in range(1200):
                output.write(json.dumps({"id":f"deg_{i}","topic":"Одна-единственная тема - 2025",
                    "messages":[{"role":"system","content":"Ты отвечаешь на вопросы. Отвечай одним словом."},
                    {"role":"user","content":f"Вопрос номер {i} про один и тот же предмет?"},
                    {"role":"assistant","content":f"Ответ: {i % 4}. вариант"}]},ensure_ascii=False)+"\n")
        stats = measure(str(path), "topic")
    result = {"version":params["collect"]["version"], "rows":len(rows),
              "old_random_split_train_test":old,
              "fixed_split":json.loads(Path(params["paths"]["metrics_contamination"]).read_text()),
              "degenerate":{"stats":stats,"violations":violations(stats,params["diversity"])}}
    Path("docs/defect-evidence.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
