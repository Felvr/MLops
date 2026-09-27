"""Стадия tokenize: JSONL -> токенизированный датасет на диске + метрики + отчёт.

Что здесь происходит с каждым примером: применяется шаблон чата,
текст режется по max_seq_len, собираются input_ids / attention_mask / labels.

Формат на диске: torch.save одного словаря со списком примеров
(input_ids / attention_mask / labels — списки int, тензоры делает коллатор).
Так проще, чем datasets.save_to_disk, и файл целиком годится в `outs`
DVC-стадии:
    deps: data/train.jsonl, data/val.jsonl, src/, params.yaml
    outs: data/tokenized/
"""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from src.collate import LABEL_PAD_ID
from src.config import load_params
from src.prompt import build_chat_text, prompt_token_len, split_messages

METRICS_PATH = Path("metrics/tokenize.json")
REPORT_PATH = Path("docs/tokenize_report.md")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"Нет {path}.\n"
            "Выполните make prepare: вход — ваши сплиты из ДЗ3, формат id/topic/messages."
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def mask_prompt(input_ids: list[int], n_prompt: int) -> list[int]:
    """labels для лосса."""
    boundary = min(n_prompt, len(input_ids))
    return [LABEL_PAD_ID] * boundary + list(input_ids[boundary:])


def encode_example(tokenizer, record: dict, params: dict, max_seq_len: int) -> dict:
    """Один пример -> input_ids / attention_mask / labels + служебная статистика."""
    if max_seq_len <= 0:
        raise ValueError("max_seq_len должен быть положительным")
    messages = record["messages"]
    split_messages(messages)
    full_text = build_chat_text(tokenizer, messages, params, add_generation_prompt=False)

    prompt_text = build_chat_text(tokenizer, messages, params, add_generation_prompt=True)
    if not full_text.startswith(prompt_text):
        raise ValueError(f"{record.get('id')}: inference не является префиксом train")
    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    input_ids = encoded["input_ids"]
    n_prompt, used_fallback = prompt_token_len(
        tokenizer, prompt_text, input_ids, encoded["offset_mapping"]
    )

    full_len = len(input_ids)
    truncated = full_len > max_seq_len
    if truncated:
        input_ids = input_ids[:max_seq_len]

    labels = mask_prompt(input_ids, n_prompt)
    return {
        "id": record.get("id"),
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "_meta": {
            "id": record.get("id"),
            "full_len": full_len,
            "prompt_len": n_prompt,
            "answer_len": full_len - n_prompt,
            "truncated": truncated,
            "bpe_fallback": used_fallback,
            "supervised": sum(1 for x in labels if x != LABEL_PAD_ID),
        },
    }


def describe(values: list[int]) -> dict:
    """Распределение длин: перцентили важнее среднего — хвост решает max_seq_len."""
    if not values:
        raise ValueError("Сплит не должен быть пустым")
    a = np.asarray(values)
    return {
        "count": int(a.size),
        "mean": round(float(a.mean()), 1),
        "p50": int(np.percentile(a, 50)),
        "p90": int(np.percentile(a, 90)),
        "p99": int(np.percentile(a, 99)),
        "max": int(a.max()),
    }


def truncation_stats(metas: list[dict], name: str, params: dict) -> dict:
    """Статистика обрезки по max_seq_len."""
    count = sum(m["truncated"] for m in metas)
    ratio = count / len(metas) if metas else 0.0
    if ratio > params["tokenize"]["truncated_warn_ratio"]:
        warnings.warn(f"{name}: обрезано {count}/{len(metas)} ({ratio:.1%}), выше порога",
                      stacklevel=2)
    return {"truncated": count, "truncated_ratio": ratio}


def process_split(
    tokenizer, name: str, path: Path, params: dict
) -> tuple[list[dict], dict, list[dict]]:
    """Токенизировать сплит и собрать по нему статистику."""
    cfg = params["tokenize"]
    max_seq_len = cfg["max_seq_len"]
    records = read_jsonl(path)

    examples: list[dict] = []
    metas: list[dict] = []
    dropped = 0
    for record in records:
        encoded = encode_example(tokenizer, record, params, max_seq_len)
        # Статистика длин считается по ВСЕМ записям, включая выброшенные:
        # иначе доля обрезанных занижается ровно на самые длинные примеры.
        meta = encoded.pop("_meta")
        metas.append(meta)
        # Обрезка съела весь ответ: учить нечему, такой пример только шумит.
        if meta["supervised"] == 0:
            dropped += 1
            continue
        examples.append(encoded)

    stats = {
        "examples_in": len(records),
        "examples_kept": len(examples),
        "dropped_no_supervision": dropped,
        "length_tokens": describe([m["full_len"] for m in metas]),
        "prompt_tokens": describe([m["prompt_len"] for m in metas]),
        "answer_tokens": describe([m["answer_len"] for m in metas]),
        "bpe_boundary_fallback": sum(m["bpe_fallback"] for m in metas),
        "total_tokens": sum(len(e["input_ids"]) for e in examples),
        "supervised_tokens": sum(m["supervised"] for m in metas),
    }
    print(
        f"  {name}: {len(examples)} примеров, токенов {stats['total_tokens']} "
        f"(в лосс идёт {stats['supervised_tokens']}), p50/p90/max = "
        f"{stats['length_tokens']['p50']}/{stats['length_tokens']['p90']}/"
        f"{stats['length_tokens']['max']}"
    )
    stats.update(truncation_stats(metas, name, params))

    if not examples:
        raise ValueError(f"{name}: после обрезки не осталось обучающих токенов")
    stats["packing"] = None
    bins = []

    return examples, stats, bins


def estimate_train_time(total_tokens: int, params: dict) -> dict:
    """Грубый прогноз времени обучения: токены x эпохи / пропускная способность."""
    cfg = params["train_estimate"]
    tps = cfg["tokens_per_sec"]
    seconds = total_tokens * cfg["epochs"] / tps
    return {
        "epochs": cfg["epochs"],
        "tokens_per_sec": tps,
        "tokens_per_epoch": total_tokens,
        "seconds": round(seconds, 1),
        "hours": round(seconds / 3600, 2),
        "training_hours_range": [round(seconds * k / 3600, 2) for k in (2, 3)],
        "source": cfg["source"],
        "note": (
            "tokens_per_sec взят из замера ГЕНЕРАЦИИ в ДЗ 1. Обучение считает "
            "ещё backward и шаг оптимизатора, поэтому реальная скорость ниже "
            "по ориентиру лекции в 2-3 раза; это предположение, не измерение обучения."
        ),
    }


def mask_line(share: float) -> str:
    """Строка отчёта про долю токенов, попавших в лосс.

    Доля около единицы означает, что промпт не замаскирован: такой отчёт
    обязан сказать об этом прямо, а не молча показать красивое число.
    """
    if share > 0.99:
        return (
            "В лосс идут ПОЧТИ ВСЕ токены последовательности — похоже, промпт "
            "не замаскирован, и модель учится воспроизводить вопрос наравне с ответом."
        )
    return (
        f"Промпт занимает {1 - share:.1%} сохранённых токенов, ответ — {share:.1%}. "
        "Маска исключает промпт из целевой функции; его токены остаются во входе."
    )


def truncation_line(metrics: dict, train: dict) -> str:
    """Строка отчёта про обрезку — или честное признание, что её не считали."""
    warn = metrics["truncated_warn_ratio"]
    if "truncated_ratio" not in train:
        return (
            "Доля обрезанных НЕ ПОСЧИТАНА: стадия не знает, сколько ответов "
            f"потеряла на max_seq_len = {metrics['max_seq_len']}."
        )
    ratio = train["truncated_ratio"]
    verdict = "в норме" if ratio <= warn else "ВЫШЕ ПОРОГА"
    return f"Порог предупреждения: {warn:.1%}. Фактически обрезано (train): {ratio:.1%} — {verdict}."


def truncated_cell(s: dict) -> str:
    """Ячейка «обрезано». Если статистики нет — так и пишем, а не молчим."""
    if "truncated_ratio" not in s:
        return "НЕ СЧИТАЛАСЬ"
    return f"{s['truncated']} ({s['truncated_ratio']:.1%})"


def render_report(metrics: dict) -> str:
    """Отчёт по датасету — то, что читают глазами перед запуском обучения."""
    lines = [
        "# Отчёт стадии tokenize",
        "",
        "Сгенерирован `make tokenize`, руками не правится.",
        "",
        f"- модель: `{metrics['model']}`",
        f"- revision: `{metrics['revision']}`",
        f"- `max_seq_len`: {metrics['max_seq_len']}",
        f"- `enable_thinking`: {str(metrics['enable_thinking']).lower()}",
        f"- `padding_side`: {metrics['padding_side']}",
        "",
        "## Длины в токенах",
        "",
        "| сплит | примеров | p50 | p90 | p99 | max | обрезано | всего токенов | в лосс |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, s in metrics["splits"].items():
        L = s["length_tokens"]
        lines.append(
            f"| {name} | {s['examples_kept']} | {L['p50']} | {L['p90']} | {L['p99']} | "
            f"{L['max']} | {truncated_cell(s)} | "
            f"{s['total_tokens']} | {s['supervised_tokens']} |"
        )

    train = metrics["splits"]["train"]
    est = metrics["train_time_estimate"]
    share = train["supervised_tokens"] / train["total_tokens"]
    lines += [
        "",
        mask_line(share),
        "",
        "## Обрезка",
        "",
        truncation_line(metrics, train),
        "",
        f"p99 длины в токенах — {train['length_tokens']['p99']} при `max_seq_len` "
        f"{metrics['max_seq_len']}; хвост длиной до {train['length_tokens']['max']} "
        "теряет конец ответа. Примеров, у которых обрезка съела ответ целиком: "
        f"{train['dropped_no_supervision']} (такие выброшены).",
        f"Val: выброшено {metrics['splits']['val']['dropped_no_supervision']}.",
        "",
        "## Граница маски и BPE",
        "",
        f"Запасной путь по символьным офсетам сработал на {train['bpe_boundary_fallback']} "
        "примерах train. Основной путь (токены промпта совпадают с началом токенов "
        "полного текста) держится, потому что шаблон Qwen3 заканчивает промпт "
        "переводом строки — BPE не склеивает его с началом ответа. У шаблона, "
        "который обрывается посреди слова, склейка будет, и тогда границу задаёт "
        "первый токен, начинающийся не раньше конца промпта.",
        "",
        "## Прогноз времени обучения",
        "",
        f"Токенов за эпоху: {est['tokens_per_epoch']}, эпох: {est['epochs']}, "
        f"скорость: {est['tokens_per_sec']} ток/с.",
        "",
        f"Оптимистичная оценка: {est['seconds']:.0f} с ({est['hours']} ч).",
        f"При замедлении в 2–3 раза: {est['training_hours_range'][0]}–"
        f"{est['training_hours_range'][1]} ч. Источник скорости: `{est['source']}`.",
        "Это экстраполяция генерации на CPU, не замер обучения. Паддинг и валидация "
        "не учтены; скорость зависит от длины, батча и оборудования.",
        "",
        est["note"],
        "",
    ]

    lines += ["## Packing", "", "Отключён: для независимости примеров требуется "
              "блочная маска внимания в обучении. Используется динамический левый паддинг.", ""]
    return "\n".join(lines)


def main() -> None:
    params = load_params()
    if params["packing"]["enabled"]:
        raise ValueError("Packing требует отдельного обучения с блочной маской внимания")
    tokenizer = AutoTokenizer.from_pretrained(
        params["model"]["name"], revision=params["model"]["revision"]
    )
    tokenizer.padding_side = params["tokenize"]["padding_side"]
    if tokenizer.padding_side != "left" or tokenizer.pad_token_id is None:
        raise ValueError("Нужен левый паддинг и pad_token_id")

    out_dir = Path(params["data"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name, key in (("train", "train_jsonl"), ("val", "val_jsonl")):
        examples, stats, bins = process_split(tokenizer, name, Path(params["data"][key]), params)
        torch.save(
            {
                "examples": examples,
                "model": params["model"]["name"],
                "revision": params["model"]["revision"],
                "max_seq_len": params["tokenize"]["max_seq_len"],
                "padding_side": tokenizer.padding_side,
                "pad_token_id": tokenizer.pad_token_id,
            },
            out_dir / f"{name}.pt",
        )
        splits[name] = stats

    metrics = {
        "model": params["model"]["name"],
        "revision": params["model"]["revision"],
        "input_sha256": {name: hashlib.sha256(Path(params["data"][key]).read_bytes()).hexdigest()
                         for name, key in (("train", "train_jsonl"), ("val", "val_jsonl"))},
        "enable_thinking": params["model"].get("enable_thinking"),
        "max_seq_len": params["tokenize"]["max_seq_len"],
        "padding_side": tokenizer.padding_side,
        "truncated_warn_ratio": params["tokenize"]["truncated_warn_ratio"],
        "splits": splits,
        "train_time_estimate": estimate_train_time(splits["train"]["total_tokens"], params),
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(metrics), encoding="utf-8")
    print(f"  -> {out_dir}/, {METRICS_PATH}, {REPORT_PATH}")


if __name__ == "__main__":
    main()
