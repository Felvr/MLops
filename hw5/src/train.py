"""Стадия train: LoRA-дообучение на выходе стадии tokenize.

    python -m src.train --variant all_layers          # полный прогон варианта
    python -m src.train --variant all_layers --max-steps 4 --out /tmp/x   # smoke

Цикл обучения написан руками, а не через Trainer: так видно всё, что
обычно прячется, — где считается val loss, как копятся градиенты, что
сохраняется рядом с адаптером.

Что сохраняется в models/adapter_<variant>/: адаптер.
В metrics/train_<variant>.json — кривые train/val loss, время, пиковая память,
число обучаемых параметров, вес адаптера и отпечаток входов.
"""

import argparse
import hashlib
import json
import math
from itertools import islice
import os
import time
from pathlib import Path

# Потолок памяти Metal — до импорта torch. Без него mps занимает сколько дадут,
# и на ноутбуке с 16–32 ГБ система уходит в своп вместо внятной ошибки.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.4")   # нижний порог не выше верхнего

import torch  # noqa: E402
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.config import load_params
from src.data import LABEL_PAD_ID, batches, load_split, validate_subset
from src.runtime import allocated_bytes, memory_metric, resolve_device, resolve_dtype, set_seed


TRAIN_CODE = ("src/train.py", "src/data.py", "src/runtime.py", "src/config.py")
TRAIN_PARAMS = ("model", "data", "lora", "train", "variants")


def inputs_fingerprint(params: dict) -> str:
    """Отпечаток кода обучения и секций конфига, от которых зависит адаптер.

    check.sh сверяет его с текущим: правили train.py или lr, а адаптер
    остался от прошлого прогона — проверять его бессмысленно.
    """
    h = hashlib.sha256()
    for name in TRAIN_CODE:
        h.update(name.encode())
        h.update(Path(name).read_bytes())
    h.update(json.dumps({k: params.get(k) for k in TRAIN_PARAMS}, sort_keys=True).encode())
    for split in ("train", "val"):
        h.update(Path(params["data"][split]).read_bytes())
    h.update(Path("uv.lock").read_bytes())
    return h.hexdigest()[:12]


def lora_config(params: dict, n_layers: int, freeze_first: int) -> LoraConfig:
    if not 0 <= freeze_first < n_layers:
        raise ValueError("freeze_first must be in [0, n_layers)")
    cfg = params["lora"]
    return LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=cfg["dropout"],
        target_modules=cfg["target_modules"],
        modules_to_save=cfg.get("modules_to_save"),
        task_type="CAUSAL_LM",
        layers_to_transform=list(range(freeze_first, n_layers)),
        revision=params["model"].get("revision"),
    )


@torch.no_grad()
def evaluate(model, examples, pad_id, device, batch_size: int) -> float:
    """Средний лосс на токен по всему val-сплиту.

    Среднее по батчам нельзя: в батчах разное число токенов под маской.
    Поэтому сумма лоссов, взвешенная числом токенов, делённая на их сумму.
    """
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in batches(examples, batch_size, pad_id, shuffle=False, seed=0):
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"][:, 1:] != LABEL_PAD_ID).sum())
        if n == 0:
            continue
        loss = model(**batch).loss
        total += loss.item() * n
        count += n
    model.train(was_training)
    if device.type == "mps":
        torch.mps.empty_cache()   # логиты оценки не должны висеть в кэше до конца обучения
    if count == 0:
        raise ValueError("No supervised validation tokens")
    return total / count


def dir_size_mb(path: Path) -> float:
    return round(sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1048576, 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all_layers")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None, help="куда писать адаптер и метрики (smoke-тесты)")
    ap.add_argument("--val-limit", type=int, default=None, help="оценивать на первых N примерах val (smoke)")
    args = ap.parse_args()

    params = load_params()
    fingerprint = inputs_fingerprint(params)
    variants = {v["name"]: v for v in params["variants"]}
    if args.variant not in variants:
        raise SystemExit(f"нет варианта {args.variant!r}, есть: {sorted(variants)}")
    variant = variants[args.variant]
    tcfg = params["train"]
    max_steps = args.max_steps if args.max_steps is not None else tcfg.get("max_steps")

    device = resolve_device(params["model"]["device"])
    dtype = resolve_dtype(params["model"]["dtype"])

    set_seed(tcfg["seed"])
    torch.set_num_threads(tcfg.get("num_threads", 4))

    train_blob = load_split(params["data"]["train"])
    val_blob = load_split(params["data"]["val"])
    subset_config = params["data"].get("subset", {})
    validate_subset(train_blob, subset_config, "train")
    validate_subset(val_blob, subset_config, "val")
    if max_steps is not None and max_steps < 1:
        raise ValueError("max_steps must be positive or null")
    if args.val_limit is not None and args.val_limit < 1:
        raise ValueError("val-limit must be positive")
    if args.val_limit:
        val_blob["examples"] = val_blob["examples"][:args.val_limit]
    pad_id = train_blob["pad_token_id"]

    for blob in (train_blob, val_blob):
        if blob.get("model") != params["model"]["name"] or blob.get("revision") != params["model"].get("revision"):
            raise ValueError("Tokenized data and base model/revision differ")
    tokenizer = AutoTokenizer.from_pretrained(params["model"]["name"], revision=params["model"].get("revision"))
    model = AutoModelForCausalLM.from_pretrained(params["model"]["name"], dtype=dtype, revision=params["model"].get("revision")).to(device)
    n_layers = model.config.num_hidden_layers
    model.config.use_cache = False
    if params["train"].get("gradient_checkpointing"):
        # Активации 28 слоёв не храним, а пересчитываем на обратном проходе:
        # памяти в разы меньше, шаг примерно на треть дольше.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model = get_peft_model(model, lora_config(params, n_layers, variant["freeze_first"]))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    model.train()

    examples = train_blob["examples"]
    micro_per_epoch = math.ceil(len(examples) / tcfg["batch_size"])
    steps_per_epoch = math.ceil(micro_per_epoch / tcfg["grad_accum"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    if max_steps:
        total_steps = min(total_steps, max_steps)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, max(1, int(total_steps * tcfg["warmup_ratio"])), total_steps
    )

    print(f"[{args.variant}] evaluating baseline on {len(val_blob['examples'])} examples", flush=True)
    eval_bs = tcfg.get("eval_batch_size", tcfg["batch_size"])
    started_eval = time.perf_counter()
    base_val = evaluate(model, val_blob["examples"], pad_id, device, eval_bs)
    eval_seconds = time.perf_counter() - started_eval
    curve_train: list[list[float]] = []
    curve_val = [[0, round(base_val, 6)]]
    print(f"[{args.variant}] device={device}; trainable={trainable:,}/{total:,}; "
          f"steps={total_steps}; base val={base_val:.6f}", flush=True)
    peak = allocated_bytes(device)
    started = time.perf_counter()
    training_eval_seconds = 0.0
    step, accum_loss, accum_tokens, processed_tokens = 0, 0.0, 0, 0
    diverged = False
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(tcfg["epochs"]):
        iterator = iter(batches(examples, tcfg["batch_size"], pad_id,
                                shuffle=True, seed=tcfg["seed"] + epoch))
        while group := list(islice(iterator, tcfg["grad_accum"])):
            group_tokens = sum(int((batch["labels"][:, 1:] != LABEL_PAD_ID).sum()) for batch in group)
            if not group_tokens:
                raise ValueError("Training group has no supervised tokens")
            for batch in group:
                batch = {k: v.to(device) for k, v in batch.items()}
                n = int((batch["labels"][:, 1:] != LABEL_PAD_ID).sum())
                if not n:
                    continue
                loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                # Normalize by actual supervised tokens, including the final short group.
                (loss * (n / group_tokens)).backward()
                accum_loss += loss.item() * n
                accum_tokens += n
                processed_tokens += int(batch["attention_mask"].sum())
                peak = max(peak, allocated_bytes(device))
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            curve_train.append([step, round(accum_loss / accum_tokens, 6)])
            accum_loss, accum_tokens = 0.0, 0
            print(f"  step {step}/{total_steps}: train={curve_train[-1][1]:.6f}", flush=True)
            if step % tcfg["eval_every"] == 0 or step == total_steps:
                t_eval = time.perf_counter()
                val_loss = evaluate(model, val_blob["examples"], pad_id, device, eval_bs)
                elapsed_eval = time.perf_counter() - t_eval
                training_eval_seconds += elapsed_eval
                eval_seconds += elapsed_eval
                curve_val.append([step, round(val_loss, 6)])
                peak = max(peak, allocated_bytes(device))
                print(f"  val={val_loss:.6f}", flush=True)
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    seconds = time.perf_counter() - started - training_eval_seconds

    out_root = Path(args.out) if args.out else Path(params["paths"]["models"])
    adapter_dir = out_root / f"adapter_{args.variant}"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    metrics = {
        "variant": args.variant,
        "freeze_first": variant["freeze_first"],
        "model": params["model"]["name"],
        "device": device.type,
        "dtype": params["model"]["dtype"],
        "seed": tcfg["seed"],
        "lr": tcfg["lr"],
        "effective_batch": tcfg["batch_size"] * tcfg["grad_accum"],
        "steps": step,
        "max_steps": max_steps,
        "train_examples": len(examples),
        "data_subset": train_blob.get("subset"),
        "validation_subset": val_blob.get("subset"),
        "val_examples": len(val_blob["examples"]),
        "processed_tokens": processed_tokens,
        "revision": params["model"].get("revision"),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_share": round(trainable / total, 6),
        "base_val_loss": base_val,
        "final_val_loss": curve_val[-1][1] if curve_val else None,
        "diverged": diverged,
        "curve_train": curve_train,
        "curve_val": curve_val,
        "seconds": round(seconds, 1),
        "eval_seconds": round(eval_seconds, 1),
        "seconds_per_step": round(seconds / max(step, 1), 3),
        "train_tokens_per_sec": round(processed_tokens / seconds, 1) if seconds else 0,
        "peak_memory_mb": round(peak / 1048576, 1),
        "memory_metric": memory_metric(device),
        "adapter_dir": str(adapter_dir),
        "adapter_size_mb": dir_size_mb(adapter_dir),
        "inputs_fingerprint": fingerprint,
    }
    mdir = out_root / "metrics" if args.out else Path(params["paths"]["metrics"])
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / f"train_{args.variant}.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[{args.variant}] {step} шагов за {seconds:.0f} с; "
          f"пик памяти {metrics['peak_memory_mb']:.0f} МБ; адаптер {metrics['adapter_size_mb']} МБ -> {adapter_dir}")


if __name__ == "__main__":
    main()
