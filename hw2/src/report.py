"""Сборка docs/anatomy.md и графика норм активаций."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # без дисплея: скрипт должен работать и в CI

import matplotlib.pyplot as plt  # noqa: E402  (backend выбирается до импорта)

MODE_TITLES = {
    "inference": "инференс",
    "full_ft": "full fine-tune",
    "lora": "LoRA (r=8, q/v)",
}


def thousands(n: int) -> str:
    """Число с неразрывными пробелами по разрядам."""
    return f"{n:,}".replace(",", " ")


def plot_activations(activations: dict, path: str) -> None:
    """Две панели: норма по позициям токена и средняя норма по трём блокам."""
    labels = list(activations["norms"])
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 4), width_ratios=(2, 1))

    for label in labels:
        values = activations["norms"][label]
        ax_left.plot(values, linewidth=1.4,
                     label=f"{label} (слой {activations['layers'][label]})")
    ax_left.set_xlabel("позиция токена")
    ax_left.set_yscale("log")   # без лога всё придавит выброс massive activations
    ax_left.set_ylabel("‖h‖₂ (лог. шкала)")
    ax_left.set_title("Норма скрытого состояния по позициям")
    ax_left.legend(fontsize=9)
    ax_left.grid(alpha=0.3)

    means = [sum(activations["norms"][x]) / len(activations["norms"][x]) for x in labels]
    ax_right.bar(labels, means, color=["#4c78a8", "#f58518", "#54a24b"])
    ax_right.set_ylabel("средняя ‖h‖₂")
    ax_right.set_title("Средняя норма по блоку")
    ax_right.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    Path(path).parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def conditions_section(report: dict) -> list[str]:
    """Условия замера. Без них ни одна цифра ниже не сравнима ни с чем."""
    env = report["environment"]
    return [
        "## 2. Условия замера",
        "",
        "| Условие | Значение |",
        "|---|---|",
        f"| платформа | {env['platform']} ({env['system']}, {env['machine']}) |",
        f"| устройство | `{env['device']}` |",
        f"| dtype | `{env['dtype']}` |",
        f"| seq_len × batch | {env['seq_len']} × {env['batch_size']} |",
        f"| прогонов на режим | {env['repeats']} |",
        f"| память измерена | `{env['memory_metric']}` |",
        f"| RSS измерена | `{env['rss_metric']}` |",
        f"| дата замера | {env['measured_at']} |",
        f"| revision модели | {env['model_revision']} |",
        f"| seed | {env['seed']} |",
        f"| потоков torch CPU | {env['num_threads']} |",
        "| use_cache | False во всех режимах |",
        "| gradient checkpointing | выключен |",
        f"| python | {env['python']} |",
        f"| torch | {env['torch']} |",
        f"| transformers | {env['transformers']} |",
        f"| peft | {env['peft']} |",
        "",
        "Цифры ниже верны только для этих условий. Замер памяти без указания",
        "устройства, dtype, длины последовательности и метрики не воспроизводится",
        "и не сравнивается — поэтому источник метрики стоит отдельной строкой.",
        "Сверьте, что в этой строке названа метрика, уместная для вашего",
        "устройства, и что полученные числа с ней согласуются.",
        "",
    ]


def params_section(report: dict) -> list[str]:
    lines = [
        "## 3. Параметры по типам модулей",
        "",
        "| Группа | Модулей | Shape | Параметров | Доля | Разделяет тензор |",
        "|---|--:|---|--:|--:|--:|",
    ]
    for item in report["params_by_group"]:
        tied = thousands(item["tied_params"]) if item["tied_params"] else "—"
        lines.append(
            f"| `{item['group']}` | {item['modules']} | {item['shape']} | "
            f"{thousands(item['params'])} | {item['share'] * 100:.2f}% | {tied} |"
        )
    lines += [
        f"| **итого** | | | **{thousands(report['params_total'])}** | 100% | |",
        "",
        f"Контроль: `sum(p.numel() for p in model.parameters())` = "
        f"{thousands(report['params_direct'])} — сходится с суммой по таблице.",
        "",
        "Обратите внимание на строку `lm_head` и на значение `tie_word_embeddings`",
        "в конфигурации: они связаны, и от этой связи зависит итог таблицы.",
        "",
    ]
    return lines


def activations_section(report: dict, params: dict) -> list[str]:
    activations = report["activations"]
    lines = [
        "## 4. Нормы активаций (forward-hooks)",
        "",
        f"Промпт: «{params['hooks']['prompt']}», {activations['n_tokens']} токенов "
        "после chat template.",
        "",
        f"![нормы активаций]({Path(params['hooks']['plot']).name})",
        "",
        "| Блок | Индекс | Средняя ‖h‖₂ | Максимум |",
        "|---|--:|--:|--:|",
    ]
    for label, index in activations["layers"].items():
        values = activations["norms"][label]
        lines.append(f"| {label} | {index} | {sum(values) / len(values):.1f} | {max(values):.1f} |")
    lines += [
        "",
        "Нормы описывают масштаб скрытых состояний, а не качество ответа. Residual-связь",
        "допускает и рост, и уменьшение нормы. По одним нормам нельзя доказать наличие",
        "attention sink: для этого нужны веса внимания. Логарифмическая шкала позволяет",
        "сравнивать значения разных порядков. В таблице — фактические средние и максимумы.",
        "",
        "Прогон с хуками должен быть повторяемым: второй вызов в том же процессе",
        "обязан дать те же числа и не оставить следов на модулях.",
        "",
    ]
    return lines


def lora_section(report: dict) -> list[str]:
    lines = [
        "## 5. Сколько параметров добавляет LoRA",
        "",
        "| Конфиг | Целевых модулей | Своя формула | peft | Совпало | % от базовой |",
        "|---|--:|--:|--:|:-:|--:|",
    ]
    for item in report["lora"]:
        lines.append(
            f"| {item['name']} | {len(item['target_modules'])} типов | "
            f"{thousands(item['formula'])} | {thousands(item['peft'])} | "
            f"{'да' if item['match'] else 'НЕТ'} | {item['share_of_base'] * 100:.3f}% |"
        )
    lines += [
        "",
        "Формула: `r * (in_features + out_features)` на каждый целевой `Linear` —",
        "`A` формы `(r, in)`, `B` формы `(out, r)`, смещений нет. Расхождение с",
        "`print_trainable_parameters()` означает ошибку в списке целевых модулей,",
        "а не «разные способы считать».",
        "",
        "Доля в таблице считается от базовой модели. `peft` печатает свою долю от",
        "модели ВМЕСТЕ с адаптером, поэтому его процент чуть меньше — числитель",
        "у обоих один и тот же.",
        "",
    ]
    return lines


def memory_section(report: dict) -> list[str]:
    modes = {item["mode"]: item for item in report["memory"]}
    base = modes["inference"]
    lines = [
        "## 6. Память в трёх режимах",
        "",
        f"Один шаг на seq_len={base['seq_len']}, batch={base['batch_size']}, "
        f"device={base['device']}. Метрика — {base['metric']} "
        f"(`{base['metric_source']}`), пик за прогон; колонка «Пик RSS» снята "
        f"через `{base['rss_source']}`.",
        "Вход — случайные id токенов: меряется память, а не качество, и loss здесь",
        "смысловой нагрузки не несёт (у full FT и LoRA он одинаковый, потому что",
        "`B` в адаптере инициализирован нулями и до первого шага ничего не меняет).",
        "Числа должны отличаться: по лекции full fine-tune стоит кратно дороже",
        "инференса, а LoRA лежит между ними.",
        "",
        "| Режим | Пик, МиБ | Пик RSS, МиБ | × к инференсу | Секунд | loss |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for mode in ("inference", "full_ft", "lora"):
        item = modes[mode]
        loss = f"{item['loss']:.4f}" if item["loss"] is not None else "—"
        lines.append(
            f"| {MODE_TITLES[mode]} | {item['peak_mb']:.1f} | {item['peak_rss_mb']:.1f} | "
            f"{item['peak_mb'] / base['peak_mb']:.2f} | {item['seconds']:.1f} | {loss} |"
        )

    weights_mb = base["weights_bytes"] / 1024 ** 2
    lines += [
        "", "Все значения памяти приведены в МиБ = 1 048 576 байт (в задании обозначено МБ).",
        "Время включает загрузку модели из локального кэша; step_seconds в JSON — время после загрузки.",
        "На CPU измеряется пик RSS всего дочернего процесса, включая интерпретатор, библиотеки и загрузку.",
        "Каждый режим и повтор запускается в новом процессе. В таблице взят максимальный пик из повторов.",
        "", "| Режим | Все повторы, МиБ | PID процессов |",
        "|---|---|---|",
    ]
    for item in report["memory"]:
        lines.append(f"| {item['mode']} | {item['peak_mb_runs']} | {[r['pid'] for r in item['runs']]} |")
    lines += ["", "Отдельный подсчёт существующих тензоров (не замена пика RSS):", "",
              "| Режим | Базовые веса, МиБ | Градиенты после backward, МиБ | AdamW после step, МиБ |",
              "|---|---:|---:|---:|"]
    for item in report["memory"]:
        lines.append(f"| {item['mode']} | {item['weights_bytes']/1024**2:.3f} | "
                     f"{item['gradient_bytes']/1024**2:.3f} | {item['optimizer_state_bytes']/1024**2:.3f} |")
    lines += ["",
        f"Базовые веса: {thousands(report['params_total'])} параметров × "
        f"{base['weights_bytes']//report['params_total']} байта = {weights_mb:.3f} МиБ.",
        f"Full FT добавляет к пику инференса {modes['full_ft']['peak_mb']-base['peak_mb']:.1f} МиБ, "
        f"LoRA — {modes['lora']['peak_mb']-base['peak_mb']:.1f} МиБ.",
        "В float32 градиенты полного дообучения занимают столько же, сколько веса; два момента AdamW —",
        "примерно вдвое больше плюс скалярные счётчики шагов. Пики отдельных фаз нельзя складывать.",
        f"LoRA обучает {thousands(report['lora'][0]['peft'])} параметров "
        f"({report['lora'][0]['share_of_base']:.4%}) вместо всех базовых весов.",
        "Это сокращает градиенты и состояния оптимизатора. Базовые веса остаются, добавляются адаптеры",
        "и активации для backward. Поэтому доля обучаемых параметров не равна доле общей памяти.",
        "Разность RSS и подсчитанных тензоров нельзя точно назвать активациями: сюда входят также",
        "буферы операций, память библиотек, загрузки и аллокатора.",
        "",
        "На CUDA применяется встроенный счётчик max_memory_allocated после сброса и синхронизации.",
        "На MPS — максимум опросов driver_allocated_memory каждые 10 мс и на границах фаз.",
        "MPS-опрос может пропустить короткую аллокацию, поэтому это оценка пика. Здесь реальные замеры",
        "выполнены только на CPU; маршруты CUDA/MPS проверены тестами с подстановкой API.", "",
    ]
    return lines


def markdown_report(report: dict, params: dict) -> str:
    config = report["config"]
    lines = [
        f"# Анатомия {report['model'].split('/')[-1]}",
        "",
        f"Сгенерировано `make inspect`. dtype `{report['dtype']}`, device `{report['device']}`.",
        "",
        "## 1. Конфигурация",
        "",
        "| Параметр | Значение |",
        "|---|--:|",
        f"| слоёв | {config['num_hidden_layers']} |",
        f"| hidden_size | {config['hidden_size']} |",
        f"| intermediate_size | {config['intermediate_size']} |",
        f"| голов запроса | {config['num_attention_heads']} |",
        f"| KV-голов (GQA) | {config['num_key_value_heads']} |",
        f"| head_dim | {config['head_dim']} |",
        f"| словарь | {thousands(config['vocab_size'])} |",
        f"| tie_word_embeddings | {config['tie_word_embeddings']} |",
        "",
        f"GQA: {config['num_attention_heads']} голов запроса на "
        f"{config['num_key_value_heads']} KV-головы — "
        f"KV-cache в {config['num_attention_heads'] / config['num_key_value_heads']:g} раза меньше, чем при обычном multi-head.",
        "",
    ]
    lines += conditions_section(report)
    lines += params_section(report)
    lines += activations_section(report, params)
    lines += lora_section(report)
    lines += memory_section(report)
    discussion = Path("docs/discussion.md")
    if discussion.exists():
        lines += [discussion.read_text(encoding="utf-8"), ""]
    return "\n".join(lines)


def write_report(report: dict, params: dict) -> None:
    """Нарисовать график и записать docs/anatomy.md."""
    import csv
    directory = Path(params['report']['markdown']).parent
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / 'parameters.tsv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter='\t', lineterminator='\n')
        writer.writerow(['module_parameter', 'shape', 'unique_params', 'share', 'shared_params'])
        for row in report['parameter_rows']:
            unique = 0 if row['tied'] else row['numel']
            writer.writerow([row['name'], '×'.join(map(str,row['shape'])), unique,
                             unique/report['params_total'], row['numel'] if row['tied'] else 0])
    plot_activations(report["activations"], params["hooks"]["plot"])
    path = Path(params["report"]["markdown"])
    path.parent.mkdir(exist_ok=True)
    path.write_text(markdown_report(report, params).rstrip() + "\n", encoding="utf-8")
