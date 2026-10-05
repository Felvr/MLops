"""Generate the five-defect report only from measured training artifacts."""
import json
from pathlib import Path

from src.config import load_params
from src.train import inputs_fingerprint


def main():
    params = load_params()
    metrics_dir = Path(params['paths']['metrics'])
    runs = {}
    fingerprint = inputs_fingerprint(params)
    for name in ('all_layers', 'freeze14'):
        path = metrics_dir / f'train_{name}.json'
        if not path.exists():
            raise SystemExit(f'Нет {path}. Сначала make train.')
        run = json.loads(path.read_text(encoding='utf-8'))
        if run['inputs_fingerprint'] != fingerprint:
            raise SystemExit(f'{path} устарел: код, входные данные или параметры изменились. Повторите make train.')
        if not Path(run['adapter_dir']).is_dir():
            raise SystemExit(f"Нет {run['adapter_dir']}. Повторите make train.")
        runs[name] = run
    a, f = runs['all_layers'], runs['freeze14']
    if len(a['curve_val']) < 3 or len(f['curve_val']) < 3:
        raise SystemExit('Нет базовой, промежуточной и финальной точек val loss.')
    if not a['final_val_loss'] < a['base_val_loss']:
        raise SystemExit('Финальный val loss не улучшился — проверьте эксперимент, отчёт не создаётся.')
    if not f['trainable_params'] < a['trainable_params']:
        raise SystemExit('Заморозка не уменьшила число обучаемых параметров.')
    compare = metrics_dir / 'compare_all_layers.json'
    if not compare.exists():
        raise SystemExit(f'Нет {compare}. Сначала make compare.')
    c = json.loads(compare.read_text(encoding='utf-8'))
    if len(c['base']) != 5 or len(c['adapter']) != 5:
        raise SystemExit('В сравнении должно быть пять пар ответов.')
    changed = sum(x != y for x, y in zip(c['base'], c['adapter']))
    true_probes = [c for c in ('tokenizer_config.json', 'tokenizer.json', 'adapter_config.json', 'adapter_model.safetensors')
                   if (Path(a['adapter_dir']) / c).exists()]
    if len(true_probes) != 4:
        raise SystemExit('Адаптер не содержит все файлы для переноса.')
    baseline = a['curve_val'][0][1]
    lines = [
        '# Пять дефектов исходной заготовки', '',
        'Источник чисел: собственные прогоны на данных MedQuAD из ДЗ4, `metrics/train_*.json` и '
        '`metrics/compare_all_layers.json`. Сырые метрики остаются локально; воспроизводимый код и '
        f'конфигурация входят в репозиторий. В эксперименте {a["train_examples"]} train и '
        f'{a["val_examples"]} val примеров из собственных тензоров ДЗ4. '
        f'Режим запуска: {a["device"]}, {a["dtype"]}, {a["steps"]} шагов, '
        f'эффективный батч {a["effective_batch"]}, seed {a["seed"]}. '
        'При `max_steps` не равном `null` это ограниченный эксперимент, не полная эпоха. '
        f'Val считается на выбранных {a["val_examples"]} примерах на шаге 0, промежуточном шаге и в конце. '
        'Если включён data.subset, это малый эксперимент на коротких целых примерах; '
        'происхождение и выбранные ID записаны в docs/data.md.', '',
        '## 1. Обучение без валидации', '',
        'В заготовке `base_val` оставался `None`, а `curve_val` пустым. Поэтому даже падающий '
        'train loss ничего не говорил о поведении на отложенном сплите: модель могла запоминать '
        'обучающие ответы или деградировать. Я добавил `evaluate()` до первого шага, по ходу '
        'обучения и в конце. Лосс суммируется с весом числа токенов ответа, поэтому короткий '
        'и длинный батч не получают одинаковый вес. Маска prompt и padding не входит в знаменатель. '
        f'Фактический val loss `all_layers`: шаг 0 — {baseline:.6f}, '
        f'шаг {a["curve_val"][1][0]} — {a["curve_val"][1][1]:.6f}, '
        f'финал — {a["final_val_loss"]:.6f}; точек {len(a["curve_val"])}. '
        'Шаг 0 даёт честную базовую точку той же модели до LoRA-обновления.', '',
        '## 2. Завышенная скорость обучения', '',
        'В `params.yaml` стояло `lr: 1e-2`. Для этого LoRA-эксперимента такой шаг может '
        'вызвать взлёт лосса до десятков, хотя программа не завершается ошибкой и сохраняет '
        'адаптер. Я поставил `2e-4` и сохранил те же `r=8`, `alpha=16`, clipping и scheduler. '
        'Менять параметры после получения результатов нельзя без нового прогона: проверка '
        'сопоставляет отпечаток входов и кода. '
        f'На собственном прогоне train loss первого шага {a["curve_train"][0][1]:.6f}, '
        f'последнего {a["curve_train"][-1][1]:.6f}; val до/после '
        f'{a["base_val_loss"]:.6f} → {a["final_val_loss"]:.6f}. '
        'Это эмпирическая проверка сходимости на выбранном числе шагов, а не гарантия '
        'медицинской точности ответов.', '',
        '## 3. Заморозка только на бумаге', '',
        'Поле `freeze_first` в исходной заготовке передавалось в функцию, но не влияло на '
        '`LoraConfig`: адаптеры оставались во всех 28 слоях. Я задаю `layers_to_transform` '
        'точным диапазоном от нужного слоя до последнего и проверяю диапазон. '
        f'Теперь обучаемых параметров {a["trainable_params"]:,} для всех слоёв и '
        f'{f["trainable_params"]:,} при заморозке первых 14. '
        f'Чистое время обучения {a["seconds"]:.1f} и {f["seconds"]:.1f} с; '
        f'память {a["peak_memory_mb"]:.1f} и {f["peak_memory_mb"]:.1f} МБ '
        f'по метрике `{a["memory_metric"]}`. Финальные val loss '
        f'{a["final_val_loss"]:.6f} и {f["final_val_loss"]:.6f}. '
        'Время может изменяться из-за фоновой нагрузки; главным доказательством заморозки '
        'служат конфигурация сохранённого адаптера и число обучаемых весов.', '',
        '## 4. Непереносимый адаптер', '',
        'Исходный `load_adapter_tokenizer()` читал токенизатор по имени базы из '
        '`params.yaml`, а обучение сохраняло только LoRA-веса. На другом компьютере '
        'получатель не получал версию токенизатора и шаблон чата вместе с адаптером. '
        'Теперь после `model.save_pretrained` выполняется '
        '`tokenizer.save_pretrained(adapter_dir)`, а локальная загрузка читает именно '
        'каталог адаптера. Веса базы всё равно нужны отдельно. '
        f'Размер полного каталога адаптера вместе с токенизатором — {a["adapter_size_mb"]:.2f} МБ; '
        f'на {len(c["base"])} фиксированных вопросах ответы базы и адаптера отличаются в '
        f'{changed} случаях. Это подтверждает подключение адаптера, а не проверяет '
        'фактическую правильность ответов. Окончательную офлайн-переносимость проверяет '
        'пункт 4 исходного `make check`, копируя каталог во временное место.', '',
        '## 5. Разные результаты при одинаковом конфиге', '',
        'В заготовке существовала функция `set_seed()`, но цикл её не вызывал перед '
        'созданием LoRA-матриц и загрузчиков. Я вызываю её до этих операций, '
        'фиксирую Python, NumPy и PyTorch, отключаю недетерминированные алгоритмы, '
        'а порядок примеров задаю локальным генератором с `seed + epoch`. '
        f'В обоих вариантах записан seed {a["seed"]}; '
        f'число шагов {a["steps"]}/{f["steps"]}. '
        'Число 42 в конфигурации само по себе не доказывает воспроизводимость: '
        'пункт 5 `make check` дважды обучает по 3 шага и сравнивает массив '
        '`curve_train` дословно. Проверку запускают на том же устройстве и в том же '
        'окружении; побитовое совпадение между разными устройствами не заявляется.', '',
        '## Ограничения и проверка', '',
        'Вопросы сравнения взяты из отложенного MedQuAD и не входят в градиенты; '
        'набор из пяти вопросов слишком мал для оценки медицинской достоверности. '
        'При включённом отборе коротких примеров меняется распределение train и val. '
        'Результаты такого CPU-прогона нельзя называть полным обучением '
        'на 1972 примерах. После `make check` следует сохранить лог и снять видео '
        'экрана с финальной строкой и отпечатком неизменённого `tests/check.sh` '
        '`06a02208c1f0`.', ''
    ]
    output = Path('docs/defects.md')
    output.write_text('\n'.join(lines), encoding='utf-8')
    print(f'-> {output}')


if __name__ == '__main__':
    main()
