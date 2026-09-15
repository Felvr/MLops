# ДЗ2 — анатомия модели

Разбор SmolLM2-135M-Instruct: параметры по модулям, forward-хуки, два конфига LoRA
и память инференса / полного дообучения / LoRA. Исправлены четыре дефекта заготовки.

Основной результат — [отчёт](docs/anatomy.md), [график](docs/activations.png)
и [заполненная сводная таблица](docs/anatomy-worksheet.xlsx).
Модель содержит **134 515 008** уникальных параметров. Адаптеры добавляют
**460 800** и **4 884 480** обучаемых параметров соответственно.

## Воспроизведение

Нужны Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/),
Git и make/bash. На Windows команды make запускаются в Git Bash либо WSL.

```bash
uv sync --locked
make inspect
make check
make test
```

Первый запуск скачивает safetensors и токенизатор из Hugging Face. Веса в Git не
хранятся. После загрузки можно запускать с `HF_HUB_OFFLINE=1`. Версия модели закреплена
через revision в params.yaml, зависимости — в uv.lock. Все семь проверок make check
должны пройти. make test дополнительно проверяет общие веса, очистку хуков при ошибке
и правильное использование API ускорителей на маленьких искусственных данных.

Прямые команды без make:

```bash
uv run python -m src.inspect_model
bash tests/check.sh
uv run python -m unittest discover -s tests -v
```

## Условия сохранённого эксперимента

Intel Core i9, MacBook Pro, 16 ГБ RAM, macOS 26.5.1; CPU, float32, seq_len=256,
batch=1, 4 потока PyTorch. Три отдельных процесса на каждый режим; в отчёт берётся
максимальный пик RSS. KV-cache и gradient checkpointing выключены. Время включает
загрузку из кэша; время самого шага отдельно есть в JSON. Единица памяти — МиБ.

SmolLM2-135M — разрешённая условием замена Qwen3 для ограниченного железа.
Все расчёты сделаны для выбранной модели, а не скопированы со слайдов Qwen3.
На Intel macOS используются torch 2.2.2 и numpy 1.26.4; на других платформах
lock выбирает torch 2.5.1. Эта совместимость нужна из-за прекращения готовых
сборок PyTorch для Intel macOS после ветки 2.2.

Настройки меняются в params.yaml. При смене модели измените также её revision.
Для Qwen3-0.6B: name=`Qwen/Qwen3-0.6B`, revision=`c1899de289a04d12100db370d81485cdf75e47ca`.
На ускорителе выберите подходящие device и dtype. После изменения конфигурации
повторите измерения и обновите сводную XLSX и пояснения, относящиеся к модели.
Аппаратные ветки CUDA/MPS в этом репозитории не измерялись на реальном ускорителе.

## Файлы для сдачи

- [docs/anatomy.md](docs/anatomy.md) — таблицы, график, условия, расчёты и разбор дефектов.
- [docs/anatomy-worksheet.xlsx](docs/anatomy-worksheet.xlsx) — пять листов с числами и контрольными формулами, оба конфига LoRA.
- [docs/report.json](docs/report.json) — полные результаты, все повторы и PID.
- [docs/parameters.tsv](docs/parameters.tsv) — каждый тензор, shape, число параметров и доля.
- [docs/baseline-check.txt](docs/baseline-check.txt) — ошибки исходной версии.
- [docs/check.txt](docs/check.txt) — вывод окончательного make check.
- [docs/check.mp4](docs/check.mp4) — видео воспроизведения настоящей записи терминала.
- [docs/check.cast](docs/check.cast) — исходная запись терминала с временными метками.
- [docs/discussion.md](docs/discussion.md) — объяснения, которые включаются в отчёт при make inspect.
- [uv.lock](uv.lock) — зафиксированное окружение.

make inspect перезаписывает JSON, график, TSV и anatomy.md, но сохраняет отдельный
разбор в discussion.md. XLSX — снимок сданного эксперимента: автоматически не
перезаписывается, её числа должны соответствовать тому же JSON.

## Запись проверки

```bash
uv run python scripts/record_check.py
```

Команда реально запускает make check в псевдотерминале и сохраняет check.cast и
check.txt. Для видео дополнительно нужен ffmpeg:

```bash
uv run python scripts/render_recording.py
```

Видео показывает терминал целиком, воспроизводит зафиксированный вывод без изменения
результатов и временного порядка. Оно не является записью рабочего стола.

## Репозиторий и сдача

Эта работа находится в папке `hw2/` общего репозитория
[MLops](https://github.com/Felvr/MLops), в ветке `MLops`.
Команды запуска выше выполняются из `hw2/`, а Git-команды — из корня проекта.
Отдельный репозиторий или вложенная папка `.git` для работы не нужны.

Для сдачи используйте ссылку на папку `hw2/` в ветке `MLops` и видео
`docs/check.mp4`. Инструкция отправки всей ветки находится в [корневом README](../README.md).
