# ДЗ 6+7 — эксперименты и проверка качества Qwen3 LoRA

Единая работа в `hw67/` корневого репозитория MLops. Код, четыре исправления,
scorer, gate и преподавательские проверки находятся здесь. Итоговые измерения
и решение о выпуске: [docs/conclusion.md](docs/conclusion.md).
[Протокол](docs/protocol.md), [дефекты](docs/defects.md),
[сравнение](docs/comparison.csv), [сценарий видео](docs/video.md).

## Окружение

Python 3.12, uv, Docker Desktop/Compose v2. На Intel Mac — CPU float32,
torch 2.2.2; на остальных платформах — torch 2.5.1. База Qwen3-0.6B
закреплена commit c1899de289a04d12100db370d81485cdf75e47ca.
Базовые веса скачиваются в HF cache, не входят в Git.

```bash
cd hw67
uv sync --locked
make check-eval
make up
```

MLflow UI: http://127.0.0.1:5006. PostgreSQL и артефакты — разные named volumes.
`make down` сохраняет их. `docker compose down -v` удаляет результаты;
для обычной остановки используйте `make down`.

Для полного повторения одной командой: `make repro` (обучение, оценка и проверки).
Для завершения уже готового sweep без повторного обучения: `make finish`.
Отказ по качеству сохраняется; остальные ошибки останавливают маршрут.

## Данные и воспроизведение

`make prepare` требует локальные результаты ДЗ3, ДЗ4 и ДЗ5:
`hw3/data/{train,test,provenance}.jsonl`, `hw4/data/tokenized/*.pt`,
`hw5/data/tokenized/*.pt`. Он проверяет, что 32/8 целых примеров ДЗ5 совпадают
с ДЗ4, переносит тензоры побайтово, экспортирует все 1972 исходных train-вопроса
и воспроизводит 50 golden-вопросов и три behavioral-кейса.
Если данных нет, сначала пройдите README соответствующих работ.

```bash
make prepare
make data-version
make sweep
make report
make candidate
make baseline-init
make evaluate
make eval
make check-negative
```

`make eval` при отказе завершится ненулевым кодом. Это допустимый результат,
он записан в metrics/quality-gate.json. После отказа переходите к проверкам
хранения, **не запускайте promote для обхода gate**. Только если gate PASS:

```bash
make promote
```

Затем для любого результата:

```bash
make check-storage
make report-final
make check-all
```

`baseline-init` вызывается один раз. При повторной оценке сохраните
`evaluation/baseline.json`: это закреплённая исходная база, а не новый кандидат.
`baseline-update` выполняется явно только после принятия champion и сохраняет
историю. Полная оценка охватывает только baseline и лучший loss-кандидат.

## Доступ преподавателя

Golden, behavioral, полный train-index и исходные выдержки с разметкой
версионированы в `data/eval.dvc`. Чтобы изучить **готовую сдачу**, приватный
DVC remote и тензоры не нужны: комплект JSONL приложен в
[docs/evaluation-data.zip](docs/evaluation-data.zip). Восстановление и проверка:

```bash
make restore-eval
uv run dvc status data/eval.dvc
```

DVC сообщит, что данные не изменились. Архив содержит только текстовые данные
оценки и источники, без весов, секретов и тензоров. `data-version` обновляет
DVC-указатель и приложение после осознанного изменения набора.
Полные сырые ответы, per-item, diff, гейты и storage-check находятся в `metrics/`
и включаются в Git. Загрузка модели в новом процессе проверяется `make check`
по alias candidate без локального params.yaml и с HF_HUB_OFFLINE=1.

Тесты преподавателя не изменены; ожидаемый fingerprint `c42ac7788026`.
Все Git-команды выполняются из корня MLops, вложенной `.git` нет.
Видео 5–8 минут пользователь записывает самостоятельно по docs/video.md.
