# ДЗ 8+9+10 — процесс выпуска, проверки и API

Исправленный проект в общем репозитории, ветка `MLops`. Исходный прототип:
коммит `38c2458`. [Отчёт](docs/report.md), [7 дефектов](defects.csv),
[сценарий видео 5–8 минут](docs/video.md).

Кандидат `14bf51e70f58417291ef58de4d1d958b` из ДЗ6+7 **отклонён** прежним
`quality_gate`: `Mandatory behavioral cases failed`. API обслуживает ранее
закреплённую baseline Qwen3-0.6B, CPU float32. У baseline тоже известен провал
`behavior-refusal`; сохранение прежней версии не выдаётся за новый PASS.
Повторного обучения и ослабления gate нет.

## Быстрые проверки без весов

Python 3.12, uv. Все команды ниже выполняются из `hw8910/`.

```bash
uv sync --locked
make check                  # 13 исходных + 7 интеграционных + 4 дополнительных
uv run --locked python scripts/regression_proof.py
make mutation               # намеренно exit 1; рабочий код не изменяется
```

`make check` не импортирует torch/Transformers, не загружает веса и не требует
Docker или внешней сети после установки зависимостей. `release.py` и 13 исходных
проверок работают также на стандартной библиотеке:
`python -m unittest discover -s tests -p test_release.py -v`.
Тесты преподавателя сохранены побайтно, fingerprint `09bd65e6b886`.
Логи: [зелёный](docs/green-tests.log), [исходник](docs/red-original.log),
[намеренная поломка](docs/red-mutation.log), [исправления](docs/fixes.diff).

## Реальные артефакты и модель

Установить окружение с моделью: `uv sync --locked --extra model`.
Текстовые eval-данные, исходные ответы и отчёты уже находятся в `evidence/`.
Весов в Git нет. Для воспроизведения кандидата нужен сохранённый MLflow volume
ДЗ6+7 и доступ к Qwen/Qwen3-0.6B на Hugging Face:

```bash
# из корня репозитория — поднять сохранённое хранилище ДЗ6+7
# Не применять down -v: это удалит единственную локальную копию артефактов.
docker compose -p mlops-hw6 -f hw67/compose.yaml up -d
cd hw67
uv run --locked python ../hw8910/scripts/export_hw67.py
cd ../hw8910
# Если базы нет в HF cache, скачать ровно закреплённую ревизию:
uv run --locked --extra model python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca')"
make bootstrap
make serve
```

`export_hw67.py` проверяет текущие отчёты ДЗ6+7 и читает готовый training run,
**не обучает**. `bootstrap.py` пересчитывает файлы, заново проверяет протокол,
пересчитывает score из сырых ответов и вызывает неизменённый gate ДЗ6+7.
`release-report.json` создаётся кодом, сейчас содержит `passed: false`.
Начальный `accepted.json` сохраняет baseline; bootstrap никогда не подменяет
существующий указатель кандидатом. `baseline-retention.json` объясняет решение.

Для преподавателя полный разбор gate доступен из Git без приватного DVC remote.
Baseline скачивается по публичной точной revision. Чтобы запустить **baseline API**
без MLflow и отвергнутого адаптера, достаточно загрузить эту revision и создать
локальную ссылку `artifacts/base` на её snapshot командой `uv run --locked --extra model python scripts/restore_base.py`.
Затем `make serve`. Кандидатские веса доступны только в локальном MLflow volume;
передача преподавателю отдельно через владельца, публичная ссылка не создана.
Сохранённые manifests позволяют сверить переданные файлы.

## API и ограничения

```bash
curl http://127.0.0.1:8091/health
curl http://127.0.0.1:8091/ready
curl -H 'Content-Type: application/json' -d '{"prompt":"Read the excerpt: Example. Options: A. Example B. Other"}' http://127.0.0.1:8091/generate
```

Загрузка/прогрев идут в фоне, до ready генерация получает 503. Вход ограничен
16000 символами, 4096 токенами с системным сообщением и телом 64 KiB.
Лимит ответа **ровно 16 новых токенов**, как в eval; другой лимит получает 422.
Вычисление последовательное (один слот), очередь ожидает в потоке. Ответ включает
`answer`, `request_id`, фактические `model_id`, `runtime_sha256`, `output_tokens`,
`queue_seconds`, `generation_seconds`, `total_seconds`. Смена указателя не меняет
модель в памяти. `/debug` по умолчанию только на STUB; реальному API доступен
лишь при явном `--debug`. Сервер привязан к localhost.

## Prefect и доказательные опыты

```bash
make experiments            # независимые subprocess; реальные Prefect states
make flow                   # сейчас ожидаемый exit 1: кандидат отклонён
make benchmark              # настоящий backend, прогрев + 8 запросов при c=1 и c=2
uv run --locked --extra model python scripts/input_invalidation.py
uv run --locked --extra model python scripts/restart_smoke.py
```

`pipeline.py`: schema/token preparation → получение существующего артефакта →
проверка исходных eval + gate → публикация. В результатах задач только описания
URI/hash. Prefect SQLite в `.prefect/`, результаты в `.state/prefect-results/`,
подготовка в `.state/prepared/`. Эти локальные каталоги игнорируются.
Синтетические стенды создаются во временных каталогах и помечены STUB ONLY.
Серия опытов использует один локальный Prefect API server, процессы flow отдельные.
Для каждого запуска сохранены состояния, run IDs и exit codes в `docs/flow-*.log`.

Публикация рассчитана на одного писателя: intent → atomic pointer → atomic journal
snapshot → удаление intent. Повтор завершает незавершённое событие, не дублируя его.
Отказ схемы/качества и несовместимые файлы не повторяются автоматически.
Скачивание артефакта выполняется отдельно; временный сетевой сбой можно повторить
командой, выпуск не маскирует его бесконечными retry.

## Триггеры проверок

| Изменение | Команды и действие |
|---|---|
| Код | `make check`; при изменении генерации — полная `make evaluate && make eval` в ДЗ6+7 с новым протоколом, затем bootstrap |
| Данные | `make experiments` проверяет инвалидацию; `make flow` проверяет схему и токены. Новый golden требует повторного eval; новые train-данные требуют решения о новом обучении, не автоматической подмены gate |
| Новый адаптер | Полная оценка через ДЗ6+7 (`make candidate && make evaluate && make eval`), экспорт, `make bootstrap`, `make flow`; загрузка только после PASS |

CI выполнен **локально**. Удалённый Actions не запускался, защита merge не заявляется.
Видео пользователь записывает по готовому сценарию. GitHub push не выполнялся.
