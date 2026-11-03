# Сценарий одного видео 5–8 минут

Видео пользователь записывает самостоятельно. Все тяжёлые шаги уже выполнены,
их можно показывать по сохранённым журналам. Не показывать секреты и чужие приложения.

## 0:00–0:45 — структура и семь причин

Открыть README, `git log --oneline -- hw8910`, `defects.csv`.
«Это единая работа 8–10. Исходник сохранён в 38c2458. Семь причин: неверный cache key,
слабый gate contract, неатомарная запись, дубли публикаций, ранний ready,
неверная версия в метаданных и отсутствие проверки запросов».

## 0:45–1:40 — красный и зелёный прогон

Показать red-original.log: 7 тестов, все находят поведенческие ошибки.
В терминале из hw8910 выполнить `make check`: 24 проверки, без загрузки модели.
Показать red-mutation.log и exit 1. Объяснить, что мутация была в отдельной
копии, рабочий код восстановлен/не менялся; преподавательские тесты побайтно исходные.

## 1:40–2:50 — Prefect и восстановление

Открыть pipeline.py и flow-stub-failure/recovery/invalidation.log.
Показать Completed → failure, затем Cached в другом процессе, затем новый Completed
при изменении байтов по прежнему пути. Показать те же состояния в реальных
flow-real-failure/recovery-rejection/invalidation.log: здесь подготовка токенизирует.
Объяснить, что obtain_artifact читает готовый run, не обучает повторно.

## 2:50–3:45 — gate, атомарность и повтор

Открыть release-report.json: passed=false, причина Mandatory behavioral cases failed.
Назвать training/eval run ID. Показать source hashes и real-rejection.json:
SHA accepted до и после одинаковый. Baseline сохраняется, хотя её refusal тоже
не идеален. Показать flow-evidence.json и журналы pointer-crash/recovery/repeat:
два разных решения дали два события, повторы того же не размножили события.

## 3:45–5:20 — настоящий API

До записи запустить `make serve`, дождаться ready. В отдельном терминале:

```bash
curl -s http://127.0.0.1:8091/health
curl -s http://127.0.0.1:8091/ready
curl -s -H 'Content-Type: application/json' -d '{"prompt":"Read the excerpt: Example. Options: A. Example B. Other","max_new_tokens":100000}' http://127.0.0.1:8091/generate
```

Для повторяемого запроса из golden:

```bash
uv run --locked python -c 'import json,urllib.request; p=json.loads(open("evidence/golden.jsonl").readline())["input"]; r=urllib.request.Request("http://127.0.0.1:8091/generate",data=json.dumps({"prompt":p}).encode(),headers={"Content-Type":"application/json"}); print(urllib.request.urlopen(r).read().decode())'
```

Показать answer, request_id, model_id, runtime_sha256 и tokens. Реальный ответ
не содержит STUB ONLY. Открыть real-api.json: 503 до прогрева, 422 без вызова backend.
Показать real-restart.json. STUB A→B→A показать отдельно по http-integration.jsonl
и тесту: ID берётся из памяти, указатель не перезагружает модель автоматически.

## 5:20–6:40 — измерения и ограничения

Открыть benchmark.csv и requests.jsonl. 8+8, все 16 успешны, 6 токенов на ответ,
лимит 16. c=1: среднее 2,82 с; c=2: 5,25 с; requests/s около 0,35 в обоих режимах.
Один compute slot создаёт очередь, а не ускорение. Объяснить prefill/decode и
рост KV-cache. TTFT/ITL не измерены, p95 на восьми запросах не заявлен.

## 6:40–7:10 — комплект сдачи

Показать отчёт, воспроизводимые команды и доступ к публичной baseline revision.
Весов в Git нет, кандидатский адаптер сохранён локально в MLflow.
Сказать: «CI локальный; удалённые Actions и защита merge не заявляются».
После записи добавить путь/ссылку на видео в README и report.md и убрать пометку
«видео ещё не записано» только после фактического создания записи.
