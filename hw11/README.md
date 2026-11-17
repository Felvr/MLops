# ДЗ 11 — Kubernetes под нагрузкой

Продолжение API ДЗ 8+9+10 в отдельной папке общего репозитория.
CPU backend явно возвращает `backend_mode=simulated`. Все длительности
и графики в [отчёте](report.html) получены HTTP-запросами в локальном kind.

## Окружение

Нужны Docker, uv, kind 0.27.0, Helm 3.17.3, kubectl с поддержкой Kubernetes 1.32.
Проверенный стенд — Intel macOS, Docker Desktop, 8 CPU и 4 GiB выделенной RAM.
Для свежей установки нужно примерно 7 GiB дополнительного места в Docker
с запасом на node, образы и runtime. Фактические размеры сохранены
в `results/environment.json`.

Команды выполняются из `hw11/`. Скрипты используют отдельный
`KUBECONFIG=$PWD/.local/kubeconfig` и кластер `mlops-hw11`.

```bash
uv sync --locked --group report
bash scripts/setup.sh
make test
make experiments
make report
make check
```

`setup.sh` загружает закреплённые версии образов и metrics-server 0.7.2.
Для локального kind metrics-server получает `--kubelet-insecure-tls`.
Запуск опытов занимает около 35–45 минут после загрузки образов.

## Эксперименты

```bash
uv run --locked python scripts/experiment.py A
uv run --locked python scripts/experiment.py B
uv run --locked python scripts/experiment.py C
```

A проверяет исходный дефект probes, его исправление, rollout A→B, выпуск
B с неверным SHA-256, настоящий Helm rollback, замену Pod, продолжение
и изоляцию диалогов, SIGKILL и корректный учёт оборванных потоков.

B выполняет 12 прогонов последовательно — 2 Service × 2 распределения
поступлений × 3 повтора, каждый по 120 с. Четыре клиентских Pod
с разными IP отправляют по сохранённому расписанию. Keep-alive выключен.
Перед каждой фазой локальный кеш сбрасывается через SIGUSR1.
Два Service остаются над одним Deployment.

C включает CPU HPA 2–3 реплики, создаёт вычислительную нагрузку,
наблюдает готовность новой реплики, алерты и возврат к двум репликам.

Для полного повторного опыта переместите предыдущие измерения
в игнорируемую папку, затем восстановите исходные values.

```bash
mkdir -p .local
mv results ".local/results-$(date +%Y%m%d-%H%M%S)"
mkdir results
bash scripts/setup.sh
make test experiments report check
```

Скрипты добавляют JSONL-записи. Проверка запрещает дубли request ID
и неполный комплект измерений. Отдельные повторы B для отладки
задаются `--reps 1`, итоговый отчёт требует три.

## Просмотр

```bash
export KUBECONFIG="$PWD/.local/kubeconfig"
kubectl --context kind-mlops-hw11 -n hw11 port-forward svc/hw11-grafana 13000:3000
```

Откройте `http://localhost:13000/d/mlops-hw11`. Анонимный доступ имеет
роль Viewer, сервис доступен через локальный port-forward.
Дашборд provisioned из `chart/monitoring/dashboard.json`.

Prometheus и API доступны отдельными командами в другом терминале.

```bash
kubectl --context kind-mlops-hw11 -n hw11 port-forward svc/hw11-prometheus 19090:9090
kubectl --context kind-mlops-hw11 -n hw11 port-forward svc/hw11-plain 18080:8080
```

```bash
curl -N http://localhost:18080/generate \
  -H 'Content-Type: application/json' -H 'X-Request-ID: video-example' \
  -d '{"prompt":"Explain readiness","history":[{"role":"user","content":"CONTROL-731"}],"session_id":"demo","stream":true,"max_new_tokens":4}'
```

`max_new_tokens` сохраняет имя поля прежнего API и ограничивает число
частей имитатора. В метриках используются TTFC и inter-chunk latency.
Для обычного JSON-ответа передайте `stream:false` или пропустите поле.
Ответ содержит `answer`, `model_id`, SHA-256 и Pod.

## Файлы сдачи

- `report.html` и четыре графика в `results/figures/`.
- `results/requests.jsonl` — одна запись на попытку, включая ошибки и обрывы.
- `results/schedule.jsonl` — план поступлений, входы, число частей, клиент.
- `results/timeline.jsonl` — Ready, EndpointSlice, HPA, состояния алертов.
- `results/checks.json` — собственные проверки через Kubernetes Service.
- `results/B-*-metrics.json`, `C-metrics.json` — выгрузки Prometheus.
- `chart/`, Dockerfile, `uv.lock` и тесты.
- [Сценарий самостоятельной записи видео](video.md).

Файл модели — небольшой JSON в immutable ConfigMap.
Процесс проверяет SHA-256 и model_id перед прогревом. Все реплики получают
одинаковый источник без PVC и ограничений ReadWriteOnce.

## Удаление стенда

```bash
bash scripts/setup.sh delete
docker image rm mlops-hw11:1.0.0
```

Кластер целиком принадлежит этому ДЗ. Команда kind удаляет его контейнер
и внутренние emptyDir. Образы мониторинга и kind можно удалить отдельно,
если они больше не нужны.

```bash
docker image rm kindest/node:v1.32.2 prom/prometheus:v3.2.1 \
  grafana/grafana:11.6.0 \
  registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.15.0 \
  registry.k8s.io/metrics-server/metrics-server:v0.7.2
```

Размер TSDB ограничен 128 MB, retention 8 часов. Grafana emptyDir — 128 MiB.
После удаления кластера сохранённые сырые данные и отчёт остаются в репозитории.
