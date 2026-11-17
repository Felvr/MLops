"""Build the submission report and four figures from measured JSONL only."""
from collections import Counter, defaultdict
from datetime import datetime
import html
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
                     "grid.alpha": .2, "figure.dpi": 150})


def readlines(name):
    return [json.loads(line) for line in (RESULTS / name).read_text().splitlines() if line]


def quantile(values, fraction):
    values = sorted(value for value in values if value is not None and math.isfinite(value))
    if not values:
        return None
    index = (len(values) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(v))}</th>" for v in headers) + (
        "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
                                      + "</tr>" for row in rows) + "</tbody></table>"
    )


def save(fig, name):
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def figure(name, caption):
    return f'<figure><img src="results/figures/{name}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def bucket_quantile(before, after, fraction):
    previous = {(row["metric"]["pod"], row["metric"]["le"]): float(row["value"][1])
                for row in before}
    deltas = defaultdict(float)
    for row in after:
        pod, bound = row["metric"]["pod"], row["metric"]["le"]
        deltas[float(bound)] += float(row["value"][1]) - previous.get((pod, bound), 0)
    total = deltas[math.inf]
    rank, lower, cumulative = fraction * total, 0, 0
    for upper, count in sorted(deltas.items()):
        if rank <= count:
            value = lower if math.isinf(upper) else lower + (upper-lower) * (rank-cumulative) / max(1, count-cumulative)
            return total, value
        lower, cumulative = upper, count
    return total, None


requests = readlines("requests.jsonl")
schedule = readlines("schedule.jsonl")
timeline = readlines("timeline.jsonl")
checks = json.loads((RESULTS / "checks.json").read_text())
phases = defaultdict(list)
for row in requests:
    phases[row["phase"]].append(row)
if len({r["id"] for r in requests}) != len(requests):
    raise ValueError("Duplicate request IDs. Archive previous results before rerunning.")
if {r["id"] for r in requests} != {r["id"] for r in schedule}:
    raise ValueError("Planned attempts and client results differ.")
bphases = sorted(name for name in phases if name.startswith("B-"))
expected_b = {f"B-{traffic}-{policy}-{rep}" for traffic in ("uniform", "80pct")
              for policy in ("plain", "sticky") for rep in (1, 2, 3)}
if set(bphases) != expected_b or not all(check["passed"] for check in checks):
    raise ValueError("The required comparisons or checks are incomplete.")

summary, per_pod = [], []
for name, rows in sorted(phases.items()):
    ok = [r for r in rows if r["outcome"] == "completed"]
    outcomes = Counter(r["outcome"] for r in rows)
    intervals = [(b-a) for r in ok for a,b in zip(r["chunk_monotonic"], r["chunk_monotonic"][1:])]
    summary.append(dict(phase=name, planned=len(rows), sent=len(rows), completed=len(ok),
                        rejected=outcomes["rejected"], aborted=outcomes["aborted"],
                        timed_out=outcomes["timed_out"], error_fraction=1-len(ok)/len(rows),
                        ttfc_p50=quantile([r["ttfc_s"] for r in ok], .5),
                        ttfc_p95=quantile([r["ttfc_s"] for r in ok], .95),
                        total_p50=quantile([r["elapsed_s"] for r in ok], .5),
                        total_p95=quantile([r["elapsed_s"] for r in ok], .95),
                        inter_chunk_p95=quantile(intervals, .95),
                        send_lag_p95=quantile([r["send_lag_s"] for r in rows], .95),
                        send_lag_max=max(r["send_lag_s"] for r in rows)))
    pod_names = {r["pod"] for r in rows if r["pod"]}
    if name.startswith("C-cpu"):
        interval_start = min(r["sent_at"] for r in rows)
        interval_end = max(r["ended_at"] for r in rows)
        pod_names.update(state["name"] for sample in timeline
                         if sample["phase"]=="C-hpa" and interval_start<=sample["ts"]<=interval_end
                         for state in sample.get("state", [])
                         if state["kind"]=="Pod" and not state["deletion"])
    for pod in sorted(pod_names):
        pr = [r for r in rows if r["pod"] == pod]
        completed = [r for r in pr if r["outcome"] == "completed"]
        per_pod.append(dict(phase=name, pod=pod, arrived=len(pr), completed=len(completed),
                            ttfc_p95=quantile([r["ttfc_s"] for r in completed], .95),
                            queue_p95=quantile([r["queue_seconds"] for r in completed], .95),
                            cache_hit_fraction=sum(bool(r["cache_hit"]) for r in completed)/max(1,len(completed))))
(RESULTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
(RESULTS / "per-pod.json").write_text(json.dumps(per_pod, ensure_ascii=False, indent=2)+"\n")
lookup = {row["phase"]: row for row in summary}

# A. Readiness, actual endpoints and successful/failed requests share a wall-clock axis.
names = ("A-warm-broken", "A-warm-fixed", "A-drain-proof")
fig, axes = plt.subplots(3, 1, figsize=(11, 10), constrained_layout=True)
for ax, name in zip(axes, names):
    states = [r for r in timeline if r["phase"] == name and "state" in r]
    origin = min(r["ts"] for r in states)
    xs = [r["ts"] - origin for r in states]
    ready = [sum(s["kind"]=="Pod" and (s["ready"] or {}).get("status")=="True"
                 and not s["deletion"] for s in r["state"]) for r in states]
    endpoints = [sum(e.get("ready",False) for s in r["state"]
                     if s["kind"]=="EndpointSlice" and s["service"]=="hw11-plain"
                     for e in s["endpoints"]) for r in states]
    ax.step(xs, ready, where="post", label="Ready Pods")
    ax.step(xs, endpoints, where="post", label="Ready endpoints", linestyle="--")
    not_ready = [sum(s["kind"]=="Pod" and (s["ready"] or {}).get("status")!="True"
                     and not s["deletion"] for s in r["state"]) for r in states]
    ax.step(xs, not_ready, where="post", label="Not Ready Pods", color="#8657a6", linestyle="-.")
    phase_rows = (phases["A-drain-service"]+phases["A-drain-old-0"]+phases["A-drain-old-1"]
                  if name=="A-drain-proof" else phases[name])
    for success, color, label in ((True,"#16855b","Completed"), (False,"#bc3a35","Rejected / interrupted")):
        points = [r for r in phase_rows if (r["outcome"]=="completed")==success]
        ax.scatter([r["sent_at"]-origin for r in points], [3.25]*len(points), s=13, color=color, label=label)
    if name=="A-drain-proof":
        drain_proof = json.loads((RESULTS / "drain-proof.json").read_text())
        for index,stream in enumerate(drain_proof["streams"]):
            ax.plot([stream["first_at"]-origin,stream["ended_at"]-origin],[3.45-index*.15]*2,
                    linewidth=3,label="Old stream lifetime" if index==0 else None)
            if stream["crossed"]:
                ax.axvline(stream["terminating_observed_at"]-origin,color="#bc3a35",
                           linestyle="--",label="Old Pod terminating")
    new_creations = [datetime.fromisoformat(s["created"].replace("Z","+00:00")).timestamp()
                     for sample in states for s in sample["state"] if s["kind"]=="Pod"]
    later = [value for value in new_creations if value>=origin]
    if later:
        ax.axvline(min(later)-origin,color="#727b89",linestyle=":",label="New Pod created")
    ax.set(title=name, xlabel="Time from observation start, s", ylabel="Replica count",
           yticks=[0,1,2,3], ylim=(-.2,3.7))
    ax.legend(loc="lower right", ncol=2, fontsize=8)
save(fig, "01-rollout.png")

# B. Actual per-request queue and TTFC in the first skewed sticky run.
name = "B-80pct-sticky-1"
rows = phases[name]
origin = min(r["planned_at"] for r in rows)
fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
for pod in sorted({r["pod"] for r in rows if r["pod"]}):
    completed = [r for r in rows if r["pod"]==pod and r["outcome"]=="completed"]
    axes[0].scatter([r["sent_at"]-origin for r in completed], [r["ttfc_s"] for r in completed],
                    s=9, alpha=.65, label=pod.rsplit("-",1)[-1])
    axes[1].scatter([r["sent_at"]-origin for r in completed], [r["queue_seconds"] for r in completed],
                    s=9, alpha=.65, label=pod.rsplit("-",1)[-1])
axes[0].set(title="ClientIP affinity, 80% of arrivals through client 0, 6 requests/s", ylabel="TTFC, s")
axes[1].set(ylabel="Queue time, s")
hot_metrics = json.loads((RESULTS / f"{name}-metrics.json").read_text())
for metric, linestyle in (("waiting","-"),("running","--")):
    for series in hot_metrics[metric]:
        pod = series["metric"]["pod"]
        suffix = pod.rsplit("-",1)[-1]
        color = "#1f77b4" if pod==sorted({r["pod"] for r in rows if r["pod"]})[0] else "#ff7f0e"
        axes[2].step([float(v[0])-origin for v in series["values"]],
                     [float(v[1]) for v in series["values"]], where="post",
                     linestyle=linestyle,color=color,label=f"{suffix} {metric}")
axes[2].set(xlabel="Time from planned start, s",ylabel="Requests, count")
for ax in axes:
    ax.legend(title="Pod suffix", fontsize=8)
save(fig, "02-hot-pod.png")

# B. Show every repeat, including errors, against the same offered traffic.
fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
labels = ["uniform / plain", "uniform / sticky", "80% / plain", "80% / sticky"]
for i, (traffic, policy) in enumerate((t,p) for t in ("uniform","80pct") for p in ("plain","sticky")):
    runs = [lookup[f"B-{traffic}-{policy}-{rep}"] for rep in (1,2,3)]
    for j, run in enumerate(runs):
        axes[0].scatter(i+(j-1)*.09, run["ttfc_p95"], s=45, color=["#356c9b","#d48632","#16855b"][j], label=f"repeat {j+1}" if i==0 else None)
        axes[1].scatter(i+(j-1)*.09, 100*run["error_fraction"], s=45, color=["#356c9b","#d48632","#16855b"][j])
for ax in axes:
    ax.set_xticks(range(4), labels, rotation=16)
axes[0].set(ylabel="Successful TTFC p95, s", title="120 s per repeat, cold prefix cache, 6 requests/s")
axes[1].set(ylabel="Failed attempts, %", title="All planned attempts are counted")
axes[0].legend()
save(fig, "03-policy-comparison.png")

# C. CPU demand, observed replicas and HPA desired state.
states = [r for r in timeline if r["phase"]=="C-hpa" and "state" in r]
origin = min(r["ts"] for r in states)
fig, axes = plt.subplots(3,1,figsize=(11,9),sharex=True,constrained_layout=True)
xs = [r["ts"]-origin for r in states]
for key, label in (("currentReplicas","HPA current"),("desiredReplicas","HPA desired")):
    axes[0].step(xs,[next((s["status"].get(key,0) for s in r["state"]
                          if s["kind"]=="HorizontalPodAutoscaler"),0) for r in states],
                 where="post",label=label)
axes[0].step(xs,[sum(s["kind"]=="Pod" and (s["ready"] or {}).get("status")=="True"
                    and not s["deletion"] for s in r["state"]) for r in states],where="post",label="Ready")
cpu = json.loads((RESULTS / "C-metrics.json").read_text())["cpu"]
for series in cpu:
    axes[1].plot([float(v[0])-origin for v in series["values"]],
                 [float(v[1]) for v in series["values"]],label=series["metric"].get("pod","").rsplit("-",1)[-1])
for phase, color in (("C-cpu-sticky","#b14a3e"),("C-cpu-plain","#356c9b")):
    rows = phases[phase]
    offered = Counter(int(r["planned_at"]-origin) for r in rows)
    positions = [min(offered)-1,*sorted(offered),max(offered)+1,int(xs[-1])+1]
    axes[2].step(positions,[offered[t] for t in positions],where="post",label=phase,color=color)
axes[0].set(title="CPU HPA, min=2 / max=3, scale-down stabilization=60 s",ylabel="Replicas",yticks=[2,3])
axes[1].set(ylabel="CPU, cores")
axes[2].set(xlabel="Time from HPA observation start, s",ylabel="Planned arrivals, requests/s")
for ax in axes:
    ax.legend(fontsize=8)
save(fig,"04-hpa.png")

comparisons = []
idle_samples = []
for name in bphases:
    data = json.loads((RESULTS / f"{name}-metrics.json").read_text())
    waiting = {entry["metric"]["pod"]: {float(t):float(v) for t,v in entry["values"]} for entry in data["waiting"]}
    running = {entry["metric"]["pod"]: {float(t):float(v) for t,v in entry["values"]} for entry in data["running"]}
    timestamps = set(t for entries in waiting.values() for t in entries)
    idle_with_queue = sum(any(waiting[pod].get(t,0)>0 and
                             any(running[other].get(t,1)<1 for other in running if other!=pod)
                             for pod in waiting) for t in timestamps)
    idle_samples.append([name,idle_with_queue,len(timestamps)])
    count,p95 = bucket_quantile(data["buckets_before"],data["buckets_after"],.95)
    server_p95 = data["p95"][0]["value"][1] if data["p95"] else None
    comparisons.append([name,lookup[name]["completed"],fmt(count,0),fmt(lookup[name]["ttfc_p95"]),
                         fmt(p95),fmt(float(server_p95)) if server_p95 is not None else "—"])
timeline_failures = sum("error" in r for r in timeline)
alerts = defaultdict(set)
for sample in timeline:
    for alert in sample.get("alerts",[]):
        alerts[alert["name"]].add(alert["state"])
hpa_states = [s for r in states for s in r["state"] if s["kind"]=="HorizontalPodAutoscaler"]
max_replicas = max(s["status"].get("currentReplicas",0) for s in hpa_states)
new_pods = {}
for sample in states:
    for pod in sample["state"]:
        if pod["kind"]=="Pod":
            new_pods.setdefault(pod["name"],dict(created=pod["created"],ready=None))
            if (pod["ready"] or {}).get("status")=="True" and new_pods[pod["name"]]["ready"] is None:
                new_pods[pod["name"]]["ready"] = pod["ready"].get("lastTransitionTime")
load_end = max(r["ended_at"] for name in ("C-cpu-sticky","C-cpu-plain") for r in phases[name])
down = next((r["ts"] for r in states if r["ts"]>load_end and
             any(s["kind"]=="HorizontalPodAutoscaler" and s["status"].get("currentReplicas")==2
                 and s["status"].get("desiredReplicas")==2 for s in r["state"])),None)

forced = json.loads((RESULTS / "forced-event.json").read_text())
forced_ts, forced_pod = forced["ts"], forced["pod"]
forced_samples = [r for r in timeline if r["phase"]=="A-force" and r["ts"]>=forced_ts and "state" in r]
not_ready = next((r["ts"] for r in forced_samples
                  if any(s["kind"]=="Pod" and s["name"]==forced_pod and (s["ready"] or {}).get("status")!="True"
                         for s in r["state"])),None)
excluded = next((r["ts"] for r in forced_samples
                 if not any(e.get("pod")==forced_pod and e.get("ready",False)
                            for s in r["state"] if s["kind"]=="EndpointSlice" and s["service"]=="hw11-plain"
                            for e in s["endpoints"])),None)
same_pod_recovered = min((r["sent_at"] for r in requests
                          if r["phase"]=="A-force-recovery" and r["pod"]==forced_pod and r["outcome"]=="completed"),default=None)
force_partial = sum(r["phase"]=="A-force" and r["outcome"]=="aborted" and r["parts"]>0 for r in requests)

body = [
'<h1>ДЗ 11. Kubernetes под нагрузкой</h1>',
'<p class="lead">Наблюдаемый CPU-стенд для потокового API. Измерения выполнены в локальном kind, backend_mode=simulated.</p>',
'<h2>Среда и воспроизводимость</h2>',
'<p>Продолжен API ДЗ 8+9+10 — FastAPI, /health, /ready, /generate, фоновый прогрев и проверенная неизменяемая идентичность загруженного артефакта. '
'В ДЗ 11 добавлены NDJSON-поток, очередь, ограничение допуска, Prometheus, история в запросе и Kubernetes. '
'Синхронный ответ содержит answer и model_id. В опыте A используются два слота на Pod, в B и C — один. max_new_tokens в имитаторе ограничивает число частей. '
'Фактическая модель и GPU в опытах отсутствуют.</p>',
'<p>Kubernetes 1.32.2, kind 0.27.0, Helm 3.17.3. Docker Desktop выделяет 8 CPU и около 3,82 GiB RAM. '
'API запрашивает 100m CPU и 64 MiB RAM, лимит 750m и 192 MiB на Pod. Две реплики, HPA до трёх. '
'Prometheus хранит 8 часов метрик с лимитом TSDB 128 MB и emptyDir 256 MiB, Grafana emptyDir 128 MiB. '
'Теги образов, фактические digest и размеры сохранены в results. Веса представляют два небольших JSON-файла в immutable ConfigMap.</p>',
'<p><a href="results/disk-before.txt">Свободное место перед запуском</a> · '
'<a href="results/docker-before.txt">Начальное использование Docker</a> · '
'<a href="results/docker-capacity.txt">Лимиты Docker</a> · '
'<a href="results/environment.json">Версии, digest и фактический размер</a> · '
'<a href="README.md">Запуск и удаление</a></p>',
'<h2>Метод измерения</h2>',
'<p>Клиенты — четыре Pod с разными IP. Одна попытка соответствует одному request ID, повторов нет. '
'Расписание поступления сохранено до отправки. HTTP keep-alive выключен через max_keepalive_connections=0. '
'Длительности считаются time.monotonic, сервер использует time.perf_counter. '
'TTFC = время первой содержательной JSON-части − время отправки. '
'Inter-chunk latency = разность времён соседних частей, для одной части наблюдение отсутствует. '
'Полная длительность включает получение терминального done. HTTP 200 без всех частей и done считается обрывом. '
'Клиентский timeout = 32 с. Процентили успешных ответов рассчитаны линейной интерполяцией, рядом указано число ответов. '
'Размер входа одинаков для сравниваемых политик, 12 префиксов, 4 части по 80 мс. '
'В каждом сравнении 720 поступлений за 120 с, 6 запросов/с. Перед каждым прогоном SIGUSR1 очищает LRU-кеш, прогрев завершён.</p>',
'<p>Состояние Ready, EndpointSlice и HPA опрашивается раз в 1 с в A и 2 с в C. '
f'Ошибок наблюдателя — {timeline_failures}. Полный журнал находится в results/timeline.jsonl. '
'Время между отправками и планом отражено как send_lag_s. Backend сообщает Pod, SHA-256, модель, request ID и наблюдаемый peer IP в заголовках и частях. '
'История и session ID остаются в запросах и журнале. Высококардинальные идентификаторы в метрики не включены.</p>',
'<h2>Опыт A — прогрев, выпуск и откат</h2>',
'<p>Гипотеза — HTTP-процесс может быть доступен раньше готовности backend. Исходная проверка probeMode=http-only возвращает готовность сразу. '
'Сценарий отправляет одинаковый поток во время замены реплик. Сервер записывает received с warm=false и возвращает warming/503. '
'Исправленная startup/readiness проверяет завершение 20-секундного прогрева и успешную проверку артефакта. '
'Liveness /health проверяет работу процесса, поэтому прогрев и некорректная версия не вызывают немедленный liveness restart. '
'Startup budget равен прогреву плюс 42 с. Старая готовая реплика остаётся в Service до готовности новой. <a href="https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/">Kubernetes о probes</a>.</p>',
table(["Проверка","Результат"],[[c["check"],"PASS" if c["passed"] else "FAIL"] for c in checks]),
'<p>Штатный rolling update A→B использует maxUnavailable=0, maxSurge=1. Длительные ответы запускаются до обновления и во время прогрева B. '
'preStop 5 с оставляет время на исключение endpoint, terminationGracePeriodSeconds=45, Uvicorn ждёт активные ответы до 35 с. '
'В дополнительной проверке длинный ответ содержит 32 части с интервалом 740 мс и длится около 24 с с учётом очереди. Начатый поток обслуживается исходным процессом до завершения. '
'Для дефектной B намеренно задан неверный SHA-256, новая реплика остаётся неготовой, A продолжает отвечать. '
'Helm rollback возвращает ревизию A. Подтверждение использует фактический model_id и хеш в ответах. '
'При удалении Pod новый процесс заново проверяет тот же файл A. Продолжение диалога и различимые контрольные строки двух сессий проверяются через Service.</p>',
'<p>Отдельный опыт SIGKILL завершает именно PID контейнера API внутри kind. Сохранены контейнер, Pod и время воздействия. '
'Сначала отправляется 11 одновременных запросов, поэтому журнал содержит как 429 на пределе допуска, так и оборванные ответы со статусом 200. '
'После отказа проверяются Ready=false, endpoint exclusion, restartCount и восстановление того же Pod после прогрева. '
'Нули ошибок штатного выпуска относятся к сохранённому объёму и тайм-аутам этой нагрузки.</p>',
table(["Поток при завершении Pod","Terminating после начала rollout, с","Ответ завершён после начала rollout, с"],
      [[s["request_id"],fmt(None if s["terminating_observed_at"] is None else s["terminating_observed_at"]-drain_proof["update_started"]),
        fmt(s["ended_at"]-drain_proof["update_started"])] for s in drain_proof["streams"]]),
table(["После SIGKILL","Время от воздействия, с"],
      [["Ready=false",fmt(None if not_ready is None else not_ready-forced_ts)],
       ["Исключение из Ready endpoints",fmt(None if excluded is None else excluded-forced_ts)],
       ["Первый успешный запрос к перезапущенному Pod",fmt(None if same_pod_recovered is None else same_pod_recovered-forced_ts)]]),
f'<p>Оборванных ответов с уже полученными частями — {force_partial}. Остальные оборванные попытки ожидали первую часть. Разрешение наблюдения готовности — около 1 с.</p>',
figure("01-rollout.png","Готовность и фактические endpoints во время дефектного прогрева, исправления и выпуска."),
'<h2>Опыт B — ClientIP и горячая реплика</h2>',
'<p>Гипотеза — закрепление клиента сохраняет локальность префикса, но концентрирует очередь. '
'Сравниваются два Service над одним Deployment. plain использует sessionAffinity=None, sticky — ClientIP с timeoutSeconds=10800. '
'Сетевые параметры kube-proxy, IP клиентов и Service сохранены в results/network.json. '
'Клиенты находятся внутри кластера, identity peer в трассе позволяет проверить адрес, который видит API. '
'Первый сценарий равномерно распределяет запросы по четырём источникам, второй отдаёт ровно 80% клиенту 0, остальные делятся поровну. '
'В равномерном сценарии ClientIP четыре IP закрепились по два за каждым Pod. В этом контрольном случае распределение 360/360 не создало горячую реплику. При 80% через клиента 0 та же таблица привязки концентрирует большую часть поступлений на его Pod. '
'На каждом Pod один running slot и максимум 5 ожидающих, LRU вместимостью 32 префикса. '
'Имитация hit стоит 35 мс, miss 180 мс, последующие части стоят по 80 мс. '
'Кеш пуст в начале каждой 120-секундной фазы. Порядок plain/sticky меняется между повторами.</p>',
figure("02-hot-pod.png","Задержка первой части и ожидание вычислительного слота для каждой реплики, sticky и 80% через одного клиента."),
figure("03-policy-comparison.png","Каждая точка — отдельный повтор. p95 успешных ответов рассматривается вместе с долей отказов."),
'<p>Фактические отправки местами отстают от плана. Максимум в третьем горячем sticky-прогоне составляет 4,86 с, в третьем горячем plain — 3,04 с. '
'В третьем равномерном sticky задержка достигает 1,57 с, восемь отказов находятся в первых 5,67 с фазы при сохранении распределения 360/360. '
'Начальные отставания создают пакеты догоняющих запросов. Первичные результаты сохранены целиком, p95 и максимум send_lag показаны отдельно. '
'Желаемое расписание и входы одинаковы, фактический тайминг отличается. Это ограничивает точность причинного сравнения отдельных отказов.</p>',

table(["Прогон","Запланировано","Успешно","Отказ","Обрыв","Timeout","p50 TTFC, с","p95 TTFC, с","p95 / max lag, с"],
      [[s["phase"],s["planned"],s["completed"],s["rejected"],s["aborted"],s["timed_out"],
        fmt(s["ttfc_p50"]),fmt(s["ttfc_p95"]),f'{fmt(s["send_lag_p95"])} / {fmt(s["send_lag_max"])}'] for s in summary if s["phase"] in bphases]),
table(["Прогон","Сэмплы с очередью и свободным слотом на другом Pod","Всего сэмплов"],idle_samples),
table(["Прогон / Pod","Получено","Успешно","p95 очередь, с","p95 TTFC, с","Hit, %"],
      [[s["phase"]+" / "+s["pod"].rsplit("-",1)[-1],s["arrived"],s["completed"],
        fmt(s["queue_p95"]),fmt(s["ttfc_p95"]),fmt(s["cache_hit_fraction"]*100,1)]
       for s in per_pod if s["phase"] in bphases]),
'<p>Исправление — sessionAffinity=None и новое TCP-соединение на каждую попытку. '
'В условиях этого имитатора повторный miss добавляет 145 мс, а накопившаяся очередь на горячем Pod может стоить больше. '
'Трасса plain с 80% поступлений через одного клиента также сохраняет контрольный результат, при котором перекос источников не превращается в закрепление за единственной репликой. '
'Каждый запрос маршрутизируется отдельно. TCP Service выбирает endpoint на соединение, поэтому сохранение длинных keep-alive соединений потребовало бы отдельного опыта. '
'На реальном inference engine следует измерить стоимость prefill, KV cache hit, eviction, длину промпта, конкурентный decode, batching и стоимость маршрутизатора. '
'Результаты относятся к заданным задержкам имитатора.</p>',
'<h2>Histogram и клиентский журнал</h2>',
'<p>На сервере Histogram учитывает TTFC, total и queue только завершённых ответов. Счётчики исходов хранят ok, overloaded, warming и aborted. '
'Журнал клиента остаётся источником правды для SIGKILL, поскольку погибший процесс не успевает увеличить aborted. '
'Для перцентилей используется sum по bucket-счётчикам с сохранением le. На дашборде применяется разрез pod. '
'В таблице приведены клиентский p95, квантиль из разности фактических накопленных bucket перед/после фазы и PromQL за соответствующее окно. '
'Число bucket-наблюдений должно совпадать с числом завершённых попыток. '
'Различие времени объясняется сетевым обменом, буферизацией и интерполяцией внутри границ bucket. '
'increase дополнительно экстраполирует края окна, поэтому точная сверка количества выполнена разностью cumulative counters. <a href="https://prometheus.io/docs/practices/histograms/">Prometheus о Histogram</a>.</p>',
'<pre>histogram_quantile(0.95,\n  sum by (le) (increase(hw11_ttfc_seconds_bucket[124s])))\n\n'
'histogram_quantile(0.95,\n  sum by (pod, le) (rate(hw11_ttfc_seconds_bucket[30s])))</pre>',
table(["Прогон","Клиент N","Histogram N","Клиент p95, с","Bucket p95, с","PromQL p95, с"],comparisons),
'<h2>Опыт C — HPA и алерты</h2>',
'<p>Гипотеза — вычислительная работа при настроенных CPU requests вызывает HPA 2→3, а снижение нагрузки возвращает две реплики. '
'В имитаторе cpu_ms=300 выполняет SHA-256 до расходования 300 мс thread CPU, работа вынесена из event loop. '
'Это создаёт измеряемое потребление CPU. Resource metrics API проверяется до опыта. '
'HPA autoscaling/v2 имеет target CPU=65% от request 100m, minReplicas=2, maxReplicas=3, scaleDown stabilizationWindowSeconds=60. '
'Для двух готовых реплик значения 100% и 0% дают среднее 50%, ниже target. Значения 150% и 0% дают 75%, выше target и стандартной 10%-й tolerance, что допускает рост до трёх реплик. '
'Очередь сама по себе не является CPU-метрикой. В опыте B значительная часть стоимости реализована ожиданием, в C добавлена вычислительная работа. '
'Первый участок повторяет горячего клиента со sticky — 6 поступлений/с в течение 80 с. Второй использует plain — 8 поступлений/с в течение 75 с. '
'Это ступени вычислительной нагрузки для HPA. Контролируемое сравнение политик выполнено в опыте B. Средняя CPU способна превысить target даже при одной горячей реплике. '
'ClientIP может продолжить отправлять большинство запросов старой реплике после появления третьей, что проверяется по ответам. <a href="https://kubernetes.io/docs/concepts/workloads/autoscaling/horizontal-pod-autoscale/">Kubernetes о HPA</a>.</p>',
f'<p>Максимум currentReplicas в журнале — {max_replicas}. '
f'Возврат current=desired=2 после завершения CPU-нагрузки занял {fmt(None if down is None else down-load_end,1)} с. '
'Эта длительность включает окно CPU-метрики, цикл HPA и стабилизацию. Сохранены исходные условия и итоговый HPA YAML.</p>',
table(["Pod","Создан UTC","Первый переход Ready UTC"],[[name,v["created"],v["ready"]] for name,v in new_pods.items()]),
figure("04-hpa.png","Фактические current/desired/Ready, CPU по Pod и открытая нагрузка в одном масштабе времени."),
table(["CPU-фаза / Pod","Получено","Успешно","p95 очередь, с","p95 TTFC, с"],
      [[s["phase"]+" / "+s["pod"].rsplit("-",1)[-1],s["arrived"],s["completed"],fmt(s["queue_p95"]),fmt(s["ttfc_p95"])]
       for s in per_pod if s["phase"].startswith("C-cpu")]),
'<p>HighTTFC проверяет p95 &gt; 0,65 с при минимум 15 завершённых ответах за 30 с. '
'HighRejectionRatio проверяет долю отказов &gt; 10% при минимум 15 исходах за 30 с. '
'Оба условия удерживаются 16 с, evaluation_interval=2 с. '
'Перед вычислительной нагрузкой выполняется 40-секундная базовая фаза. CPU-нагрузка повышает очередь и отказы, после её снятия условие уходит в inactive. '
'Состояния pending/firing сохраняются в timeline и в query_range ALERTS.</p>',
table(["Алерт","Наблюдавшиеся состояния"],[[name,", ".join(sorted(v))] for name,v in sorted(alerts.items())]),
'<h2>Все попытки</h2>',
table(["Фаза","План / отправлено","Успешно","Отказ / обрыв / timeout","p50 / p95 TTFC, с","p95 полного ответа, с"],
      [[s["phase"],f'{s["planned"]} / {s["sent"]}',s["completed"],
        f'{s["rejected"]} / {s["aborted"]} / {s["timed_out"]}',
        f'{fmt(s["ttfc_p50"])} / {fmt(s["ttfc_p95"])}',fmt(s["total_p95"])] for s in summary]),
'<h2>Ревью предложения закрепить диалог за Pod</h2>',
'<p>При очереди на A в 3 с и стоимости пересчёта на свободной B в 0,8 с новый запрос следует отправить B. '
'При одинаковой оставшейся работе ожидаемое преимущество по первой части составляет 2,2 с. '
'В расчёт нужно добавить измеренные p95/p99 задержки очереди, prefill и decode, размер входа, доступную память KV-кеша, '
'точность сведений о кеше, сетевые задержки и срок актуальности загрузки. '
'Допуск к запросу требует свободного слота либо ограниченной очереди с оценкой завершения до deadline, '
'при исчерпании лимита возвращается явный 429. Повтор клиента получает отдельный attempt ID и учитывается отдельно.</p>',
'<p>Историю диалога следует передавать вместе с запросом или восстанавливать из внешнего хранилища с разделением пользователей. '
'После потери A новая реплика восстанавливает историю и пересчитывает префикс. Прикладная история остаётся источником контекста, '
'а локальный KV-кеш ускоряет prefill. Для начатого потокового ответа сохраняется исходный backend до done в пределах grace period. '
'При внезапном падении клиент фиксирует обрыв, а продолжение потребует явного протокола дедупликации и смещения. '
'Автоматический повтор после уже выданных частей может создать дубли, поэтому он должен быть отдельным управляемым действием.</p>',
'<p>Связанный промышленный пример — NVIDIA Dynamo. Его KV-aware router учитывает перекрытие префикса и текущую работу prefill/decode. '
'Этот подход соответствует выбору по полной ожидаемой стоимости обработки с учётом очереди. '
'На учебном стенде проверено более простое исправление — отключение ClientIP. '
'Лекция 11 не приложена, поэтому присутствие этого кейса именно в лекции не подтверждено. '
'<a href="https://docs.dynamo.nvidia.com/dynamo/v1.0.0/user-guides/kv-cache-aware-routing">Первичный источник NVIDIA Dynamo Router Guide</a>.</p>',
'<h2>Ограничения и артефакты</h2>',
'<p>Стенд состоит из одного kind node и использует CPU-имитатор. Он проверяет жизненный цикл Kubernetes, очереди, привязку, метрики и клиентский учёт. '
'Производительность настоящей модели, токенизация, GPU, multi-node сеть и перенос KV-кеша требуют отдельных измерений. '
'emptyDir метрики исчезают при замене Prometheus, поэтому измеренные серии выгружены в results. '
'Материалы для видео находятся в video.md.</p>',
'<p><a href="results/requests.jsonl">Сырые запросы</a> · <a href="results/schedule.jsonl">Расписание</a> · '
'<a href="results/timeline.jsonl">Kubernetes и alerts</a> · <a href="results/checks.json">Kubernetes проверки</a> · '
'<a href="chart/monitoring/dashboard.json">Grafana dashboard</a></p>',
]
style = """body{margin:0;background:#f3f5f8;color:#172233;font:16px/1.65 system-ui,sans-serif}
main{max-width:1200px;margin:auto;padding:40px 32px 80px;background:white}h1{font-size:34px;line-height:1.25}
h2{margin-top:40px;font-size:25px}p{max-width:1050px}.lead{font-size:20px;color:#45546a}
table{border-collapse:collapse;width:100%;font-size:12px;margin:24px 0}th,td{padding:9px 7px;text-align:left;border-bottom:1px solid #d5dce5;overflow-wrap:anywhere}
th{background:#e9eff5}figure{margin:28px 0}img{width:100%;height:auto}figcaption{font-size:14px;color:#46556a}
pre{padding:20px;background:#edf2f7;overflow:auto}a{color:#195a9d}
@media print{body{background:white}main{padding:0}h2{break-after:avoid}figure,tr{break-inside:avoid}table{font-size:10px}}"""
(ROOT / "report.html").write_text('<!doctype html><html lang="ru"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1"><title>ДЗ 11 · Отчёт</title>'
    f'<style>{style}</style></head><body><main>' + "\n".join(body) + '</main></body></html>')
print(f"Report built from {len(requests)} attempts, {len(bphases)} B runs and {len(timeline)} state observations")
