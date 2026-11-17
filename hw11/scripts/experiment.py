"""Control the dedicated lab and record Kubernetes state, schedules and raw client attempts."""
import argparse
import asyncio
from collections import Counter
from contextlib import asynccontextmanager, contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import httpx
from logs import collect

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = "kind-mlops-hw11"
os.environ.setdefault("KUBECONFIG", str(ROOT / ".local/kubeconfig"))
KUBE = ["kubectl", "--context", CONTEXT, "-n", "hw11"]
HELM = ["helm", "--kube-context", CONTEXT, "-n", "hw11"]
RESULTS = ROOT / "results"
PROM = "http://127.0.0.1:19090"
CLIENTS = 4


def command(*args, **kwargs):
    return subprocess.check_output(args, text=True, cwd=ROOT, timeout=150, **kwargs)


async def capture(*args, stdin=None):
    process = await asyncio.create_subprocess_exec(
        *args, cwd=ROOT, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        output, errors = await process.communicate(None if stdin is None else stdin.encode())
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        raise
    if process.returncode:
        raise RuntimeError(f"{' '.join(args)}: {errors.decode()}")
    return output.decode()


def dump(name, data):
    (RESULTS / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def append(path, rows):
    with path.open("a") as log:
        for row in rows:
            log.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def install(*options):
    values = command(*HELM, "upgrade", "hw11", "chart", "--reset-values", *options)
    print(values.splitlines()[0], *options, flush=True)


def rollout(timeout="120s"):
    command(*KUBE, "rollout", "status", "deployment/hw11-api", "--timeout", timeout)


def pods():
    return json.loads(command(*KUBE, "get", "pods", "-l", "app=hw11-api", "-o", "json"))["items"]


def server_logs(label):
    for pod in pods():
        name = pod["metadata"]["name"]
        result = subprocess.run([*KUBE, "logs", name, "--since=8m"], text=True, capture_output=True)
        (RESULTS / f"{label}-{name}.log").write_text(result.stdout + result.stderr)


def row(client=0, prefix="shared", **payload):
    return {"client": client, "payload": {"prompt": f"{prefix}|test", "stream": True,
                                         "session_id": f"client-{client}", **payload}}


def schedule(label, length, rate, traffic, **payload):
    skewed = (0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3)
    uniform = tuple(range(CLIENTS))
    sequence = skewed if traffic == "80pct" else uniform
    return [
        dict(row(sequence[index % len(sequence)], f"prefix-{index % 12:02}", **payload),
             offset_s=index / rate)
        for index in range(round(length * rate))
    ]


def record_plan(label, start, plan):
    rows = [dict(item, id=f"{label}-{index:04}", phase=label,
                 planned_at=start + item.get("offset_s", 0))
            for index, item in enumerate(plan)]
    append(RESULTS / "schedule.jsonl", rows)
    return rows


async def drive(label, plan, target="plain", start=None):
    start = time.time() + 3 if start is None else start
    rows = record_plan(label, start, plan)
    target_url = f"http://hw11-{target}:8080" if "://" not in target else target

    async def client_run(index):
        selected = [item for item in rows if item["client"] == index]
        if not selected:
            return []
        out = await capture(*KUBE, "exec", "-i", f"hw11-client-{index}", "--",
                            "python", "scripts/client.py", target_url, stdin=json.dumps(selected))
        return [json.loads(line) for line in out.splitlines()]

    outcomes = await asyncio.gather(*(client_run(index) for index in range(CLIENTS)))
    combined = sorted((item for group in outcomes for item in group), key=lambda item: item["id"])
    if {item["id"] for item in combined} != {item["id"] for item in rows}:
        raise RuntimeError(f"Missing or unexpected request results in {label}")
    append(RESULTS / "requests.jsonl", combined)
    print(label, dict(Counter(item["outcome"] for item in combined)), flush=True)
    return combined


async def query(client, expr, instant=None, interval=None):
    params = {"query": expr}
    path = "query" if interval is None else "query_range"
    if instant is not None:
        params["time"] = instant
    if interval:
        params.update(start=interval[0], end=interval[1], step=interval[2])
    response = await client.get(f"{PROM}/api/v1/{path}", params=params)
    response.raise_for_status()
    return response.json()["data"]["result"]


@contextmanager
def port_forward():
    with subprocess.Popen([*KUBE, "port-forward", "svc/hw11-prometheus", "19090:9090"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as process:
        try:
            for _ in range(35):
                try:
                    if httpx.get(PROM + "/-/ready", timeout=.5, trust_env=False).is_success:
                        break
                except httpx.HTTPError:
                    pass
                if process.poll() is not None:
                    raise RuntimeError("Prometheus port-forward ended")
                time.sleep(.4)
            else:
                raise RuntimeError("Prometheus did not become ready")
            yield
        finally:
            process.terminate()


def compress_snapshot(item):
    kind = item["kind"]
    name = item["metadata"]["name"]
    status = item.get("status", {})
    if kind == "Pod":
        if item["metadata"].get("labels", {}).get("app") != "hw11-api":
            return None
        container = (status.get("containerStatuses") or [{}])[0]
        return dict(kind=kind, name=name, ip=status.get("podIP"),
                    created=item["metadata"]["creationTimestamp"],
                    ready=next((c for c in status.get("conditions", []) if c["type"] == "Ready"), None),
                    deletion=item["metadata"].get("deletionTimestamp"), restart=container.get("restartCount"),
                    state=container.get("state"), last_state=container.get("lastState"))
    if kind == "EndpointSlice":
        service = item["metadata"].get("labels", {}).get("kubernetes.io/service-name")
        if service not in ("hw11-plain", "hw11-sticky"):
            return None
        return dict(kind=kind, name=name, service=service,
                    endpoints=[dict(pod=e.get("targetRef", {}).get("name"), addresses=e["addresses"],
                                    **e["conditions"]) for e in item["endpoints"]])
    if name == "hw11-api":
        return dict(kind=kind, name=name, status=status)
    return None


@asynccontextmanager
async def watch(label, poll=1):
    async def loop():
        async with httpx.AsyncClient(trust_env=False, timeout=4) as client:
            while True:
                start = time.time()
                try:
                    items = json.loads(await capture(*KUBE, "get", "pods,endpointslices,hpa,deployments",
                                                     "-o", "json"))["items"]
                    state = [entry for item in items if (entry := compress_snapshot(item))]
                    alerts = (await client.get(PROM + "/api/v1/rules")).json()["data"]["groups"]
                    active = [
                        dict(name=rule["name"], state=rule["state"])
                        for group in alerts for rule in group["rules"] if rule["type"] == "alerting"
                    ]
                    append(RESULTS / "timeline.jsonl",
                           [dict(phase=label, ts=start, state=state, alerts=active)])
                except (RuntimeError, ValueError, KeyError, httpx.HTTPError) as exc:
                    append(RESULTS / "timeline.jsonl", [dict(phase=label, ts=start, error=str(exc))])
                await asyncio.sleep(max(0, poll - (time.time() - start)))

    task = asyncio.create_task(loop())
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def balance(reps):
    await asyncio.to_thread(install)
    await asyncio.to_thread(rollout)
    dump("network.json", {
        "proxy": json.loads(command(*KUBE, "-n", "kube-system", "get", "cm", "kube-proxy", "-o", "json"))["data"],
        "clients": [
            {"name": p["metadata"]["name"], "ip": p["status"]["podIP"]}
            for p in json.loads(command(*KUBE, "get", "pods", "-l", "app=hw11-client", "-o", "json"))["items"]
        ],
        "services": [
            {"name": item["metadata"]["name"], "spec": item["spec"]}
            for item in json.loads(command(*KUBE, "get", "svc", "hw11-plain", "hw11-sticky",
                                           "-o", "json"))["items"]
        ],
    })
    (RESULTS / "B-values.yaml").write_text(command(*HELM, "get", "values", "hw11", "--all"))
    (RESULTS / "B-deployment.yaml").write_text(command(*KUBE, "get", "deployment", "hw11-api", "-o", "yaml"))
    duration, rate = 120, 6
    for rep in range(1, reps + 1):
        policies = ("plain", "sticky") if rep % 2 else ("sticky", "plain")
        for traffic in ("uniform", "80pct"):
            for policy in policies:
                name = f"B-{traffic}-{policy}-{rep}"
                for pod in pods():
                    if not pod["metadata"].get("deletionTimestamp"):
                        command(*KUBE, "exec", pod["metadata"]["name"], "--", "python", "-c", "import os,signal; os.kill(1,signal.SIGUSR1)")
                async with httpx.AsyncClient(trust_env=False, timeout=8) as client:
                    before_hist = await query(client, "hw11_ttfc_seconds_bucket")
                start = time.time() + 4
                plan = schedule(name, duration, rate, traffic, max_new_tokens=4, chunk_ms=80)
                records = await drive(name, plan, policy, start)
                await asyncio.sleep(max(0, start + duration - time.time()) + 4)
                async with httpx.AsyncClient(trust_env=False, timeout=8) as client:
                    window = (start - 2, max(item["ended_at"] for item in records) + 2, 2)
                    length = int(window[1] - window[0] + .999)
                    instant = window[1]
                    series = {
                        "buckets_before": before_hist,
                        "buckets_after": await query(client, "hw11_ttfc_seconds_bucket"),
                        "waiting": await query(client, "hw11_waiting", interval=window),
                        "running": await query(client, "hw11_running", interval=window),
                        "p95": await query(client, f"histogram_quantile(0.95, sum by (le)"
                                          f"(increase(hw11_ttfc_seconds_bucket[{length}s])))", instant=instant),
                        "count": await query(client, f"sum(increase(hw11_ttfc_seconds_count[{length}s]))",
                                             instant=instant),
                    }
                dump(f"{name}-metrics.json", {"start": start, "duration": duration, "rate": rate,
                                             "window": window, **series})


async def deploy_during(label, options, plan, delay=0, wait_ready=True, target="plain"):
    async with watch(label):
        start = time.time() + 4
        load = asyncio.create_task(drive(label, plan, target=target, start=start))
        await asyncio.sleep(start + delay - time.time())
        await asyncio.to_thread(install, *options)
        if wait_ready:
            await asyncio.to_thread(rollout)
        results = await load
        server_logs(label)
    return results


def verify(checks, name, passed, **details):
    checks.append(dict(check=name, passed=bool(passed), **details))
    dump("checks.json", checks)
    print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)


async def release():
    checks = []
    broken = await deploy_during(
        "A-warm-broken", ("--set", "probeMode=http-only,slots=2"),
        schedule("broken", 35, 2, "uniform", max_new_tokens=1),
    )
    correct = await deploy_during(
        "A-warm-fixed", ("--set", "probeMode=correct,slots=2"),
        schedule("fixed", 35, 2, "uniform", max_new_tokens=1),
    )
    old_failures = [r["id"] for r in broken if r["rejection"] == "warming"]
    fixed_failures = [r["id"] for r in correct if r["rejection"] == "warming"]
    verify(checks, "cold request reached the defective deployment", old_failures, request_ids=old_failures)
    verify(checks, "corrected warmup never receives a cold request", not fixed_failures,
           request_ids=fixed_failures)
    long = dict(row(0, "long-before-rollout", max_new_tokens=27, chunk_ms=700), timeout=32)
    background = [dict(item, offset_s=item["offset_s"] + 5)
                  for item in schedule("rollout", 44, .5, "uniform", max_new_tokens=2, chunk_ms=100)]
    late = dict(row(2, "long-during-rollout", max_new_tokens=31, chunk_ms=700), offset_s=12)
    result = await deploy_during("A-rollout", ("--set", "modelVersion=B,slots=2"),
                                 [long, late, *background], delay=3)
    first = result[0]
    verify(checks, "old stream completes while healthy B rolls out",
           first["model_id"] == "A" and first["outcome"] == "completed"
           and all(r["outcome"] == "completed" for r in result)
           and any(r["model_id"] == "B" for r in result), stream=first["id"])

    await deploy_during("A-restore-A", ("--set", "slots=2"), schedule("restore", 32, 1, "uniform", max_new_tokens=1))
    revision = str(json.loads(command(*HELM, "history", "hw11", "-o", "json"))[-1]["revision"])
    async with watch("A-bad-B"):
        await asyncio.to_thread(install, "--set", "modelVersion=B,corruptChecksum=true,slots=2")
        rejected = await drive("A-bad-B", schedule("bad", 30, 1, "uniform", max_new_tokens=1))
        server_logs("A-bad-B")
        await capture(*HELM, "rollback", "hw11", revision, "--wait", "--timeout", "120s")
        restored = await drive("A-rollback", schedule("rollback", 5, 2, "uniform", max_new_tokens=1))
    expected = hashlib.sha256((ROOT / "chart/models/A.json").read_bytes()).hexdigest()
    verify(checks, "bad B excluded; actual A hash after Helm rollback",
           all(r["model_id"] == "A" and r["runtime_sha256"] == expected
               and r["outcome"] == "completed" for r in [*rejected, *restored]),
           rollback_to=revision, expected_sha256=expected)

    history = [{"role": "user", "content": "DIALOGUE-ALPHA-731"}]
    first = (await drive("A-dialogue-before", [row(0, history=history, session_id="alpha")], "sticky"))[0]
    victim = first["pod"]
    async with watch("A-replace"):
        await capture(*KUBE, "delete", "pod", victim, "--wait=false")
        await asyncio.to_thread(rollout)
        for _ in range(70):
            ready_pods = [p for p in await asyncio.to_thread(pods)
                          if not p["metadata"].get("deletionTimestamp")
                          and any(c["type"] == "Ready" and c["status"] == "True"
                                  for c in p["status"].get("conditions", []))]
            if len(ready_pods) == 2:
                break
            await asyncio.sleep(1)
        beta = [{"role": "user", "content": "DIALOGUE-BETA-984"}]
        after = await drive("A-dialogue-after",
                            [row(0, history=history, session_id="alpha"),
                             row(1, history=beta, session_id="beta")], "sticky")
    verify(checks, "replacement restores verified version and dialogue", len(ready_pods) == 2
           and all(p["metadata"]["name"] != victim for p in ready_pods)
           and after[0]["pod"] != victim and after[0]["runtime_sha256"] == expected
           and after[0]["text"] == first["text"] == history[0]["content"], replaced_pod=victim)
    verify(checks, "dialogues stay isolated", [r["text"] for r in after] ==
           [history[0]["content"], beta[0]["content"]])

    await forced_failure(checks)
    await drain(checks)
    if not all(item["passed"] for item in checks):
        raise RuntimeError("At least one Kubernetes check failed; see checks.json")


async def forced_failure(checks):
    binding = (await drive("A-force-bind", [row(0, max_new_tokens=1)], "sticky"))[0]
    victim = next(p for p in pods() if p["metadata"]["name"] == binding["pod"])
    name = victim["metadata"]["name"]
    container = victim["status"]["containerStatuses"][0]["containerID"].split("://")[1]
    node = victim["spec"]["nodeName"]
    runtime_data = json.loads(command("docker", "exec", node, "crictl", "inspect", container))
    pid = int(runtime_data["info"]["pid"])
    async with watch("A-force"):
        start = time.time() + 4
        burst = [row(0, "force", max_new_tokens=25, chunk_ms=680) for _ in range(11)]
        load = asyncio.create_task(drive("A-force", burst, "sticky", start))
        await asyncio.sleep(start + 3 - time.time())
        await capture("docker", "exec", node, "kill", "-KILL", str(pid))
        dump("forced-event.json", {"ts": time.time(), "pod": name, "container": container})
        rows = await load
        recovered = await drive("A-force-recovery", schedule("recovery", 36, 1, "uniform", max_new_tokens=1))
    verify(checks, "forced kill counts admissions, rejections and interrupted streams separately",
           any(r["outcome"] == "aborted" and r["parts"] > 0 for r in rows)
           and any(r["outcome"] == "rejected" and r["status"] == 429 for r in rows)
           and any(r["pod"] == name and r["outcome"] == "completed" for r in recovered))


async def drain(checks):
    """Keep both old Pods streaming across the first termination in a healthy release."""
    await asyncio.to_thread(install, "--set", "slots=2")
    await asyncio.to_thread(rollout)
    old = [p for p in pods() if not p["metadata"].get("deletionTimestamp")]
    if len(old) != 2:
        raise RuntimeError("Drain proof requires exactly two original Pods")
    async with watch("A-drain-proof"):
        start = time.time() + 4
        plan = [dict(row(0, "pre-update-stream", max_new_tokens=27, chunk_ms=700))]
        plan.extend(dict(item, offset_s=item["offset_s"]+5)
                    for item in schedule("drain", 45, .5, "uniform", max_new_tokens=2, chunk_ms=80))
        service_task = asyncio.create_task(drive("A-drain-service", plan, start=start))
        targeted = [
            asyncio.create_task(drive(f"A-drain-old-{index}",
                                     [dict(row(index+1, "drain-proof", max_new_tokens=32, chunk_ms=740),
                                           offset_s=15)],
                                     target=f"http://{pod['status']['podIP']}:8080", start=start))
            for index,pod in enumerate(old)
        ]
        await asyncio.sleep(start+3-time.time())
        update_started = time.time()
        await asyncio.to_thread(install, "--set", "modelVersion=B,slots=2")
        await asyncio.to_thread(rollout)
        result = await service_task
        streams = [group[0] for group in await asyncio.gather(*targeted)]
        await asyncio.sleep(7)
    observed = [
        json.loads(line) for line in (RESULTS / "timeline.jsonl").read_text().splitlines()
        if json.loads(line)["phase"] == "A-drain-proof"
    ]
    crossings = []
    for stream in streams:
        times = [sample["ts"] for sample in observed for state in sample.get("state", [])
                 if state["kind"]=="Pod" and state["name"]==stream["pod"] and state["deletion"]]
        began_termination = min(times) if times else None
        crossings.append(dict(request_id=stream["id"],pod=stream["pod"],
                              sent_at=stream["sent_at"],first_at=stream["first_at"],
                              ended_at=stream["ended_at"],terminating_observed_at=began_termination,
                              crossed=began_termination is not None
                              and stream["first_at"] < began_termination < stream["ended_at"]))
    dump("drain-proof.json", dict(update_started=update_started,streams=crossings))
    verify(checks, "active old stream completes after Pod enters Terminating",
           any(item["crossed"] for item in crossings)
           and all(r["outcome"]=="completed" for r in [*streams,*result])
           and any(r["model_id"]=="B" for r in result), proof=crossings)
    if not checks[-1]["passed"]:
        raise RuntimeError("Active-stream drain proof failed")


async def scaling():
    await asyncio.to_thread(install, "--set", "hpa.enabled=true")
    await asyncio.to_thread(rollout)
    for _ in range(20):
        try:
            metrics = json.loads(await capture(*KUBE, "get", "--raw",
                                              "/apis/metrics.k8s.io/v1beta1/namespaces/hw11/pods"))
            if any(item["metadata"]["name"].startswith("hw11-api") for item in metrics["items"]):
                dump("resource-metrics.json", metrics)
                break
        except RuntimeError:
            pass
        await asyncio.sleep(3)
    else:
        raise RuntimeError("Metrics API did not provide API Pod samples")
    (RESULTS / "hpa.yaml").write_text(command(*KUBE, "get", "hpa", "hw11-api", "-o", "yaml"))
    start = time.time()
    async with watch("C-hpa", poll=2):
        await drive("C-baseline", schedule("baseline", 40, 2, "uniform", max_new_tokens=2, chunk_ms=80))
        await drive("C-cpu-sticky", schedule("cpu-sticky", 80, 6, "80pct", max_new_tokens=2,
                                             chunk_ms=80, cpu_ms=300), "sticky")
        await drive("C-cpu-plain", schedule("cpu-plain", 75, 8, "80pct", max_new_tokens=2,
                                            chunk_ms=80, cpu_ms=300), "plain")
        for _ in range(90):
            status = json.loads(await capture(*KUBE, "get", "hpa", "hw11-api", "-o", "json"))["status"]
            if status.get("currentReplicas") == status.get("desiredReplicas") == 2:
                await asyncio.sleep(35)
                break
            await asyncio.sleep(3)
        else:
            raise RuntimeError("HPA did not return to two replicas")
    async with httpx.AsyncClient(trust_env=False, timeout=10) as client:
        series = {
            "cpu": 'sum by (pod)(rate(container_cpu_usage_seconds_total{container="api"}[20s]))',
            "memory": 'container_memory_working_set_bytes{container="api"}',
            "ready": 'kube_deployment_status_replicas_ready{deployment="hw11-api",namespace="hw11"}',
            "requests": 'sum(rate(hw11_requests_total[20s]))',
            "alerts": "ALERTS",
        }
        dump("C-metrics.json", {name: await query(client, expr, interval=(start, time.time(), 2))
                                for name, expr in series.items()})
    (RESULTS / "hpa-final.yaml").write_text(command(*KUBE, "get", "hpa", "hw11-api", "-o", "yaml"))


async def main(args):
    RESULTS.mkdir(exist_ok=True)
    logger = asyncio.create_task(collect())
    try:
        with port_forward():
            for phase in args.phases:
                if phase == "A":
                    await release()
                elif phase == "B":
                    await balance(args.reps)
                elif phase == "C":
                    await scaling()
                else:
                    checks = json.loads((RESULTS / "checks.json").read_text())
                    await drain(checks)
            await capture(sys.executable, "scripts/environment.py")
    finally:
        logger.cancel()
        await asyncio.gather(logger, return_exceptions=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phases", nargs="+", choices=("A", "B", "C", "drain"))
    parser.add_argument("--reps", type=int, default=3)
    asyncio.run(main(parser.parse_args()))
