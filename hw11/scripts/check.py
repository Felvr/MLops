"""Validate the measured submission and all links without creating experiment data."""
from collections import Counter
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def rows(name):
    return [json.loads(line) for line in (RESULTS / name).read_text().splitlines()]


requests = rows("requests.jsonl")
plan = rows("schedule.jsonl")
assert len({r["id"] for r in requests}) == len(requests), "duplicate request IDs"
assert {r["id"] for r in plan} == {r["id"] for r in requests}, "attempt accounting differs"
checks = json.loads((RESULTS / "checks.json").read_text())
assert len(checks) >= 8 and all(c["passed"] for c in checks), "Kubernetes check failed"
counts = Counter(r["phase"] for r in requests)
for traffic in ("uniform", "80pct"):
    for policy in ("plain", "sticky"):
        for repeat in (1, 2, 3):
            name = f"B-{traffic}-{policy}-{repeat}"
            assert counts[name] == 720, (name, counts[name])
            data = json.loads((RESULTS / f"{name}-metrics.json").read_text())
            assert data["duration"] >= 120
            previous = {(r["metric"]["pod"],r["metric"]["le"]):float(r["value"][1])
                        for r in data["buckets_before"]}
            total = sum(float(r["value"][1])-previous.get((r["metric"]["pod"],"+Inf"),0)
                        for r in data["buckets_after"] if r["metric"]["le"]=="+Inf")
            expected = sum(r["outcome"]=="completed" for r in requests if r["phase"]==name)
            assert abs(total-expected) < .01, (name,total,expected)
for row in requests:
    assert row["ended_at"] >= row["sent_at"]
    assert row["parts"] == len(row["chunk_monotonic"])
    if row["outcome"]=="completed":
        assert row["complete"] and row["status"]==200
        assert row["parts"]==row["payload"].get("max_new_tokens",4)
        assert row["backend_mode"]=="simulated"
        assert len(row["runtime_sha256"])==64
        assert row["ttfc_s"]>0
    if row["first_monotonic"] is None:
        assert row["ttfc_s"] is None
timeline = rows("timeline.jsonl")
assert sum("error" in r for r in timeline)==0, "observer failed"
for alert in ("HighTTFC","HighRejectionRatio"):
    states={a["state"] for r in timeline for a in r.get("alerts",[]) if a["name"]==alert}
    assert {"inactive","pending","firing"} <= states, (alert,states)
c = [r for r in timeline if r["phase"]=="C-hpa"]
replicas = [s["status"].get("currentReplicas") for r in c for s in r.get("state",[])
            if s["kind"]=="HorizontalPodAutoscaler"]
assert replicas[0]==2 and 3 in replicas and replicas[-1]==2, replicas
assert all(a["state"]=="inactive" for a in c[-1]["alerts"]), "alerts did not recover"
metrics=json.loads((RESULTS / "C-metrics.json").read_text())
for name in ("cpu","memory","ready","alerts"):
    assert metrics[name], f"missing {name} time series"
drain=json.loads((RESULTS / "drain-proof.json").read_text())
assert any(stream["crossed"] for stream in drain["streams"]), "no old stream crossed termination"
report=(ROOT / "report.html").read_text()
for target in re.findall(r'(?:src|href)="([^"]+)"',report):
    if not target.startswith(("http:","https:","#")):
        assert (ROOT / target).is_file(), target
assert len(list((RESULTS / "figures").glob("*.png")))==4
assert len(json.loads((RESULTS / "grafana-dashboard.json").read_text())["panels"])==8
assert (ROOT / "uv.lock").is_file()
print(f"PASS — {len(requests)} attempts, {len(checks)} Kubernetes checks, 12 comparisons, 4 figures, HPA and alerts")
