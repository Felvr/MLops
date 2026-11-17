"""Record tool versions, immutable image identities and the dedicated lab's footprint."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
os.environ.setdefault("KUBECONFIG",str(ROOT / ".local/kubeconfig"))


def run(*args):
    return subprocess.check_output(args,text=True,cwd=ROOT).strip()


images = ["mlops-hw11:1.0.0","kindest/node:v1.32.2","prom/prometheus:v3.2.1",
          "grafana/grafana:11.6.0","registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.15.0",
          "registry.k8s.io/metrics-server/metrics-server:v0.7.2"]
identities=[]
for image in images:
    details=json.loads(run("docker","image","inspect",image))[0]
    identities.append(dict(tag=image,id=details["Id"],repo_digests=details.get("RepoDigests",[]),
                           bytes=details["Size"]))
container=json.loads(run("docker","inspect","--size","mlops-hw11-control-plane"))[0]
volume_bytes=int(run("docker","exec","mlops-hw11-control-plane","du","-sx","-B1","/var").split()[0])
lab=dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
         kubernetes=json.loads(run("kubectl","--context","kind-mlops-hw11","version","-o","json")),
         docker=run("docker","version","--format",'{{.Server.Version}}'),
         cpu=int(run("docker","info","--format",'{{.NCPU}}')),
         docker_memory_bytes=int(run("docker","info","--format",'{{.MemTotal}}')),
         kind=run("kind","version"),helm=run("helm","version","--short"),uv=run("uv","--version"),
         images=identities,image_logical_bytes=sum(image["bytes"] for image in identities),
         node_writable_bytes=container["SizeRw"],node_var_volume_bytes=volume_bytes,
         approximate_lab_logical_bytes=sum(image["bytes"] for image in identities)+container["SizeRw"]+volume_bytes,
         sizing_method="Docker image Size sum plus dedicated kind node SizeRw and allocated /var volume blocks. Shared layers and sparse host allocation differ.",
         containers=run("docker","ps","--size","--filter","name=mlops-hw11"),
         namespaces=run("kubectl","--context","kind-mlops-hw11","-n","hw11","get","pods","-o","wide"),
         disk=run("df","-h","."))
(RESULTS / "environment.json").write_text(json.dumps(lab,indent=2,ensure_ascii=False)+"\n")
print(f"Lab logical footprint ~{lab['approximate_lab_logical_bytes']/2**30:.2f} GiB")

for name, url, member in (
    ("grafana-dashboard.json","http://hw11-grafana:3000/api/dashboards/uid/mlops-hw11","dashboard"),
    ("prometheus-targets.json","http://hw11-prometheus:9090/api/v1/targets",None),
):
    code = f"import httpx,json; r=httpx.get({url!r},timeout=10); r.raise_for_status(); print(r.text)"
    data = json.loads(run("kubectl","--context","kind-mlops-hw11","-n","hw11",
                          "exec","hw11-client-0","--","python","-c",code))
    if member:
        data=data[member]
    (RESULTS / name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
