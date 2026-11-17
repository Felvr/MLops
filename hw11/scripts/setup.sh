#!/usr/bin/env bash
# Dedicated kind cluster. Preserves the user's default kubeconfig and Docker images.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results .local
export KUBECONFIG="$PWD/.local/kubeconfig"
cluster=mlops-hw11
kubectl_ctx="kind-${cluster}"
if [[ ${1:-} == delete ]]; then
  kind delete cluster --name "$cluster"
  rm -f "$KUBECONFIG"
  exit
fi

df -h . > results/disk-before.txt
docker system df > results/docker-before.txt
docker version > results/docker-version.txt
docker info --format '{{.NCPU}} CPU {{.MemTotal}} bytes' > results/docker-capacity.txt
{
  kind version
  helm version --short
  uv --version
} > results/tool-versions.txt
if ! kind get clusters 2>/dev/null | grep -qx "$cluster"; then
  kind create cluster --name "$cluster" --image kindest/node:v1.32.2 --wait 180s
else
  kind export kubeconfig --name "$cluster" --kubeconfig "$KUBECONFIG"
fi
docker build -t mlops-hw11:1.0.0 .
kind load docker-image --name "$cluster" mlops-hw11:1.0.0
for image in prom/prometheus:v3.2.1 grafana/grafana:11.6.0 \
  registry.k8s.io/kube-state-metrics/kube-state-metrics:v2.15.0 \
  registry.k8s.io/metrics-server/metrics-server:v0.7.2; do
  docker image inspect "$image" >/dev/null 2>&1 || docker pull "$image"
  kind load docker-image --name "$cluster" "$image"
done
metrics_url=https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.7.2/components.yaml
curl -fsSL --retry 3 "$metrics_url" -o .local/metrics-server.yaml
kubectl --context "$kubectl_ctx" apply -f .local/metrics-server.yaml
kubectl --context "$kubectl_ctx" -n kube-system patch deployment metrics-server --type=json \
  -p='[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
helm upgrade --install hw11 chart --reset-values --kube-context "$kubectl_ctx" --namespace hw11 --create-namespace
kubectl --context "$kubectl_ctx" -n hw11 wait --for=condition=Ready pod -l app=hw11-client --timeout=150s
for deployment in hw11-api hw11-prometheus hw11-grafana hw11-kube-state; do
  kubectl --context "$kubectl_ctx" -n hw11 rollout status "deployment/$deployment" --timeout=150s
done
kubectl --context "$kubectl_ctx" version > results/kubernetes-version.txt

uv run --locked python scripts/environment.py
