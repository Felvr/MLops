FROM ghcr.io/astral-sh/uv:0.7.22@sha256:629240833dd25d03949509fc01ceff56ae74f5e5f0fd264da634dd2f70e9cc70 AS uv
FROM python:3.11.12-slim-bookworm@sha256:dbf1de478a55d6763afaa39c2f3d7b54b25230614980276de5cacdde79529d0c AS python
WORKDIR /app

FROM python AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/app/.venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-cache

FROM python
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/app/.venv/bin:$PATH"
COPY --from=builder /app/.venv /app/.venv
COPY service.py ./
COPY scripts/client.py ./scripts/
USER 10001
EXPOSE 8080
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-graceful-shutdown", "35", "--no-access-log"]
