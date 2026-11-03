.PHONY: check mutation bootstrap serve flow experiments benchmark
check:
	uv run --locked python scripts/check.py
mutation:
	uv run --locked python scripts/check.py --mutation
bootstrap:
	uv run --locked --extra model python scripts/bootstrap.py
serve:
	uv run --locked --extra model python service.py --workspace . --backend transformers --port 8091 --warmup-seconds 0
flow:
	uv run --locked --extra model python pipeline.py
experiments:
	uv run --locked --extra model python scripts/flow_experiments.py
benchmark:
	uv run --locked --extra model python scripts/benchmark.py
