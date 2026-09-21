.PHONY: install repro v1 v2 diff dag diversity contamination check test clean

install:
	uv sync --locked

repro:
	uv run dvc repro

v1:
	uv run python scripts/set_version.py v1
	uv run dvc repro

v2:
	uv run python scripts/set_version.py v2
	uv run dvc repro

diff:
	uv run dvc metrics diff hw3-v1 hw3-v2 --md

dag:
	uv run dvc dag

diversity:
	uv run python -m src.diversity

contamination:
	uv run python scripts/check_contamination.py

check:
	bash tests/check.sh

test:
	uv run python -m unittest discover -s tests -v

clean:
	rm -rf data metrics/*.json params.yaml.bak params.yaml.orig
