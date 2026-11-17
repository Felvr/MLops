.PHONY: setup test experiments report check delete

setup:
	bash scripts/setup.sh

test:
	uv run --locked python -m unittest discover -s tests -v

experiments:
	uv run --locked python scripts/experiment.py A B C

report:
	uv run --locked --group report python scripts/report.py

check:
	uv run --locked python scripts/check.py

delete:
	bash scripts/setup.sh delete
