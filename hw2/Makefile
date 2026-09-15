.PHONY: install inspect check test clean

install:
	uv sync --locked

inspect:
	uv run python -m src.inspect_model

check:
	bash tests/check.sh

test:
	uv run python -m unittest discover -s tests -v

clean:
	rm -rf docs/report.json src/__pycache__ tests/__pycache__
