.PHONY: install prepare sample tokenize repro check test clean

install:
	uv sync --locked

prepare:
	uv run python -m scripts.prepare_data

# Совместимость с неизменённой проверкой курса: вход всегда из своего ДЗ3.
sample: prepare

tokenize:
	uv run python -m src.tokenize_data

repro:
	uv run dvc repro

check:
	bash tests/check.sh

test:
	uv run python -m unittest discover -s tests -v

# Удаляются только воспроизводимые выходы, исходные JSONL остаются.
clean:
	rm -rf data/tokenized
