.PHONY: install up down sweep promote predict check check-storage
install:
	uv sync --frozen
up:
	docker compose -p mlops-hw6 up --build -d --wait
down:
	docker compose -p mlops-hw6 down
sweep:
	uv run python -m src.sweep
promote:
	uv run python -m evaluation.cli promote
predict:
	uv run python -m src.predict --uri models:/my-qwen-adapter@champion --prompt "Какой показатель отражает запасы железа?"
check:
	bash tests/check.sh
check-storage:
	uv run python tests/check_storage.py

.PHONY: report candidate baseline-init evaluate eval check-eval check-negative check-all baseline-update predict-candidate
report:
	uv run python -m src.report
candidate:
	uv run python -m evaluation.cli candidate
baseline-init:
	uv run python -m evaluation.cli baseline-init
evaluate:
	uv run python -m evaluation.cli evaluate
eval:
	uv run python -m evaluation.cli gate
check-eval:
	uv run python -m unittest discover -s tests -p 'test_*.py' -v
check-negative:
	uv run python -m evaluation.cli negative
check-all: check check-eval
baseline-update:
	uv run python -m evaluation.cli baseline-update
predict-candidate:
	uv run python -m src.predict --uri models:/my-qwen-adapter@candidate --prompt "Выберите правильный вариант."

.PHONY: prepare data-version restore-eval test
prepare:
	uv run python scripts/prepare_data.py
data-version:
	uv run dvc add data/eval
	uv run python scripts/package_eval.py
restore-eval:
	uv run python scripts/package_eval.py --restore
test: check-eval

.PHONY: repro finish report-final
repro:
	uv run python -m scripts.run_pipeline
finish:
	uv run python -m scripts.run_pipeline --after-sweep
report-final:
	uv run python -m scripts.finalize_report
