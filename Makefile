# photochron — developer task runner.
#
# Common entry points:
#   make check     run lint + type + unit tests (the canonical "is my PR
#                  ready" command — what CI runs too)
#   make lint      ruff check
#   make fmt       ruff format + ruff check --fix
#   make type      mypy src/
#   make test      full pytest (needs Ollama + InsightFace installed)
#   make test-fast unit tests only, no integration deps required
#   make cov       unit tests with coverage report
#
# All recipes assume `uv sync --group dev` has been run; everything is
# invoked via `uv run` so the same toolchain versions are used in CI and
# locally.

UV ?= uv
PY ?= $(UV) run

.PHONY: help check lint fmt type test test-fast cov clean install

help:
	@echo "photochron — make targets"
	@echo ""
	@echo "  make install      uv sync --group dev (one-time setup)"
	@echo "  make check        lint + type + unit tests (canonical PR check)"
	@echo "  make lint         ruff check"
	@echo "  make fmt          ruff format + ruff --fix"
	@echo "  make type         mypy src/"
	@echo "  make test         full pytest (needs Ollama + InsightFace)"
	@echo "  make test-fast    unit tests only"
	@echo "  make cov          unit tests with coverage report"
	@echo "  make clean        remove caches and build artifacts"

install:
	$(UV) sync --group dev
	$(PY) pre-commit install

check: lint type test-fast

lint:
	$(PY) ruff check .

fmt:
	$(PY) ruff format .
	$(PY) ruff check --fix .

type:
	$(PY) mypy src/

test:
	$(PY) pytest

test-fast:
	$(PY) pytest tests/unit -m "not integration" -q

cov:
	$(PY) pytest tests/unit -m "not integration" \
	    --cov=src/photochron --cov-report=term-missing --cov-fail-under=80

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
