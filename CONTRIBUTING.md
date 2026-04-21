# Contributing to PhotoChron

Thanks for your interest in improving PhotoChron! This document covers the development setup and the conventions we follow.

## Development setup

Requires Python 3.12+.

```bash
git clone https://github.com/micschr0/image-age-sorter.git
cd image-age-sorter

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
pre-commit install
```

## Running tests

```bash
# Unit tests only (fast, no external models required)
pytest -v tests/ -m "not integration"

# Full suite (requires Ollama + InsightFace models locally)
pytest -v tests/

# With coverage
pytest --cov=src/photochron --cov-report=term-missing
```

Coverage target: 80% (enforced via `pyproject.toml`).

## Lint & type-check

```bash
ruff check .
mypy src/
```

Both are enforced via `pre-commit` and in CI.

## Coding conventions

- **Python 3.12+** features are allowed (use them).
- **Type hints everywhere.** `mypy` is configured strictly — no untyped defs.
- **Confidence scores on every AI result.** PhotoChron's core contract: low-confidence photos are flagged, not silently wrong.
- **Non-destructive.** Never modify files in a user's input directory. Always write copies.
- **Local-only.** No HTTP calls to external image-analysis APIs. All inference must run on-device.

## Branches & pull requests

- Branch from `main`.
- Keep PRs focused — one logical change per PR.
- Include tests for new behavior.
- Update `docs/CHANGELOG.md` for user-visible changes.
- Run `ruff check . && mypy src/ && pytest -v tests/ -m "not integration"` before pushing.

## Reporting bugs

Open a GitHub issue with:
- PhotoChron version (`python -m photochron --version`)
- OS and Python version
- Minimal reproduction (config snippet + command)
- Full error output

## Questions

Open a GitHub Discussion or issue — happy to help.
