# Contributing to photochron

Thanks for your interest in improving photochron! This document covers the development setup and the conventions we follow.

## Development setup

Requires Python 3.12+.

```bash
git clone https://github.com/micschr0/photochron.git
cd photochron

# Recommended:
uv sync --group dev
uv run pre-commit install

# Plain-pip fallback (requires pip >= 25 for --group):
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e . --group dev
pre-commit install
```

## Running tests

The fastest path is the `Makefile`:

```bash
make check       # lint + type + unit tests (what CI runs)
make test-fast   # unit tests only, no integration deps
make test        # full suite (needs Ollama + InsightFace installed)
make cov         # unit tests with coverage report
```

Equivalent raw commands:

```bash
# Unit tests only (fast, no external models required)
uv run pytest tests/unit -m "not integration"

# Full suite (requires Ollama + InsightFace models locally)
uv run pytest tests/

# With coverage
uv run pytest tests/unit --cov=src/photochron --cov-report=term-missing
```

Coverage target: 80% (enforced via `pyproject.toml`).

## Lint & type-check

```bash
make lint        # ruff check .
make type        # mypy src/
make fmt         # ruff format + ruff --fix
```

Both lint and type-check are enforced via `pre-commit` and in CI.

## Coding conventions

- **Python 3.12+** features are allowed (use them).
- **Type hints everywhere.** `mypy` is configured strictly — no untyped defs.
- **Confidence scores on every AI result.** photochron's core contract: low-confidence photos are flagged, not silently wrong.
- **Non-destructive.** Never modify files in a user's input directory. Always write copies.
- **Local-only.** No HTTP calls to external image-analysis APIs. All inference must run on-device.
- **Models are opt-in.** Do not restore hardcoded model defaults; users must uncomment model entries in `config.yaml` after verifying licenses.

## Configuration via environment variables

Every field in `config.yaml` can be overridden via `PHOTOCHRON_<SECTION>_<KEY>`.
Sections are matched against the known top-level keys (`paths`, `models`,
`ingestion`, `face`, `pipeline`, `context`, `logging`); everything after the
section name becomes the field name, so multi-word keys like `ollama_host`
work as expected.

```bash
# Point at a remote Ollama service (useful in container/cloud setups)
export PHOTOCHRON_CONTEXT_OLLAMA_HOST=http://ollama.internal:11434

# Enable GPU for InsightFace
export PHOTOCHRON_FACE_USE_GPU=true

# Use a larger working image size
export PHOTOCHRON_MODELS_MAX_IMAGE_SIZE=2048
```

Values are parsed as int, float, or bool (`true/false/yes/no/1/0`) when
possible; otherwise kept as strings.

## Branches & pull requests

- Branch from `main`.
- Keep PRs focused — one logical change per PR.
- Include tests for new behavior.
- Update `CHANGELOG.md` for user-visible changes.
- Run `make check` before pushing.

## Reporting bugs

Open a GitHub issue with:
- photochron version (`python -m photochron --version`)
- OS and Python version
- Minimal reproduction (config snippet + command)
- Full error output

## Questions

Open a GitHub Discussion or issue — happy to help.
