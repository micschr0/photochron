# photochron — Claude guide

## Dev workflow (prefer the Makefile)

- `make check` — lint + type + unit tests; the "is my PR ready?" command (mirrors CI).
- `make cov` — unit tests + coverage (gate 80%, also in ci.yml + Makefile + pyproject.toml).
- `make fmt` — ruff format + autofix. `make install` first (= `uv sync --group dev` + pre-commit).
- All recipes run via `uv run`, so local and CI use the same toolchain versions.

## Tests

- `tests/unit` mock Ollama and InsightFace — no daemon/models needed; this is what CI runs.
- `make test` / `-m integration` need a local Ollama daemon + InsightFace weights (not in CI).

## Conventions (details in CONTRIBUTING.md)

- Conventional commits; branch from `main`, one logical change per PR.
- Non-destructive: never modify the input dir, always write copies; confidence score on every AI result.
- Multi-step / cross-session work: see `plans/` (durable, cold-start-resumable plans).
