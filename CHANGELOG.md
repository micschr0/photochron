# Changelog

All notable changes to photochron will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Hardening pass (audit + UX, May 2026)

#### Added
- **`photochron init`** — interactive first-time setup wizard. Collapses
  the old "edit config.yaml, then anchors.yaml, then verify licenses,
  then run doctor" flow into one prompt-driven session. `--no-input` and
  `--force` for scripting.
- **`photochron review`** — TUI that walks every photo with confidence
  below a threshold and lets you accept / edit / skip each AI guess.
  Edits persist into a new `review_overrides` table.
- **`photochron doctor` next-steps + `--json`** — every detected gap
  (missing onnxruntime, CoreML EP not available, missing model name,
  Ollama unreachable) is now followed by an exact remediation command.
  `--json` for scripting.
- **`photochron status --json`** for scripted health checks.
- **Rich progress bar** during pipeline runs (spinner + bar + elapsed
  time, rendered on stderr).
- **GitHub Actions CI** — lint (ruff), type-check (mypy), and unit-test
  jobs on Ubuntu + macOS. No secrets, no heavy ML deps; the integration
  suite remains a local `make test` command.
- **Dependabot** — weekly `pip` + `github-actions` update PRs.
- **`Makefile`** — `make check`, `make lint`, `make type`, `make
  test-fast`, `make cov`, `make fmt`.
- **`SECURITY.md`** — disclosure channel + threat model + EXIF
  embedding caveat.
- **`docs/faq.md`** and **`docs/architecture.md`** — first-day questions
  and a feature → module map.
- **`pipeline_stage_runs` ledger** (SCHEMA_VERSION=2, additive
  migration) lets `PipelineStage.should_run` skip stages that already
  finished for a given run_id.

#### Changed
- **Per-stage `should_run`** semantics: previously checked the whole-run
  status; now per `(run_id, stage_name)`. Resume-after-failure works.
- **`PipelineRunner` no longer mutates the `Config` singleton.** Frozen
  `RunContext` bound via `stage.bind_context(ctx)`; stages read
  `self.context.input_dir / output_dir / dry_run`.
- **`PipelineRegistry.get_dependency_order`** uses Kahn's topological
  sort with registration order as tiebreaker. Raises on cycles.
- **`PipelineStage.mark_failed`** persists the error message to both the
  per-stage ledger and `pipeline_runs.error_message` (truncated to 1024
  chars).
- **`face/insightface_wrapper.resolve_providers`** promoted from
  `_resolve_providers`. Underscore alias kept for one release.
- **Pre-commit hooks**: ruff v0.1.0 → v0.15.13; mypy → v1.20.0; push
  hook restricted to `tests/unit -m "not integration"`.
- **`pyproject.toml`**: single source of truth for dev deps is now
  `[dependency-groups].dev` (PEP 735). Linux + Windows + Python 3.13
  classifiers added. `[project.urls]` added. Legacy `License ::`
  classifier removed.
- **`CONTRIBUTING.md`** recommends `uv sync --group dev` and `make
  check`.
- **`anchors.yaml`** ships as a fully-commented-out template with a
  banner warning instead of realistic placeholder data.
- **`docs/__init__.py`** / **`examples/__init__.py`** deleted.
- **CHANGELOG** moved to repo root so GitHub auto-renders it.

#### Fixed
- **(P0) broken `git clone` URL** in README, CONTRIBUTING, and
  `docs/README.md` (pointed at non-existent `image-age-sorter` repo).
- **(P0) import-time `mkdir`** in `photochron/__init__.py` resolved
  into `site-packages/...` after `pip install`. Cache directory is now
  created lazily by `DatabaseStore`.
- **(P0) TODO-stub CLI commands** (`cluster`, `rerun`) hidden from
  `--help`.
- Pre-existing test bug in
  `test_analyze_with_different_strategies_on_error`.
- 313 → 322 passing unit tests.

#### Configuration changes
- `Config.input_dir` / `Config.dry_run` removed (they were runtime
  inputs; moved to `RunContext`). Code that wrote
  `get_config().input_dir = ...` now fails loudly via
  `extra="forbid"` — migrate to
  `PipelineRunner.run_pipeline(input_dir=..., output_dir=...,
  dry_run=...)`.
- SCHEMA_VERSION bumped to 2 via additive migration; existing v1
  databases auto-upgrade on next pipeline run.

### Added
- **Comprehensive Phase 3 testing suite**: Added extensive test coverage for Context Layer implementation:
  - **ContextAnalyzer strategy tests**: Tests for all 4 analysis strategies (DEFAULT, AGGRESSIVE, CONSERVATIVE, FAST)
  - **Integration tests**: Full pipeline integration tests for ContextLayerStage with database
  - **Error handling tests**: Comprehensive tests for retry logic and fallback strategies
  - **Confidence validation tests**: Tests for confidence score validation and propagation
  - **Database integration tests**: Tests for transaction integrity and schema compliance
- **Memory threshold configuration**: Added memory management fields to `ConfigContext`:
  - `memory_warning_threshold_mb`: Warning threshold for available memory (default: `100`)
  - `memory_critical_threshold_mb`: Critical threshold for available memory (default: `50`)
  - `memory_retry_delay_seconds`: Delay when memory is critically low (default: `30`)
- **Memory checking**: Before each batch, checks available system memory against configurable thresholds
  - Logs warning when memory falls below warning threshold
  - Skips batch and waits when memory falls below critical threshold
  - Graceful degradation if psutil unavailable
- **Progress reporting**: Enhanced batch processing with detailed percentage-based progress logging
  - Batch-level progress with percentage complete
  - Photo-level progress every 10 photos
  - Final completion percentage with failure count
- **Comprehensive error handling tests**: Added tests for:
  - JSON parsing fallback scenarios
  - Timeout handling for Ollama requests
  - Model fallback behavior
  - Memory check scenarios (warning, critical, error)
  - Progress reporting format and calculation
- **ConfigContext class**: New configuration model for context layer with comprehensive Ollama integration settings
- **Configuration validation**: Runtime health checks for Ollama server and model availability
- **Graceful degradation**: Automatic fallback to simpler models and degraded mode operation
- **Retry logic**: Configurable retry attempts with exponential backoff for transient failures
- **Health monitoring**: Real-time health status reporting via `health_status` property
- **Database integration tests**: Comprehensive tests for `_get_photos_without_context()` method

### Changed
- **Config class**: Updated to include `context` configuration section with memory management fields
- **ContextLayerStage initialization**: Modified to use `self.config.context` for configuration
- **ContextLayerStage.run()**: Enhanced with memory checking and progress reporting
- **Package structure**: Established complete Python package structure with `__init__.py` files in all directories

### Configuration Changes
- Added `context` section to configuration with the following options:
  - `ollama_host`: Ollama server URL (default: `http://localhost:11434`)
  - `ollama_timeout`: Timeout in seconds for Ollama requests (default: `300`)
  - `max_retries`: Maximum retry attempts for LLM failures (default: `3`)
  - `retry_delay`: Delay between retries in seconds (default: `2.0`)
  - `primary_model`: Primary vision LLM model (default: `llava-next:7b`)
  - `fallback_model`: Fallback vision model (default: `moondream2`)
  - `batch_size`: Batch size for processing images (default: `1`)
  - `min_decade_confidence`: Minimum confidence for decade estimates (default: `0.3`)
  - `min_season_confidence`: Minimum confidence for season estimates (default: `0.4`)
  - `use_fallback_on_failure`: Use fallback strategies on analysis failure (default: `true`)
  - `store_minimal_on_complete_failure`: Store minimal data when analysis completely fails (default: `true`)
  - `memory_warning_threshold_mb`: Memory warning threshold in MB (default: `100`)
  - `memory_critical_threshold_mb`: Memory critical threshold in MB (default: `50`)
  - `memory_retry_delay_seconds`: Delay in seconds when memory is critically low (default: `30`)

### Documentation
- Created comprehensive pipeline documentation in `docs/pipeline.md`
- Added configuration reference in `docs/configuration.md`
- Updated agent documentation to reflect new configuration structure

## [0.1.0] - 2026-04-13

### Added
- Initial project foundation with 6-stage pipeline architecture
- Ingestion stage with EXIF extraction and downsampling
- Face layer with InsightFace integration for face detection and age estimation
- Basic configuration management with Pydantic models
- SQLite feature store for caching inference results
- CLI interface with Typer framework
- Comprehensive test infrastructure

### Configuration
- Basic configuration structure with paths, models, ingestion, face, and pipeline sections
- Environment variable support for configuration overrides
- YAML configuration file support