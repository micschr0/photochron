CLI Commands Reference

photochron run
Full pipeline run (all 6 stages).

photochron run --input PATH [OPTIONS]

Options:
--input PATH          Input directory with JPEG files (required)
--output PATH         Output directory (default: ./photochron_output)
--anchors PATH        anchors.yaml path (default: ./anchors.yaml)
--dry-run             Run pipeline, skip all file writes
--no-cache            Force re-inference, ignore Feature Store cache
--verbose             Show per-photo detail in terminal
--model MODEL         Override LLM model (e.g. moondream2)

Expected output:

- `{output}/renamed/*.jpg` – renamed copies
- `{output}/exif_enriched/*.jpg` – EXIF-enriched copies
- `{output}/photochron_report.json`
- `{output}/photochron_timeline.csv`
- Rich terminal table: photo | est. date | confidence | flags

photochron cluster
Face clustering + interactive person assignment (one-time setup).

photochron cluster --input PATH

Outputs a contact sheet per cluster to terminal (Rich) and prompts user to
assign names. Saves mapping to Feature Store `persons` table.
Must be run before `photochron run` if person-anchored ranking is desired.

photochron rerun
Re-run a single pipeline stage without full re-inference.

photochron rerun --stage STAGE

Stages: ingestion | face | context | ranking | output

Use case: anchors.yaml changed → `photochron rerun --stage ranking`

photochron report
Generate report from existing Feature Store (no inference).

photochron report [--format json|csv|table]

photochron config
Validate and display current configuration.

photochron config [--anchors PATH] [--config PATH]

photochron status
Show Feature Store stats and pipeline run history.

photochron status

Output: cached photos count, last run timestamp, model versions used, review-needed count.

Development Commands

# Run tests

pytest -v tests/

# Lint + type check

ruff check . && mypy src/

# Install dev dependencies

pip install -e ".[dev]"

# Reset Feature Store (WARNING: clears all cache)

rm .photochron/cache.db