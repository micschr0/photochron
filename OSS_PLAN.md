# photochron — OSS Hardening Plan

Derived from [`OSS_AUDIT.md`](./OSS_AUDIT.md). Scope of this pass:

- **In scope:** correctness & release-blocker fixes (P0), structural hardening (P1),
  tests + CI bootstrap, docs polish, **plus a UX layer** that makes day-1 use easier
  (interactive `init`, Rich progress, friendlier `doctor`, opt-in review of low-confidence
  results).
- **Out of scope:** redesign of the 6-stage pipeline, embedding-cache rewrite of the
  face wrapper, batch-inference work for InsightFace, PyPI release automation, swap of
  the build backend.

The plan is organised so each tier can be merged independently. Each item below is
sized to fit a single reviewable PR.

---

## Execution order

```
Tier 0 — Release blockers (P0)         must merge first
Tier 1 — Dev experience & tooling      depends on Tier 0
Tier 2 — CI                            depends on Tier 1 (lint/test commands must be stable)
Tier 3 — Pipeline correctness          depends on Tier 0 (no shared files with later tiers)
Tier 4 — Tests & fixtures              depends on Tier 3
Tier 5 — Docs & metadata               depends on Tier 2 (so the CI badge points somewhere)
Tier 6 — UX layer                      depends on Tier 1 (uses the Makefile + dev env)
Tier 7 — P2 polish                     last
```

**Stop points (do not auto-advance past):**

- After **Tier 0** if any of the four P0 fixes break local `uv sync` or `python -m photochron --help`.
- After **Tier 3** (pipeline correctness) if the golden-fixture ranking test regresses.
- After **Tier 4** if any newly introduced test is flaky on a clean clone.
- After **Tier 6** if the new UX surface confuses contributors more than the old CLI did
  (sanity-check by reading `photochron --help` and `photochron init --help` from scratch).

---

## Minimum shippable subset (single small PR)

If only one PR can land before public attention arrives, ship **Tier 0** only:

1. Fix the broken `git clone` URL everywhere.
2. Remove the import-time `mkdir` side-effect.
3. Hide the two TODO-stub CLI commands.
4. Replace PII-shaped sample data in `anchors.yaml`.

That alone removes the highest-embarrassment failures of a public release and is
under 50 lines of diff.

---

## Tier 0 — Release blockers (P0)

- [ ] **Fix broken `git clone` URLs**
  Purpose: the documented setup command currently points to a 404 (`image-age-sorter`); first-touch user experience is broken.
  Change points: `README.md` (lines 93-94), `CONTRIBUTING.md` (lines 10-11), `docs/README.md` (line 19).
  Acceptance criteria:
  - `grep -RIn "image-age-sorter" .` returns no matches.
  - Following the README cold leads to a working clone.
  Suggested commands:
  ```bash
  rg -l "image-age-sorter" | xargs sed -i 's|image-age-sorter|photochron|g'
  rg "image-age-sorter"    # expect zero hits
  ```
  Estimated impact radius: docs-only, three files, zero behaviour change.

- [ ] **Remove import-time filesystem side-effect from `__init__.py`**
  Purpose: `CACHE_DIR.mkdir(exist_ok=True)` runs at import time using a path that is wrong after `pip install` (resolves into `site-packages/...`).
  Change points: `src/photochron/__init__.py` (drop lines 15-21), `src/photochron/store/__init__.py` (let `DatabaseStore.__init__` lazily create `cache_dir` from `get_config().paths.cache_dir`).
  Acceptance criteria:
  - `python -c "import photochron"` performs no filesystem writes (verifiable with `strace -e openat,mkdir`, or simply by importing under a read-only `site-packages` mount).
  - The first call to `get_store()` creates the configured cache dir if missing.
  - All existing unit tests still pass.
  Suggested commands:
  ```bash
  uv run python -c "import photochron, os; print('cwd-ok')"
  uv run pytest tests/unit -k "store or config" -x
  ```
  Estimated impact radius: low; two files, contained behaviour change.

- [ ] **Hide TODO-stub CLI commands until implemented**
  Purpose: `photochron cluster` and `photochron rerun` are advertised in `--help` but print "not yet implemented".
  Change points: `src/photochron/cli/__init__.py` (mark both `app.command(...)` registrations with `hidden=True`, or move them under a `photochron experimental ...` sub-app).
  Acceptance criteria:
  - `python -m photochron --help` does not list `cluster` or `rerun`.
  - `python -m photochron cluster --help` still works (so future docs/PRs can link to it).
  - Unit test asserts the two commands are hidden.
  Suggested commands:
  ```bash
  uv run python -m photochron --help | grep -E "cluster|rerun"   # expect empty
  uv run python -m photochron cluster --help                     # still works
  ```
  Estimated impact radius: low; one file.

- [ ] **Replace PII-shaped sample data in `anchors.yaml`**
  Purpose: shipping realistic-looking names + birthdays invites copy-paste-and-forget mistakes when users commit their own anchor file.
  Change points: `anchors.yaml`. Add a top-of-file banner ("DO NOT COMMIT REAL BIRTHDAYS"). Use placeholders such as `id: example_person_a`, `name: "<your-name>"`, `birthday: "YYYY-MM-DD"`.
  Acceptance criteria:
  - `anchors.yaml` contains no plausible-real names or dates.
  - `parse_anchors` still loads the file without errors (validates the YAML shape).
  Suggested commands:
  ```bash
  uv run python -c "from pathlib import Path; from photochron.anchor.loader import load_anchors; print(load_anchors(Path('anchors.yaml')))"
  ```
  Estimated impact radius: docs/config only.

---

## Tier 1 — Dev experience & tooling

- [ ] **Single source of truth for dev deps**
  Purpose: `pyproject.toml` declares dev deps twice with conflicting floors (`[project.optional-dependencies].dev` has `ruff>=0.1.0`; `[dependency-groups].dev` has `ruff>=0.15.10`).
  Change points: `pyproject.toml` (drop `[project.optional-dependencies]`; expand `[dependency-groups].dev` to include the four missing tools — `pytest`, `pytest-cov`, `pre-commit`, plus the two already there).
  Acceptance criteria:
  - `uv sync --group dev` installs `ruff>=0.15.10`, `mypy>=1.20.0`, `pytest>=7`, `pytest-cov>=4`, `pre-commit>=3`.
  - `uv tree --group dev | grep -E "ruff|mypy|pytest"` reports versions matching `uv.lock`.
  Suggested commands:
  ```bash
  uv sync --group dev
  uv run ruff --version
  uv run mypy --version
  ```
  Estimated impact radius: contained; one file.

- [ ] **Bump pre-commit hook versions and limit push hook scope**
  Purpose: `ruff-pre-commit` pinned to `v0.1.0` (Oct 2023) drifts from `pyproject.toml`; push hook runs the full test suite, which needs Ollama + InsightFace.
  Change points: `.pre-commit-config.yaml` — `pre-commit autoupdate`, then change the `pytest` entry to `entry: uv run pytest tests/unit -x -m "not integration"`.
  Acceptance criteria:
  - `pre-commit run --all-files` passes on a clean clone with `uv sync --group dev`.
  - `git push` (with the push hook) does not require Ollama to be running.
  Suggested commands:
  ```bash
  uv run pre-commit autoupdate
  uv run pre-commit run --all-files
  ```
  Estimated impact radius: low; one file plus generated config update.

- [ ] **Add `Makefile` (or `justfile`) with canonical commands**
  Purpose: contributors currently must remember three separate commands (`ruff check .`, `mypy src/`, `pytest`) every PR.
  Change points: `Makefile` (new) with targets `lint`, `type`, `test`, `test-fast`, `check`, `fmt`. Optional: `justfile` instead, depending on preference.
  Acceptance criteria:
  - `make check` runs lint + type + fast tests and exits non-zero on any failure.
  - `make fmt` runs `ruff format .` and `ruff check --fix`.
  - `CONTRIBUTING.md` references `make check`.
  Suggested commands:
  ```bash
  make check
  make fmt
  ```
  Estimated impact radius: tooling-only; one new file plus two CONTRIBUTING lines.

- [ ] **Update `CONTRIBUTING.md` to use `uv`**
  Purpose: setup instructions still recommend `python -m venv` + `pip install -e ".[dev]"`, which hits the now-removed deprecated `[project.optional-dependencies]` block.
  Change points: `CONTRIBUTING.md` — switch to `uv sync --group dev` (with a fallback note for `pip install -e . --group dev` on pip ≥ 25).
  Acceptance criteria:
  - Following CONTRIBUTING.md from a fresh clone lands a working dev env.
  - `make check` passes after that setup.
  Suggested commands:
  ```bash
  rm -rf .venv && uv sync --group dev
  uv run make check
  ```
  Estimated impact radius: docs-only.

---

## Tier 2 — CI

- [ ] **Add `.github/workflows/ci.yml`**
  Purpose: PRs currently have zero automated gate. Match what `make check` does locally.
  Change points: `.github/workflows/ci.yml` (new). Jobs: `lint`, `type`, `test-fast`. Use `astral-sh/setup-uv@v6` with `uv sync --group dev`. Trigger on `pull_request` and `push` to `main`. Matrix: `ubuntu-latest` + `macos-latest`, Python 3.12 only. Cache via `setup-uv`'s built-in cache.
  Acceptance criteria:
  - First green run lands on a fresh PR.
  - No secrets referenced; no Ollama installed; no `insightface` model downloads.
  - `pytest --cov=src/photochron --cov-fail-under=80` runs in the `test-fast` job (or coverage threshold is intentionally lowered for unit tests only).
  Suggested commands (local dry-run via `act`):
  ```bash
  act -j test-fast
  ```
  Estimated impact radius: CI-only; one new file.

- [ ] **Add `.github/workflows/pre-commit.yml` (or fold into `ci.yml`)**
  Purpose: enforce that pre-commit hooks pass on every PR even if the contributor forgot `pre-commit install`.
  Change points: `.github/workflows/pre-commit.yml` (new). Uses `pre-commit/action@v3` with the dev env from `uv`.
  Acceptance criteria: red PR when ruff/mypy/yaml-check fails.
  Estimated impact radius: CI-only.

---

## Tier 3 — Pipeline correctness

- [ ] **Fix `PipelineStage.should_run` to be per-stage**
  Purpose: the current implementation checks the whole `pipeline_runs` row by `run_id`, not whether the specific stage already completed for that run. Docstring lies.
  Change points: `src/photochron/store/schema.py` (add `pipeline_stage_runs(run_id, stage_name, status, started_at, ended_at, error_message)` table). `src/photochron/pipeline/__init__.py` (`should_run`, `mark_complete`, `mark_failed`).
  Acceptance criteria:
  - New unit test: a stage with `should_run=True` runs; after `mark_complete`, the next call with the same `run_id` returns False.
  - Existing pipeline integration tests still pass (`pytest -m integration tests/integration/test_pipeline_flow.py`).
  - Schema migration is additive — no breaking change to the existing `pipeline_runs` table.
  Suggested commands:
  ```bash
  uv run pytest tests/unit -k "pipeline or stage" -x
  ```
  Estimated impact radius: medium; introduces one new table and changes two methods.

- [ ] **Stop mutating the `Config` singleton from `PipelineRunner`**
  Purpose: `run_pipeline` writes `input_dir`, `paths.output_dir`, `dry_run` into the shared singleton, poisoning later runs and test isolation.
  Change points: `src/photochron/pipeline/__init__.py` (introduce a small `RunContext` dataclass with `input_dir`, `output_dir`, `dry_run`, `run_id`, `config_hash`; pass it explicitly to `PipelineStage.run`).
  Acceptance criteria:
  - `Config` no longer carries `input_dir` and `dry_run` fields (or carries them as deprecated for one release with a `DeprecationWarning`).
  - All six stages accept `RunContext` in their `run()` signature.
  - Unit test: two back-to-back `run_pipeline` calls with different `input_dir`s do not leak into each other.
  Suggested commands:
  ```bash
  uv run pytest tests/unit -x
  uv run pytest tests/integration/test_pipeline_flow.py -x
  ```
  Estimated impact radius: medium; touches all six stage `run()` signatures.

- [ ] **Implement topological sort for stage execution order**
  Purpose: registration order currently doubles as execution order; a wrongly-ordered registration silently runs stages out of dependency.
  Change points: `src/photochron/pipeline/__init__.py:PipelineRegistry.get_dependency_order` (Kahn's algorithm, ~15 lines).
  Acceptance criteria:
  - New unit test: register stages out of order, assert `get_dependency_order()` returns the topo order.
  - Cycle detection raises `ValueError`.
  Suggested commands:
  ```bash
  uv run pytest tests/unit/test_pipeline_registry.py -x
  ```
  Estimated impact radius: low; single function plus one new test file.

---

## Tier 4 — Tests & fixtures

- [ ] **Fix `sample_image_path` fixture and reset all singletons in cleanup**
  Purpose: `sample_image_path` returns a non-existent path; `cleanup_global_state` only resets the database store, not `_config` or the pipeline `_registry`.
  Change points: `tests/conftest.py`. `sample_image_path` should write a tiny synthetic JPEG to `tmp_path` via `tests/fixtures/images.py`. `cleanup_global_state` should also reset `photochron.config._config` and `photochron.pipeline._registry`.
  Acceptance criteria:
  - Tests using `sample_image_path` no longer raise `FileNotFoundError`.
  - Running `pytest -p no:randomly tests/unit tests/unit --co` then `pytest tests/unit` deterministically (no order-dependent failures).
  Suggested commands:
  ```bash
  uv run pytest tests/unit -x --tb=short
  ```
  Estimated impact radius: test-only.

- [ ] **Mark `tests/integration/` tests with `pytestmark = pytest.mark.integration`**
  Purpose: `pytest -m "not integration"` currently does not exclude unmarked integration files.
  Change points: `tests/integration/conftest.py` (new) with a `pytest_collection_modifyitems` hook that adds the `integration` marker to every test under `tests/integration/`.
  Acceptance criteria:
  - `pytest -m "not integration" --collect-only | grep integration/` returns no test IDs.
  - CI's `test-fast` job runs only the unit tests.
  Suggested commands:
  ```bash
  uv run pytest -m "not integration" --collect-only -q | head
  ```
  Estimated impact radius: test-collection only.

---

## Tier 5 — Docs & metadata

- [ ] **Add `SECURITY.md`**
  Purpose: AGPL repo handling private family photos must have a disclosure channel and a "supported versions" note.
  Change points: `SECURITY.md` (new). Reference GitHub Security Advisories (private). Note that enriched EXIF (Mode B) embeds the full per-photo result JSON in `UserComment`; users sharing enriched copies should be aware.
  Acceptance criteria:
  - GitHub repo shows the "Security" tab populated.
  - README links to it.
  Estimated impact radius: docs-only.

- [ ] **Add `docs/faq.md` and link from README**
  Purpose: collect the top 5 first-day questions in one place (Ollama not running, onnxruntime CPU-only on macOS, low-memory, AGPL questions, "is my data uploaded?").
  Change points: `docs/faq.md` (new); add link in `README.md > Documentation` section.
  Estimated impact radius: docs-only.

- [ ] **Add `docs/architecture.md` (feature → module map)**
  Purpose: shorten the "where does the EXIF write live?" hunt for contributors.
  Change points: `docs/architecture.md` (new). One table mapping high-level feature → module → entry point.
  Estimated impact radius: docs-only.

- [ ] **Move `CHANGELOG.md` to the repo root**
  Purpose: GitHub auto-renders `CHANGELOG.md` from root on the Releases page; nested `docs/CHANGELOG.md` is invisible there.
  Change points: `git mv docs/CHANGELOG.md CHANGELOG.md`; update the README link.
  Estimated impact radius: low; one move + one link update.

- [ ] **Tidy `pyproject.toml` classifiers and metadata**
  Purpose: classifiers claim macOS only; README claims Linux/Windows are best-effort.
  Change points: `pyproject.toml`. Add `Operating System :: POSIX :: Linux`, `Operating System :: Microsoft :: Windows`. Add `urls.Homepage`, `urls.Repository`, `urls.Issues`.
  Estimated impact radius: metadata-only.

- [ ] **Delete `docs/__init__.py` and `examples/__init__.py`**
  Purpose: neither directory is a Python package; the marker file misleads tooling and readers.
  Change points: delete two files.
  Acceptance criteria:
  - `uv run pytest --collect-only` still passes.
  - `uv build` still produces a wheel without those files.
  Estimated impact radius: trivial.

---

## Tier 6 — UX layer (answer to "wie kann ich dem Nutzer die Bedienung erleichtern")

- [ ] **`photochron init` — interactive setup wizard**
  Purpose: first run currently requires editing two YAML files and verifying model licenses. A guided wizard collapses this to one prompt-driven step.
  Change points: `src/photochron/cli/commands.py` (new `init` command); `src/photochron/config/wizard.py` (new module — Rich `Prompt.ask`, `Confirm.ask`); `tests/unit/cli/test_init.py` (new).
  Behaviour:
  - Prompts for: photos dir, output dir, opt-in vision LLM model, opt-in face model, whether to enable GPS extraction (default no), whether to write a sample `anchors.yaml`.
  - Writes the chosen `config.yaml` and optionally a `anchors.yaml` template.
  - Re-runnable: confirms before overwriting existing files.
  - Exits cleanly under `--no-input` (CI / scripting friendly).
  Acceptance criteria:
  - Unit test pipes scripted answers via `typer.testing.CliRunner` and asserts the generated YAML round-trips through `Config.model_validate`.
  - Running `photochron init && photochron doctor` ends with `Reachable: yes` *or* clear remediation.
  Suggested commands:
  ```bash
  uv run python -m photochron init --no-input
  ```
  Estimated impact radius: contained; one new command + one new module + tests.

- [ ] **Rich progress bar across pipeline stages**
  Purpose: long runs currently emit only loguru log lines; users have no visual feedback on progress.
  Change points: `src/photochron/pipeline/__init__.py:PipelineRunner.run_pipeline` (wrap the stage loop in a Rich `Progress`); each stage exposes an optional `progress_callback(done, total)` it can call for per-photo updates.
  Acceptance criteria:
  - Running `photochron run --input small_set` shows a per-stage bar with photo counter.
  - `--quiet` (already on the root callback) suppresses the bar but keeps WARN+.
  - `--json-progress` *(optional follow-up)* emits one NDJSON line per stage start/end for IDE integration.
  Suggested commands:
  ```bash
  uv run python -m photochron run --input tests/fixtures/small_set --dry-run
  ```
  Estimated impact radius: medium; visual change only, no behaviour change.

- [ ] **`photochron doctor` — actionable remediation hints**
  Purpose: the existing `doctor` reports facts but no fix-it. UX win: a one-line "what to do next" per failure.
  Change points: `src/photochron/cli/commands.py:doctor` (extend the existing flow). Add a `--fix-hints` flag (default on) that, for each detected gap (no Ollama, CPU-only onnxruntime, missing model name), prints the exact command to fix it.
  Acceptance criteria:
  - On a clean machine, `photochron doctor` ends with a numbered "Next steps:" list referring to the missing pieces.
  - Already-healthy installs print "All checks passed." with no noise.
  Estimated impact radius: low; single command.

- [ ] **Friendlier first-run errors in `photochron run`**
  Purpose: `PipelineConfigurationError` currently prints a one-liner. UX: catch common errors and translate them to "do X" hints.
  Change points: `src/photochron/cli/commands.py:run` (catch `PipelineConfigurationError` and known runtime errors, emit Rich-formatted remediation that links to `photochron doctor` and `photochron init`).
  Acceptance criteria:
  - Running `photochron run --input ./photos` without any model configured exits with `[red]No AI model configured.[/red] Run `photochron init` to set one up.`
  Estimated impact radius: low.

- [ ] **`--json` output mode for `status` and `doctor`**
  Purpose: scriptable health check for users who want to wire `doctor` into Home Assistant / cron / NAS dashboards.
  Change points: `src/photochron/cli/commands.py` (`status`, `doctor`).
  Acceptance criteria:
  - `photochron doctor --json` emits a single JSON object with one boolean per check and a `next_steps` array.
  - Output validates against a small JSON schema in `tests/unit/cli/test_doctor_json.py`.
  Estimated impact radius: low.

- [ ] **Interactive review of low-confidence photos (`photochron review`)**
  Purpose: today the user must open `photochron_report.json` to find `review_needed=true` photos. UX: a TUI that walks them through low-confidence cases and lets them assign a year manually, writing back to the database.
  Change points: `src/photochron/cli/commands.py` (new `review` command); `src/photochron/review/tui.py` (new module — Rich `Prompt`, simple loop).
  Acceptance criteria:
  - `photochron review --threshold 0.5` walks every photo with confidence below the threshold.
  - User can `[s]kip`, `[a]ccept`, or `[e]dit year`; edits are persisted to the cache DB.
  - Unit test drives the TUI via scripted stdin.
  Estimated impact radius: medium; new command + new module; reuses the existing DB.

---

## Tier 7 — P2 polish (last)

- [ ] **Promote `face/insightface_wrapper._resolve_providers` to public**
  Purpose: it is already called from `cli/commands.py`; the underscore lies.
  Change points: rename and keep `_resolve_providers` as a deprecated alias for one release.

- [ ] **Surface EXIF-write failures in `photochron_report.json`**
  Purpose: today they only land in logs.
  Change points: `src/photochron/output/exif_writer.py`, `src/photochron/output/reports.py`.

- [ ] **Replace bare `Exception` prints in `cli/commands.py` with `logger.exception`**
  Purpose: keep the full traceback in the file sink so user-submitted bug reports can include it.
  Change points: `src/photochron/cli/commands.py` (four call sites).

- [ ] **Add `.github/dependabot.yml`**
  Purpose: weekly `pip` and `github-actions` updates so CVEs surface as PRs.

- [ ] **Track perf TODOs as a single follow-up issue**
  Purpose: keep the embedding-cache and batch-detect items out of the hardening pass.

---

## Notes on PR shape

Suggested PR slicing for a solo maintainer review-friendly cadence:

| PR | Tiers | Approx diff |
|----|------|-----|
| #1 | Tier 0 | < 80 lines |
| #2 | Tier 1 + Tier 5 (delete `__init__.py`, classifier tidy, CHANGELOG move) | < 200 lines |
| #3 | Tier 2 | < 100 lines (new files only) |
| #4 | Tier 3 | medium, mostly pipeline + new schema table + tests |
| #5 | Tier 4 | small, test-only |
| #6 | Tier 5 remainder (SECURITY, FAQ, architecture, README updates) | docs-only |
| #7 | Tier 6 (UX) | the largest, but additive — no changes to existing behaviour |
| #8 | Tier 7 | grab-bag of P2 polish |

For the **continuous hardening pass** the agent is currently executing, PRs #1–#3
and #5–#7 can be bundled into one feature branch (`hardening/audit-and-ux`) with
clearly-separated commits, and split out only if review requests it. Tier 3
(pipeline correctness) should remain its own PR because it is the only tier that
materially changes runtime behaviour.
