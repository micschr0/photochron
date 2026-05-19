# photochron — OSS Audit

Scope: hardening + UX (no pipeline rewrite). Read-only audit of the repository at
`origin/main` (commit `c448399`, "chore: prepare for public release").

## 1. Repository summary

| Field | Value |
|---|---|
| Purpose | Local-first CLI that sorts undated digitized photos chronologically using on-device face age estimation + vision-LLM context analysis + user anchors |
| Language | Python 3.12+ (single language, plain Python) |
| Package | `src/`-layout, hatch/setuptools build via `pyproject.toml`, `uv.lock` present |
| Entry points | `photochron` script and `python -m photochron`, both call `photochron.cli:app` (Typer) |
| Lint / type / format | `ruff` (lint + format), `mypy --strict-ish` configured |
| Tests | `pytest`, ~30 test files, unit / integration / pipeline split, golden-fixture for ranking |
| Pre-commit | `.pre-commit-config.yaml` exists (ruff + mypy + pytest-on-push) |
| **CI** | **none** — no `.github/workflows/` directory |
| License | AGPL-3.0-or-later, `LICENSE` present, classifier set |
| Reproducibility | `uv.lock` is the authoritative pin; `pyproject.toml` only carries floors |
| Privacy claim | "All inference runs fully on-device" — verified at code level (no HTTP calls to external image APIs) |
| Platform | Apple Silicon tested; Linux/Windows code paths exist but unverified |
| Heavy runtime deps | `insightface`, `onnxruntime`, `ollama` (vision LLM via local Ollama daemon) |

### High-level risks

1. **Public release with broken `git clone` URLs** (P0). Both `README.md` and
   `CONTRIBUTING.md` instruct users to clone `github.com/micschr0/image-age-sorter.git` —
   that repo does not exist; the actual repo is `photochron`.
2. **No CI** (P0). PR review has no automated lint/type/test gate.
3. **Import-time filesystem side effect** in `src/photochron/__init__.py` creates
   `<project_root>/.photochron` on every import. After `pip install`, that path is
   `site-packages/...` — writes into the install location and is wrong-by-design.
4. **Two TODO-stub CLI commands** (`cluster`, `rerun`) are exposed in the public help
   but only print "not yet implemented" — surfaces unfinished surface area on day one.
5. **No `SECURITY.md`** — AGPL repo handling private family photos with no disclosure
   channel.

### What works well

- Architecture (6-stage SQLite-cached pipeline) is cleanly separated and documented.
- Pydantic v2 configuration with `extra="forbid"`, env-var overrides, and a
  migration path (`migrate_config`) is unusually mature for an early-alpha repo.
- Privacy posture is concrete: GPS extraction is opt-in, models are opt-in (no
  license-questionable defaults), prompts and weights live in config not source.
- Test suite scaffolding (unit/integration split, golden fixtures, loguru ↔ caplog
  bridge) is well above average.

---

## 2. Findings

### 2.1 Correctness

| # | Pri | Finding | Evidence | Why it matters | Smallest credible fix |
|---|---|---|---|---|---|
| C1 | **P0** | `__init__.py` does `CACHE_DIR.mkdir()` at import time, computing the path relative to `PROJECT_ROOT = Path(__file__).parent.parent.parent`. | `src/photochron/__init__.py:17-21` | After `pip install`, `PROJECT_ROOT` resolves to whatever lives two dirs above `site-packages/photochron/` — `mkdir` may write into the install root, or silently target a meaningless directory. The cache dir must come from the runtime `ConfigPaths.cache_dir`, not the install layout. | Remove the import-time `mkdir`. Compute cache dir lazily in `DatabaseStore.__init__` from `get_config().paths.cache_dir`. |
| C2 | P1 | `PipelineStage.should_run` claims to short-circuit completed stages, but the SQL only checks the `pipeline_runs` row by `run_id` — not by stage. | `src/photochron/pipeline/__init__.py:53-72` | The docstring promises per-stage skipping. The code returns True iff the whole run is incomplete, which is identical for all stages of a given run. Resume after stage failure cannot work. | Introduce a `pipeline_stage_runs(run_id, stage_name, status)` table, or check the per-stage output table for that `run_id`. Update docstring either way. |
| C3 | P1 | `PipelineRunner.run_pipeline` mutates the global `Config` singleton (`config.input_dir`, `config.paths.output_dir`, `config.dry_run`). | `src/photochron/pipeline/__init__.py:228-232` | The singleton in `photochron.config.get_config` is shared across the process and across tests. Mutation poisons subsequent runs and breaks parallel test isolation. | Pass `run_inputs` through the stage `run()` signature (or via a small `RunContext`) instead of writing into the singleton. |
| C4 | P1 | `PipelineRegistry.get_dependency_order` returns registration order; comment admits "in a real implementation we'd do topological sort". | `src/photochron/pipeline/__init__.py:119-127` | If a stage is registered before its declared dependency, the pipeline silently runs stages out of order. `validate_dependencies` only checks existence. | Implement topological sort (Kahn's algorithm, ~15 lines). Stable order via insertion order as tiebreaker. |
| C5 | P2 | `PipelineStage.mark_failed(run_id, error)` accepts an error string but never persists it. | `src/photochron/pipeline/__init__.py:87-99` | Failure forensics are limited to log files. The `# Log error (could add error logging table)` comment confirms intent. | Add `error_message` column to `pipeline_runs`, or persist a row in a `pipeline_errors` table. |
| C6 | P2 | Every stage's `mark_complete` overwrites `pipeline_runs.photos_processed`. Final value reflects the last stage, not pipeline-wide volume. | `src/photochron/pipeline/__init__.py:74-85` | Misleading status output. | Either move per-stage counts into a new column / table, or stop overwriting and only update from the final stage. |
| C7 | P2 | `output/exif_writer.py` catches a bare `Exception` and swallows the failure with a `logger.warning`. | `src/photochron/output/exif_writer.py:92-94` | EXIF write failures on Mode B output go unobserved by the report unless the user reads logs. | Surface the failure in the per-photo report row (e.g., set `exif_written=False` with reason). |

### 2.2 Maintainability

| # | Pri | Finding | Evidence | Why it matters | Smallest credible fix |
|---|---|---|---|---|---|
| M1 | P1 | `pyproject.toml` declares dev dependencies twice with conflicting floors. `[project.optional-dependencies].dev` has `ruff>=0.1.0`, `mypy>=1.0.0`. `[dependency-groups].dev` has `ruff>=0.15.10`, `mypy>=1.20.0`. | `pyproject.toml:38-45` vs `116-120` | `pip install -e ".[dev]"` (the path documented in `CONTRIBUTING.md`) reads the first block and may pull a 2-year-old `ruff` whose ruleset diverges from the lockfile. `uv sync` reads the second. | Drop `[project.optional-dependencies].dev`; keep PEP-735 `[dependency-groups].dev` as the single source of truth. Update `CONTRIBUTING.md` to recommend `uv sync` or `pip install -e . --group dev` (pip 25+). |
| M2 | P1 | `cluster` and `rerun` CLI commands print `[yellow]Note: ... not yet implemented[/yellow]` and exit. They are exposed in `--help`. | `src/photochron/cli/commands.py:97-162` | Public CLI surface advertises functionality that does not exist. Users discover the gap by trying. | Either hide them (`hidden=True` on `app.command`) until implemented, or move them behind `photochron experimental cluster` / a `PHOTOCHRON_EXPERIMENTAL=1` env-gate. |
| M3 | P1 | `.pre-commit-config.yaml` pins `ruff-pre-commit` to `v0.1.0` (Oct 2023). Current is `v0.13.x`. | `.pre-commit-config.yaml:11-13` | Pre-commit and `pyproject.toml` will lint with different rule sets, and `pre-commit autoupdate` has not been run. | `pre-commit autoupdate`, then verify clean. |
| M4 | P1 | `.pre-commit-config.yaml` push-stage hook runs `python -m pytest tests/ -x` (full suite). The full suite needs `insightface`, `onnxruntime`, and a reachable Ollama. | `.pre-commit-config.yaml:24-31` | A laptop without Ollama running can't `git push`. Surprising. | Limit the push hook to `pytest tests/unit -x -m "not integration"` and let CI run the full suite. |
| M5 | P2 | `docs/` and `examples/` contain `__init__.py` files, turning them into Python packages. | `docs/__init__.py`, `examples/__init__.py` | They are picked up by `setuptools.packages.find` (because of the `where = ["src"]` scope it actually isn't, but the markers still mislead readers and editors). Documentation directories should not look like packages. | Delete both `__init__.py` files. |
| M6 | P2 | `Optional[T]` and `T \| None` are mixed across the source. `Optional` is imported in `cli/__init__.py` and `store/__init__.py` but not used in either file after the typing migration. | `src/photochron/cli/__init__.py:6`, `src/photochron/store/__init__.py:13` | Dead import + style drift. Ruff `UP007` would catch the mixed form if enabled. | Add `UP007` (or rely on existing `UP`) and let ruff autofix. Remove dead `Optional` imports. |
| M7 | P2 | `face/insightface_wrapper._resolve_providers` is named private but called from `cli/commands.py` and from elsewhere. | `src/photochron/cli/commands.py:218, 251, 285` | Leading underscore is a lie; refactors that "clean up" the symbol will break the CLI. | Promote to `resolve_providers` (public), keep the old name as a deprecated alias for one release. |
| M8 | P2 | `pyproject.toml` build-system is `setuptools` but the rest of the toolchain is `uv`-native. | `pyproject.toml:1-3` | `uv build` works, but most modern uv-driven projects prefer `hatchling` for faster build and simpler `[tool.hatch.*]` configuration. | Optional: swap to `hatchling` later; not blocking. |

### 2.3 Testability

| # | Pri | Finding | Evidence | Why it matters | Smallest credible fix |
|---|---|---|---|---|---|
| T1 | P1 | `tests/conftest.py:sample_image_path` returns a non-existent path with a TODO. | `tests/conftest.py:75-80` | Any test consuming the fixture either silently skips or fails with a confusing "FileNotFoundError". Reduce surprise. | Use `tests/fixtures/images.py` (already exists) to generate a real on-the-fly PIL image and write it to a `tmp_path`. |
| T2 | P1 | `cleanup_global_state` autouse fixture only resets the database store, not `_config` and not the pipeline `_registry`. | `tests/conftest.py:113-120` | Tests that touch `get_config()` or register stages leak state into the next test. Order-dependent failures will be hard to diagnose. | Add `from photochron import config as _cfg; _cfg._config = None` and `from photochron.pipeline import _registry as _reg; _reg = None` (via helper) to the fixture. |
| T3 | P2 | `pyproject.toml:fail_under = 80` is configured but no command currently enforces it (no CI, and `pytest --cov` is optional in CONTRIBUTING). | `pyproject.toml:114` | Coverage drift goes unnoticed. | Add `--cov=src/photochron --cov-fail-under=80` to the CI test job. |
| T4 | P2 | Integration tests under `tests/integration/` rely on real Ollama + InsightFace; many are not marked `@pytest.mark.integration`, so `pytest -m "not integration"` does not exclude them (they aren't matched by the marker, but they also aren't skipped via the marker mechanism). | `tests/integration/test_*.py` | Confusing convention; new tests will copy the wrong template. | Add `pytestmark = pytest.mark.integration` to each file, or autodetect by path via a `conftest.py` in that directory. |

### 2.4 Security

| # | Pri | Finding | Evidence | Why it matters | Smallest credible fix |
|---|---|---|---|---|---|
| S1 | P1 | No `SECURITY.md`. | repo root listing | Project handles private family photos and writes EXIF. There must be a clear vuln-disclosure channel before a public release. | Add a small `SECURITY.md` with private contact (security advisory or email) and a "supported versions" line. |
| S2 | P1 | `anchors.yaml` ships with realistic-looking PII placeholders ("Mama", "Papa", "Anna" with concrete birthdays). | `anchors.yaml:6-22` | Users may copy-paste-and-forget; demo data near real data is a known footgun, especially since `anchors.yaml` is read on every run. | Replace with explicit `<EXAMPLE>` placeholders (e.g., `id: example_person_a`, `name: "<your-name>"`, `birthday: "YYYY-MM-DD"`) and add a banner comment "DO NOT COMMIT REAL DATES". |
| S3 | P2 | EXIF write puts the full result JSON into `UserComment`. | `src/photochron/output/exif_writer.py:84-87` | If the user shares an enriched copy, the JSON includes per-signal confidence and possibly person hints. Document this in `SECURITY.md`. | Document in `SECURITY.md`. Add a config flag `output.embed_full_result_in_exif` (default true) so privacy-sensitive users can opt out. |
| S4 | P2 | No supply-chain hygiene workflow (Dependabot / Renovate / `pip-audit`). | repo listing | AGPL repo with heavy ML deps; CVE notifications would be cheap to enable. | Add `.github/dependabot.yml` for `pip` weekly + `github-actions` weekly. |

### 2.5 Performance

| # | Pri | Finding | Evidence | Why it matters | Smallest credible fix |
|---|---|---|---|---|---|
| P1 | P2 | `InsightFaceWrapper.batch_detect` is a plain Python loop. | `src/photochron/face/insightface_wrapper.py:237-256` | Self-acknowledged as a known limitation. Not a blocker. | Track as a separate optimisation; leave the code with a `# TODO(perf)` comment that references an issue. |
| P2 | P2 | `compute_embedding` re-runs detection on a cropped face. | `src/photochron/face/insightface_wrapper.py:181-209` | Wastes a detection pass per embedding when one already exists. | Cache embedding from `detect_faces` results. Out of scope for this hardening pass; document with `# TODO(perf)`. |

### 2.6 Observability

| # | Pri | Finding | Evidence | Why it matters | Smallest credible fix |
|---|---|---|---|---|---|
| O1 | P1 | `status` and `doctor` swallow exceptions with `# noqa: BLE001` and print a one-line message. | `src/photochron/cli/commands.py:209-211, 226-227, 307-308, 320-322` | Users have no actionable signal when something fails; loguru-formatted exception traces never reach the console. | Run with `logger.exception(...)` so the file sink keeps the traceback, and surface a "run with `--verbose` for details" hint. |
| O2 | P2 | No `--json` output mode for `status`, `doctor`, or pipeline runs. | CLI surface | Hard to script around. | Add `--json` to `status` and `doctor` (small change). Pipeline runs already produce `photochron_report.json`. |

### 2.7 Documentation

| # | Pri | Finding | Evidence | Why it matters | Smallest credible fix |
|---|---|---|---|---|---|
| D1 | **P0** | `README.md`, `CONTRIBUTING.md`, and `docs/README.md` instruct users to `git clone https://github.com/micschr0/image-age-sorter.git` and `cd image-age-sorter`. The repo is `photochron`. | `README.md:93-94`, `CONTRIBUTING.md:10-11`, `docs/README.md:19` | First five seconds of the user journey: 404. Hard release-blocker. | Replace with the actual repo URL and directory name. |
| D2 | P1 | No CI badge in README (no CI exists yet). | `README.md` head | Once CI exists, the badge is a strong "this is healthy" signal. | Add CI status + license badge after CI is added. |
| D3 | P1 | No FAQ / troubleshooting section in README or docs. | docs listing | Two predictable pain points (Ollama not running, `onnxruntime` CPU-only on macOS) are mentioned but not centralised. | Add `docs/faq.md` with the top 5 questions (mostly already drafted in README install notes). |
| D4 | P1 | No `docs/architecture.md` overview that maps "feature → module". | docs listing | `pipeline.md` covers the 6 stages but new contributors still hunt for "where does the EXIF writing live". | Short table mapping public feature → module → entry point. |
| D5 | P2 | `pyproject.toml` `description` and `classifiers` reference `Operating System :: MacOS` only. | `pyproject.toml:18` | Limits PyPI discoverability and contradicts the README's "should run on Linux/Windows". | Add `Operating System :: POSIX :: Linux` and `Operating System :: Microsoft :: Windows`, plus `Operating System :: OS Independent` if the unverified platforms are best-effort. |

### 2.8 CI / Reproducibility / Release

| # | Pri | Finding | Evidence | Why it matters | Smallest credible fix |
|---|---|---|---|---|---|
| CI1 | **P0** | No `.github/workflows/*`. | repo listing | No automated gate for `ruff`, `mypy`, or `pytest tests/unit` on PRs. | Add `.github/workflows/ci.yml` that runs lint + type-check + unit tests on Python 3.12 across `ubuntu-latest` and `macos-latest`. No secrets, no heavy ML deps. |
| CI2 | P1 | No `Makefile` / task runner. CONTRIBUTING.md lists three commands users must remember and chain manually. | `CONTRIBUTING.md:36-42, 79-81` | Friction for contributors. | Add a `Makefile` (or `just`/`uv run` recipes) with `make lint`, `make type`, `make test`, `make test-fast`, `make check`. |
| CI3 | P2 | No release automation. | repo listing | Manual `uv publish` works but invites mistakes. | Out of scope for this pass; track in CHANGELOG. |

---

## 3. File-level change list

| Priority | Path | Action | Reason |
|----------|------|--------|--------|
| **P0** | `README.md` | edit | Fix `image-age-sorter` clone URL → `photochron`. |
| **P0** | `CONTRIBUTING.md` | edit | Same broken clone URL. |
| **P0** | `docs/README.md` | edit | Same broken clone URL. |
| **P0** | `src/photochron/__init__.py` | edit | Remove import-time `CACHE_DIR.mkdir()`; let `DatabaseStore` create its own directory lazily from `ConfigPaths.cache_dir`. |
| **P0** | `.github/workflows/ci.yml` | add | Lint + type-check + unit tests on PRs, matrix across Python 3.12 + macOS/Linux. |
| **P0** | `.github/workflows/pre-commit.yml` *(optional split)* | add | Or fold into `ci.yml`. Run `pre-commit run --all-files`. |
| P1 | `pyproject.toml` | edit | Drop `[project.optional-dependencies].dev`; keep `[dependency-groups].dev`. Update lower bounds to match the lockfile. Add Linux/Windows classifiers. |
| P1 | `.pre-commit-config.yaml` | edit | Bump `ruff-pre-commit` to current; bump `mirrors-mypy` to current; limit push hook to `tests/unit`. |
| P1 | `CONTRIBUTING.md` | edit | Switch to `uv sync` (or `pip install -e . --group dev`) and `make check`. |
| P1 | `Makefile` | add | One-line targets: `lint`, `type`, `test`, `test-fast`, `check`, `fmt`. |
| P1 | `SECURITY.md` | add | Vuln-disclosure channel, supported versions, note on EXIF embedding. |
| P1 | `anchors.yaml` | edit | Replace realistic PII placeholders with `<EXAMPLE>` markers. |
| P1 | `src/photochron/cli/__init__.py` | edit | Hide `cluster` and `rerun` (Typer `hidden=True`) until implemented, or move behind `experimental` sub-app. |
| P1 | `src/photochron/cli/commands.py` | edit | Replace bare-`Exception` print with `logger.exception` + user-facing "run with `--verbose`" hint. Add `--json` output to `status` and `doctor`. |
| P1 | `src/photochron/pipeline/__init__.py` | edit | Fix `should_run` semantics (per-stage check) and stop mutating the config singleton. Add topo sort to `get_dependency_order`. |
| P1 | `tests/conftest.py` | edit | Fix `sample_image_path` fixture, reset `_config` and `_registry` in cleanup. |
| P1 | `docs/faq.md` | add | Ollama not running, onnxruntime CPU-only on macOS, low-memory machine, AGPL question. |
| P1 | `docs/architecture.md` | add | Feature → module map. |
| P2 | `docs/__init__.py` | delete | Docs are not a Python package. |
| P2 | `examples/__init__.py` | delete | Examples are not a Python package. |
| P2 | `.github/dependabot.yml` | add | Weekly `pip` + `github-actions` updates. |
| P2 | `src/photochron/output/exif_writer.py` | edit | Surface EXIF write failures in the report row (`exif_written=False, exif_error="…"`). |
| P2 | `src/photochron/face/insightface_wrapper.py` | edit | Rename `_resolve_providers` → `resolve_providers`; keep deprecated alias for one release. |
| P2 | `CHANGELOG.md` *(promote `docs/CHANGELOG.md`)* | move | Put `CHANGELOG.md` at the repo root, where GitHub auto-renders it on Releases. |
| P2 | UX layer (see plan) | add | `photochron init` wizard, Rich progress bar, friendlier `doctor`, opt-in interactive `cluster` review — UX answer for "wie kann ich dem Nutzer die Bedienung erleichtern". |

---

## 4. Do first / do later

### Do first (release blockers, ≤ 2 hours of work)

1. **Fix the clone URL everywhere** (`README.md`, `CONTRIBUTING.md`, `docs/README.md`). Single grep + sed.
2. **Remove the import-time `mkdir` in `src/photochron/__init__.py`.** Move cache dir creation into `DatabaseStore.__init__` using `get_config().paths.cache_dir`.
3. **Add the smallest possible CI workflow** — `actions/setup-python@v5`, `astral-sh/setup-uv@v6`, `uv sync --group dev`, `make check` (or three direct commands). No matrix yet, just `ubuntu-latest` + Python 3.12.
4. **Hide the two TODO-stub CLI commands** (`cluster`, `rerun`) so the public help surface only advertises what works.
5. **Replace PII-shaped sample data** in `anchors.yaml`.

### Do next (this hardening pass)

6. Deduplicate dev deps in `pyproject.toml`; bump pre-commit pins.
7. Add `SECURITY.md` and `docs/faq.md`.
8. Add `Makefile` with `lint / type / test / check / fmt` targets.
9. Fix `PipelineStage.should_run` semantics and pipeline registry topological sort.
10. Reset `_config` / `_registry` in the autouse fixture.
11. Promote `_resolve_providers` to public; replace bare-Exception prints with `logger.exception`.
12. Move `CHANGELOG.md` to repo root.
13. Add the UX layer (interactive `photochron init` wizard; Rich progress bar wrapping the pipeline iteration; richer `doctor` output with actionable fix-it hints; opt-in interactive review for low-confidence photos). This is the answer to *"womit kann ich dem Nutzer die Bedienung erleichtern"*.

### Do later (P2, future iterations)

14. Topological sort + per-stage failure persistence.
15. Embedding cache in `face/insightface_wrapper.py`.
16. Dependabot + pip-audit.
17. PyPI publish workflow.
18. Switch build backend to `hatchling`.

---

## 5. Assumptions / blockers

- The audit ran without executing the test suite (`insightface` + `onnxruntime` + Ollama not available in the audit sandbox). The findings are based on static inspection plus the project's own docs. The Plan stage will spot-check by attempting `uv sync --group dev && uv run pytest tests/unit -k "not integration"` once an environment is ready.
- The license is settled (AGPL-3.0-or-later) and not subject to change in this pass.
- The project explicitly *wants* to stay local-first; no recommendation suggests adding network features.
