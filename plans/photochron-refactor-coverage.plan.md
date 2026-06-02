# Plan: photochron — Split oversized modules (A) + raise unit coverage to ≥90% (C)

> Durable source of truth for an **autonomous-long-run** program.
> A cold agent with zero chat context must resume from THIS file alone.
> Pairs with `photochron-refactor-coverage-DECISIONS.md` + `-baseline.json`.
> Rev 2 — incorporates Phase-2 adversarial-review fixes (see Plan-mutation log).

## Objective

- **A** — Split the two modules over the ~800-line ceiling into focused sub-modules,
  **behavior-preserving** (no public-API change, no prompt-text change, no test deletion):
  - `src/photochron/models/ollama_client.py` (547 stmts / 1274 lines)
  - `src/photochron/context/analyzer.py` (352 stmts / 934 lines)
- **C** — Raise unit-test coverage **82.87% → ≥ 90%** and ratchet the enforced gate to 90.

## ⚠️ Pre-kickoff BLOCKER — continuity trigger (Lesson 21)

A lock cannot self-start a run. **No trigger artifact exists yet** in `plans/`. Before any
unattended kickoff, EITHER:
- create + prove an active trigger (`CronCreate` + a versioned `-cron.md`), OR
- run this as a **bounded, single-session** program (owner re-invokes between steps).

Until a trigger is provisioned and proven, treat this plan as **single-session bounded**;
do NOT claim unattended continuity. Surface this to the owner at kickoff.

## Operating parameters (see DECISIONS for authority)

- **Authority:** full-autonomy; council is terminal (non-blocking safeguards apply).
- **Navigation:** Serena (single-root) → **serialize ALL editing; NO worktrees** (Lesson 4).
- **No UI work** → "gate can't see UX" risk N/A (Lesson 7).
- **Integration verification impossible here** (no Ollama daemon / InsightFace weights).
  Every step MUST be fully unit-gate-verifiable. Confirmed safe: face_layer/context tests
  pass with `insightface` absent (mocked). Do NOT add an integration-dependent step (Lesson 25/D5).

## The gate (run before EVERY commit)

```bash
# 1. lint + format
uv tool run ruff check .
uv tool run ruff format --check .
# 2. types
.venv-ci/bin/mypy src/
# 3. unit gate at the CURRENT floor (baseline.json -> cov_floor)
.venv-ci/bin/pytest tests/unit -m "not integration" \
  --cov=src/photochron --cov-report=term-missing --cov-fail-under=<cov_floor> -q
```

Recreate `.venv-ci` if missing (~1 min):

```bash
uv venv .venv-ci
uv pip install --python .venv-ci/bin/python \
  pydantic pyyaml loguru typer rich pillow piexif imagehash \
  ollama onnxruntime pytest pytest-cov "mypy>=1.20,<3" types-pyyaml
uv pip install --python .venv-ci/bin/python -e . --no-deps
```

> Note: CI uses two separate venvs (type job: mypy, no onnxruntime; test job: onnxruntime,
> no mypy). The combined `.venv-ci` is a superset — benign for mypy on `src/`, but if mypy
> ever passes locally and fails in CI, suspect a transitive-dep shadowing a stub (m-1).

### Coverage floor reconciliation (was a review finding — read this)

- **Enforced in CI today = 80**, in THREE files: `.github/workflows/ci.yml`,
  `Makefile`, `pyproject.toml [tool.coverage.report] fail_under`.
- **Observed now = 82.87%.** This plan's LOCAL anti-gaming gate floor = **82**
  (stricter than CI on purpose — coverage may not fall below current observed).
- **Target = 90.** Step C5 raises ALL THREE files 80→90 (CLI `--cov-fail-under`
  overrides pyproject, so the bare `pyproject` value must also be fixed, not just CI).

### Anti-gaming (HARD — the same agent writes code AND its tests, invariant 3)

1. **Test-count floor = 413**, monotonic; set to the new real count after each step.
2. **Coverage floor = 82**, ratchets up only in C5, never down.
3. **Forbidden in any diff:** `@pytest.mark.skip`, `xfail`, deleted test files/cases,
   weakened assertions, changed prompt text.
4. **Assertion-density gate (mechanizable):** every newly added `def test_*` MUST contain
   ≥1 of `assert` / `pytest.raises` / `assert_called` / `assert_*`. Enforce per step:
   ```bash
   # fails (prints offenders) if any new test function has zero assertions
   for f in $(git diff --name-only origin/main -- 'tests/**/*.py'); do
     awk '/^def test_/{n=$0; a=0} /assert|pytest\.raises|assert_called/{a=1}
          /^def test_/ && NR>1 && pa==0 {print FILENAME": "pn" has NO assertion"}
          {pn=n; pa=a}' "$f"
   done
   ```
   (A coverage-only test that imports/executes but never asserts is the dominant gaming
   vector; counting lines as "covered" because they merely ran does not prove behavior.)
5. **SHOULD: mutation test the new pure modules** (`response_parser`, `prompts`, `scoring`,
   `errors`) with `mutmut`/`cosmic-ray`, min kill-rate ≥ 80%, when time permits.
6. **Independent review agent** (read-only, different context) approves each diff vs the
   step's stated intent before merge. Veto → BLOCKED (but verify falsifiable veto claims
   against the green gate — Lesson 13).

## Integration & status protocol

- Branch per step off live `main`: `git checkout main && git pull`.
- Status updates commit to **`main`** (this file), never the feature branch (Lesson 8).
- Merge under lock: `git fetch` → `git rebase origin/main` → re-gate →
  `git push origin <branch>` (plain push is ff-only by default — NOT `--ff-only`, Lesson 18)
  → open PR. Confirm the remote advanced via the `a..b -> <branch>` line (Lesson 17/17b).
- **Targeted `git add <paths>` only — never `git add -A`** (Lesson 9).
- Per-step attempts ≥ 3 → BLOCKED.

---

## Dependency graph

```
A1(schema) ─> A2(parser) ─> A3(prompts) ─> A4(analyzer-split) ─>
   C0(re-measure) ─> C1 ─> C2 ─> C3 ─> C4 ─> C5(ratchet gate)
```

All serial (single-root). A1–A3 all edit `ollama_client.py`; schema FIRST so parser/prompts
import from a stable `schemas.py` instead of forcing a later step to rewrite earlier imports.

---

## Step A1 — Extract result schema from `ollama_client.py` (FIRST)

- **status:** TODO   **attempts:** 0   **depends_on:** []   **model:** default
- **Context brief:** `ContextAnalysisResult` (pydantic model + 6 validators, ~line 41),
  `ModelType` (~line 34), `MAX_FILE_SIZE`, `MAX_BASE64_SIZE` sit at the TOP of the file and
  are imported widely. Extracting them first gives parser/prompts a stable import target.
- **Known importers that must NOT break** (verified): `tests/integration/test_context_layer.py`,
  `tests/fixtures/ollama_mocks.py`, `tests/unit/pipeline/stages/test_context_layer_methods.py`,
  `tests/unit/pipeline/stages/test_context_layer.py`, `context/analyzer.py`,
  `pipeline/stages/context_layer.py`.
- **Tasks:**
  1. Create `src/photochron/models/schemas.py` with the model, enum, constants.
  2. In `ollama_client.py`, `from .schemas import ContextAnalysisResult, ModelType, MAX_FILE_SIZE, MAX_BASE64_SIZE`
     and **re-export** so `from photochron.models.ollama_client import ContextAnalysisResult`
     keeps working (D7).
  3. Verify all importers via Serena `find_referencing_symbols`.
- **Verify:** gate green; all importers resolve; validator tests in `test_ollama_client.py`
  unchanged + green.
- **Exit:** schema in one place; zero importer broken; test count ≥ floor.
- **Rollback:** `git reset --hard origin/main`.

## Step A2 — Extract response parsing from `ollama_client.py`

- **status:** TODO   **attempts:** 0   **depends_on:** [A1]   **model:** default
- **Context brief:** Four helpers form a self-contained string→`ContextAnalysisResult`
  group: `_parse_llm_response` (~923), `_attempt_json_fix` (~984), `_fix_unescaped_quotes`
  (~1031), `_create_fallback_result` (~1098). They call each other but read no external
  `self` state — once moved together they become plain functions importing the schema from A1.
- **Tasks:**
  1. Create `src/photochron/models/response_parser.py`; move the four as module-level
     functions; import the schema from `.schemas`.
  2. Delegate from `OllamaClient` (thin wrappers) so the public API is unchanged.
  3. Add `tests/unit/models/test_response_parser.py`: valid JSON, quote-fix path,
     unparseable→fallback. (Recovers a chunk of ollama_client's 168 missing → feeds C.)
- **Verify:** gate green; module imports; existing `test_ollama_client.py` green unchanged.
- **Exit:** new module < 250 lines; test count > floor; behavior-preserving.
- **Rollback:** `git reset --hard origin/main`.

## Step A3 — Extract prompt templates from `ollama_client.py`

- **status:** TODO   **attempts:** 0   **depends_on:** [A2]   **model:** default
- **Context brief:** Prompt methods `_get_default_prompt`, `_get_prompt_template`,
  `get_available_prompts`, `get_prompt_template` span ~665–921 (~257 lines) and carry the
  `E501` per-file ignore (`pyproject.toml`). Prompt text is model behavior — keep it
  **byte-identical** (a changed prompt is a behavior change, forbidden).
- **Tasks:**
  1. Create `src/photochron/models/prompts.py`; move templates + lookups verbatim.
  2. Delegate from `OllamaClient`.
  3. Migrate the `E501` per-file-ignore from `ollama_client.py` to `prompts.py` in
     `pyproject.toml`; **remove** it from `ollama_client.py` if no long lines remain.
  4. Add `tests/unit/models/test_prompts.py`: available-prompt keys; each template
     non-empty + contains required placeholders.
- **Verify:** gate green; prompt diff is a pure move (no text change); ruff clean.
- **A-track exit (ollama_client):** after A1+A2+A3, `ollama_client.py` < 800 lines
  (est. ~480 after all three — the <800 win comes from the COMBINATION, not prompts alone).
- **Rollback:** `git reset --hard origin/main`.

## Step A4 — Split `analyzer.py` (errors/retry + scoring)

- **status:** TODO   **attempts:** 0   **depends_on:** [A3]   **model:** default→strong if tricky
- **Context brief:** Extract two cohesive units; keep `analyze()` orchestration in place.
  **Purity caveat (review finding):** `_calculate_overall_confidence` (~798) IS pure.
  `_validate_and_clean_result` (~716) is **NOT** pure — it reads
  `self.config.min_decade_confidence` / `min_season_confidence` / `min_event_confidence`
  (analyzer.py:731/740/750). It must take those thresholds as **explicit arguments**, e.g.
  `clean_result(result, *, min_decade, min_season, min_event)`. Dropping the config-driven
  field-clearing would be a behavior change — forbidden.
- **Tasks:**
  1. `src/photochron/context/errors.py` — `_is_model_not_found_error` (~179),
     `_extract_model_name_from_error` (~208), the `OLLAMA_EXCEPTIONS`/`_DummyOllamaException`
     shim (~30–43), and `_with_retry` (~344) as a function taking an explicit callable + args.
  2. `src/photochron/context/scoring.py` — `_calculate_overall_confidence` (pure) and
     `_validate_and_clean_result` (thresholds passed explicitly).
  3. Rewire `ContextAnalyzer` to call the extracted functions (pass `self.config.min_*`).
  4. Add `tests/unit/context/test_scoring.py` + `test_errors.py`: confidence boundaries
     0.0/1.0, low-confidence field clearing for each threshold, model-not-found classification.
- **Verify:** gate green; `analyzer.py` < 800 lines; existing analyzer/strategy/error tests
  unchanged + green.
- **Exit:** both new modules < 250 lines; analyzer < 800; test count > floor.
- **Rollback:** `git reset --hard origin/main`.

---

## Step C0 — Re-measure coverage after the split (math gate)

- **status:** TODO   **attempts:** 0   **depends_on:** [A4]   **model:** default
- **Context brief:** A1–A4 add tests and relocate code, so the baseline per-module missing
  counts are stale. Recompute before sizing C-work (Lessons 1/10/16).
- **Tasks:**
  1. Run the gate with `--cov-report=term-missing`; record the new TOTAL missing and the
     per-module missing for: `ollama_client.py`, `models/*` (new), `pipeline/__init__.py`,
     `ingestion.py`, `cli/commands.py`, `face_layer.py`, `context_layer.py`, `constraints.py`,
     `analyzer.py`, `scoring.py`, `errors.py`.
  2. Fill the math table below with real numbers; confirm a path to ≤333 missing (=90%).
  3. If 90% is NOT reachable from the listed modules, route to council: raise target ceiling
     or add modules — do NOT silently set an unreachable C5 target (review finding C-1).
- **Verify:** math table complete; planned coverage gains sum to ≥ (TOTAL_missing − 333).
- **Exit:** a feasible, numeric C-plan recorded here.

### Coverage math (baseline numbers — REPLACE in C0 after A4)

Need ≤ **333** missing of 3334 for 90%. Baseline 571 missing → cover ≥ **238**.

| Module | missing (baseline) | plan: cover | source |
|--------|-------------------:|------------:|--------|
| ollama_client.py | 168 | ~120 | A2/A3 new tests + C |
| ingestion.py | 68 | ~55 | C2 |
| pipeline/__init__.py | 63 | ~50 | C1 |
| cli/commands.py | 54 | ~40 | C3 |
| face_layer.py | 41 | ~33 | C3 |
| context_layer.py | 27 | ~17 | C4 |
| constraints.py | 17 | ~15 | C4 |
| **sum covered** | | **~330** | → ~241 residual missing → ~92.8% |

Comfortable buffer over 90%. C1–C3-only (189 max) would fall ~49 short — hence
`ollama_client.py` + `cli/commands.py` are mandatory C-scope.

## Step C1 — Cover `pipeline/__init__.py` (63% → ≥ 85%)

- **status:** TODO   **attempts:** 0   **depends_on:** [C0]   **model:** default
- **Context brief:** 406-line, **multi-class** module (`RunContext`, `PipelineStage`,
  `PipelineRegistry`, `PipelineRunner.run_pipeline` ~304) — NOT greenfield. **Extend**
  existing `tests/unit/pipeline/test_registry.py`, `test_run_context.py`, `test_stage_ledger.py`.
  Use `term-missing` for exact gaps (error/degraded paths, stage-skip, summary assembly).
  Note: `tests/test_pipeline/*` is OUTSIDE the unit gate — don't duplicate what it covers (m/12).
- **Tasks:** add cases for missing branches using in-memory store + mocked stages
  (`tests/conftest.py`). No real Ollama/InsightFace.
- **Verify:** module ≥ 85%; suite green; count > floor.
- **Exit / Rollback:** as above.

## Step C2 — Cover `pipeline/stages/ingestion.py` (69% → ≥ 85%)

- **status:** TODO   **attempts:** 0   **depends_on:** [C1]   **model:** default
- **Context brief:** Extend `test_ingestion_helpers.py` + `test_ingestion_parallel.py`.
  Gaps: EXIF edge cases, hash/dedup branches, unreadable-file + parallel-error paths.
- **Verify / Exit / Rollback:** module ≥ 85%; suite green; count > floor; reset on fail.

## Step C3 — Cover `cli/commands.py` (75%) + `face_layer.py` (75%)

- **status:** TODO   **attempts:** 0   **depends_on:** [C2]   **model:** default
- **Context brief:** `cli/commands.py` (54 missing) extend `tests/unit/cli/test_commands.py`
  (use Typer's `CliRunner`; cover error exits, flag combinations, dry-run). `face_layer.py`
  (41 missing) extend `tests/unit/pipeline/stages/test_face_layer.py` — InsightFace **mocked**
  via `patch(...InsightFaceWrapper)` (existing pattern); cover no-face/multi-face/low-conf/
  detector-unavailable. Never load real weights.
- **Verify / Exit / Rollback:** both ≥ 85%; suite green; count > floor; reset on fail.

## Step C4 — Cover `constraints.py` (78%) + `context_layer.py` (87%) + residual

- **status:** TODO   **attempts:** 0   **depends_on:** [C3]   **model:** default
- **Context brief:** `constraints.py` pure logic, extend `test_constraints.py` (easy, high
  value). `context_layer.py` extend `test_context_layer*.py`. Then confirm TOTAL ≥ 90%; if
  short, cover the next-largest residual (use C0 table).
- **Verify:** overall coverage ≥ 90% via the gate; suite green; count > floor.
- **Exit / Rollback:** ≥ 90% overall; reset on fail.

## Step C5 — Ratchet the enforced gate to 90% (all 3 files + docs)

- **status:** TODO   **attempts:** 0   **depends_on:** [C4]   **model:** default
- **Tasks:**
  1. Confirm overall ≥ 90%.
  2. Set `--cov-fail-under` 80→90 in BOTH `.github/workflows/ci.yml` AND `Makefile`,
     AND `fail_under = 80`→`90` in `pyproject.toml [tool.coverage.report]` (all three — M-2).
  3. Update `docs/testing.md` + CONTRIBUTING coverage target to 90%.
  4. Record final test count + cov as the new floors in `baseline.json`.
- **Verify:** gate green at `--cov-fail-under=90`; ci.yml YAML parses
  (`python -c "import yaml;yaml.safe_load(open('.github/workflows/ci.yml'))"`).
- **Exit:** CI enforces ≥ 90% in all three locations; docs updated.
- **Rollback:** revert the threshold commit only.

---

## Stop conditions

- Max consecutive BLOCKED = 2 → halt + notify.
- Per-step attempts ≥ 3 → BLOCKED.
- Kill-switch file `photochron-refactor-coverage.STOP` → exit.
- Wall-clock / commit budget per run: defined in the (not-yet-created) cron artifact.

## Plan-mutation log

- **Rev 2** (Phase-2 adversarial review): reordered A so schema is extracted FIRST
  (parser/prompts then import a stable module, avoiding rewrite-earlier-step churn);
  fixed A4 — `_validate_and_clean_result` is not pure, thresholds now passed explicitly;
  clarified the `<800` win is A1+A2+A3 combined, not prompts alone; reconciled the coverage
  floor (CI-enforced 80 in 3 files, local anti-gaming 82, target 90 — C5 updates all three);
  added `cli/commands.py` + `ollama_client.py` to C-scope (C1–C3 alone fall ~49 stmts short
  of 90%); added Step C0 (re-measure + math gate) so C-targets use post-A4 numbers; fixed
  C1 (406-line multi-class, extend existing tests); added a mechanizable assertion-density
  anti-gaming gate + mutation-test SHOULD; surfaced the missing continuity-trigger blocker.
- **Rev 1**: initial draft.
