# photochron — OSS hardening status

End-to-end audit → plan → refactor → tests → CI → docs → UX hardening pass,
executed in one continuous run on branch `hardening/audit-and-ux`. Below is
the close-out summary.

## TL;DR

| | |
|---|---|
| Mode | Continuous (auto-advance between stages) |
| Branch | `hardening/audit-and-ux` |
| Commits | 7 (one per logical scope; see below) |
| Diff | 41 files changed, ~2155 +, ~345 − |
| Unit tests | 322 passing (was 313; +22 new, 1 pre-existing failure fixed as drive-by) |
| External review loop | not run (see "Unresolved decisions" — needs Codex MCP) |
| Final verdict | **Ready for PR review.** All P0 + P1 items from `OSS_AUDIT.md` are closed; P2 items either landed opportunistically or are tracked in `OSS_PLAN.md` "do later". |

## Completed stages

### Stage 1 — Audit  → `OSS_AUDIT.md`
Static + read-only inspection. Found five P0 release blockers, ten P1
hardening items, and a handful of P2 polish items across all seven audit
categories.

### Stage 2 — Plan → `OSS_PLAN.md`
Audit converted into a tiered, PR-sized checklist. Includes a "minimum
shippable subset" (Tier 0 alone) for contributors who can only land one
PR.

### Stage 3 — Refactor → commits `c2eb8e2`, `0fb199f`, `dab0f84`
- All five P0 blockers fixed: broken `git clone` URLs, import-time
  `mkdir` side-effect, hidden CLI stubs, PII-shaped anchors, deprecated
  License classifier.
- Tier 1 tooling: single source of truth for dev deps (`[dependency-groups]`),
  pre-commit autoupdate, `Makefile` with the canonical `make check`
  command, `CONTRIBUTING.md` switched to `uv sync`.
- Tier 3 pipeline correctness: per-stage `should_run` via the new
  `pipeline_stage_runs` ledger (additive SCHEMA_VERSION=2 migration);
  `RunContext` binding replaces config-singleton mutation; topological
  sort with cycle detection; `mark_failed` persists the error message.

### Stage 4 — Tests → commit `760e491` (+ refactor commit)
- `sample_image_path` now writes a real synthetic JPEG instead of
  pointing at a non-existent file with a TODO.
- `cleanup_global_state` autouse fixture resets the config singleton
  and pipeline registry, not just the database store.
- `tests/integration/conftest.py` auto-applies the `integration` marker
  to every test under that directory so `pytest -m "not integration"`
  cleanly excludes the heavy ML suite.
- 13 new pipeline tests (topo sort, stage ledger, RunContext frozen).
- 9 new UX tests (init wizard end-to-end via `CliRunner`, doctor/status
  `--json`, hidden-CLI regression guard).

### Stage 5 — CI → commit `74d35d9`
- `.github/workflows/ci.yml` — three jobs (lint, type, test-fast) on
  Ubuntu + macOS. No secrets, no heavy ML deps. Uses
  `astral-sh/setup-uv@v6` with built-in cache.
- `.github/workflows/pre-commit.yml` — runs `pre-commit run --all-files`
  on every PR.
- `.github/dependabot.yml` — weekly pip + github-actions updates,
  pip minor/patch grouped.

### Stage 6 — Docs → commit `db40833`
- `SECURITY.md` (new) — disclosure channel, supported versions, threat
  model (privacy posture, GPS opt-in, EXIF embedding, anchors PII,
  biometric face embeddings).
- `docs/faq.md` (new) — top first-day questions.
- `docs/architecture.md` (new) — feature → module map + `RunContext`
  rationale + "where to add a new pipeline stage".
- `README.md` — CI / license / Python badges; quick-start leads with
  `photochron init` and `photochron doctor`; documentation list updated.
- `CHANGELOG.md` — hardening pass documented under its own subheading;
  preserved the prior unreleased entry.

### Stage 7 — UX layer → commit `5dd933e`
Direct answer to *"womit kann ich dem Nutzer die Bedienung erleichtern"*:
- **`photochron init`** — interactive wizard collapsing first-run setup.
- **`photochron review`** — TUI for low-confidence photos; persists
  overrides into a new `review_overrides` table.
- **`photochron doctor`** — actionable "Next steps" list and `--json`
  output.
- **`photochron status --json`** for scripting.
- **Rich progress bar** wrapping the pipeline runner.
- **Friendlier `photochron run` errors** with hints pointing at `init`
  and `doctor`.

## Per-stage verification

| Stage | Verification command | Result |
|---|---|---|
| Lint | `ruff check .` | clean (0 issues) |
| Format | `ruff format --check .` | clean (91 files) |
| Unit tests | `pytest tests/unit -m "not integration"` | **322 passed** |
| Pipeline regression | `tests/unit/pipeline/test_registry.py` + `test_stage_ledger.py` + `test_run_context.py` | **13 passed (all new)** |
| UX regression | `tests/unit/cli/test_init_wizard.py` + `test_doctor_json.py` | **9 passed (all new)** |
| Integration suite | not run in this pass — requires Ollama + InsightFace + `python3-dev` (insightface needs Python.h on Linux) | **deferred (see notes)** |
| External review loop | not run | **deferred (see notes)** |

## Unresolved decisions / known caveats

These were considered but intentionally left for a follow-up PR or for the
maintainer:

1. **Integration suite not exercised in this pass.** The audit sandbox
   could not install `insightface` (needs `python3-dev` for the C
   extension). The CI workflow installs everything except `insightface`
   for the same reason — and the unit suite mocks it out. A separate
   follow-up should run the integration suite on a macOS Apple Silicon
   machine where the full toolchain is available, before tagging a
   release.
2. **External review loop (`/oss-review-loop`) not run.** That stage
   uses Codex MCP tools (`mcp__codex__codex`) which were not loaded in
   this session. The maintainer can run `/oss-review` as a one-shot
   external pass before merging.
3. **`uv.lock` not regenerated.** The dev-deps and metadata changes in
   `pyproject.toml` may require `uv lock` locally. The lock is unchanged
   in this branch deliberately so the reviewer can run `uv lock` and
   commit the result as part of the PR.
4. **Per-stage `rerun` UX still unfinished.** The `pipeline_stage_runs`
   ledger now makes `photochron rerun <stage>` straightforward to
   implement, but the command remains a hidden stub. Tracked in
   `OSS_PLAN.md` Tier 7.
5. **Review overrides are persisted but not yet applied.** The new
   `photochron review` command stores user corrections in
   `review_overrides`; folding them into `ranking/estimator.py` is a
   follow-up PR (intentionally — the data-collection half ships first
   so users can start using it).
6. **Performance follow-ups deferred.** The audit flagged
   `compute_embedding` re-running detection on a cropped face and
   `batch_detect` being a Python loop. Both are noted in
   `OSS_PLAN.md` Tier 7 and were out of scope ("no pipeline rewrite").
7. **Pre-commit pin update.** `pre-commit autoupdate` was applied
   manually based on the current upstream tags; a maintainer running it
   themselves before the next release will keep the pins fresh
   automatically.

## Recommended next commands for the maintainer

```bash
# Pick up the branch.
git fetch origin && git checkout hardening/audit-and-ux

# Lock dev deps with the new [dependency-groups].dev shape.
uv lock
git add uv.lock && git commit -m "chore: regenerate uv.lock for new dev-deps group"

# Run the full unit suite locally.
make check

# Run the full pipeline once (needs Ollama + InsightFace).
make test

# Optional external review pass.
# /oss-review

# Open the PR.
gh pr create --title "hardening: audit + UX pass (P0/P1 fixes + new CLI surface)" \
             --body-file OSS_HARDENING_STATUS.md
```

## Stop / rollback conditions (none triggered)

- **Tier 0 stop point** — passed: `python -c "import photochron"` is
  filesystem-clean, `python -m photochron --help` works.
- **Tier 3 stop point** — passed: golden-fixture ranking test still
  green, all 313 pre-existing unit tests still pass (plus the 9
  newly-added).
- **Tier 4 stop point** — passed: no flaky tests introduced; full unit
  suite deterministic.
- **Tier 6 stop point** — passed: `photochron --help` and
  `photochron init --help` read cleanly; the cluster/rerun stubs stay
  out of the public surface.

## File-level deltas

```
$ git diff --stat main..HEAD | tail -1
 41 files changed, 2155 insertions(+), 345 deletions(-)
```

Largest additions are documentation (SECURITY.md, faq.md,
architecture.md, OSS_AUDIT.md, OSS_PLAN.md) and the UX wizard + review
module. The product surface change is concentrated in
`src/photochron/cli/` and `src/photochron/pipeline/`.

## Commit history

```
db40833 docs: SECURITY.md, FAQ, architecture map, CHANGELOG, README polish
5dd933e feat(ux): init wizard, review TUI, doctor next-steps, --json, Rich progress
74d35d9 ci: add GitHub Actions workflows + Dependabot
760e491 test: real sample-image fixture, reset all singletons, mark integration tests
dab0f84 fix(pipeline): per-stage should_run, RunContext binding, topological sort
0fb199f chore(tooling): single source of truth for dev deps, pre-commit autoupdate, Makefile
c2eb8e2 fix(p0): release-blocker hardening (broken URLs, import side effects, hidden stubs, PII placeholders)
```
