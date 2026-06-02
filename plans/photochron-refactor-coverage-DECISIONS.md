# Decisions — photochron-refactor-coverage

Council rulings + standing defaults for the autonomous run. Owner may overrule any entry.

## Standing decisions (set at kickoff)

| ID | Decision | Value | Rationale |
|----|----------|-------|-----------|
| D1 | Decision authority (invariant 5) | **full-autonomy; council is terminal** | Owner requested hands-off completion. Non-blocking safeguards apply: every destructive/irreversible step gets a mandatory irreversibility review, skeptic-veto → more rounds or a safer variant (never a human), and a test proving safe behavior. |
| D2 | Editing concurrency | **serialized, no worktrees** | Serena single-root backend; parallel edits would corrupt the checkout (Lesson 4). |
| D3 | UI/visual merge mode | **N/A** | This program has no UI/visual work; "gate can't see UX" risk does not apply (Lesson 7). |
| D4 | Behavior preservation | **mandatory for all A steps** | Splits move code only; no public-API or prompt-text change. A prompt change = a model-behavior change and is out of scope. |
| D5 | Verification ceiling | **unit gate only** | Integration suite needs Ollama daemon + InsightFace weights, absent in the run environment (Lesson 25). No step may depend on integration verification. |
| D6 | Anti-gaming floor | test-count ≥ 413; coverage ratchets 82 → 90, never down | The agent that writes code may not weaken the tests grading it (invariant 3). |
| D7 | Re-export on schema move (A3) | **keep old import paths working** | Avoid breaking importers/tests; reduces blast radius of the move. |

## Open questions routed to council

- (none yet)

## Audit trail

- Kickoff: plan + decisions + baseline authored on branch `plan/refactor-coverage`,
  pending adversarial review (Phase 2) before any execution step.
