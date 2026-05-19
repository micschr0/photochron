# Architecture map

A quick "where does X live" reference for new contributors. For the deep
dive on each pipeline stage, see [pipeline.md](./pipeline.md); for
configuration semantics, [configuration.md](./configuration.md).

## Feature → module map

| Feature | Public entrypoint | Module | Notes |
|---|---|---|---|
| Run the full pipeline | `photochron run` | `cli/commands.py:run` → `pipeline/__init__.py:PipelineRunner` | The runner binds a `RunContext` to every stage; stages read inputs from `self.context`. |
| Interactive first-time setup | `photochron init` | `cli/commands.py:init` → `cli/wizard.py` | Pure-function wizard; no external services. |
| Health check + remediation hints | `photochron doctor` | `cli/commands.py:doctor` | Reads only; emits a numbered "Next steps" list. `--json` for scripting. |
| Cache + run inspection | `photochron status` | `cli/commands.py:status` | Reads only; `--json` for scripting. |
| Manual override of low-confidence photos | `photochron review` | `cli/commands.py:review` → `review/__init__.py` | Persists into `review_overrides`. |
| Configuration loading | (lib) | `config/__init__.py`, `config/models.py` | Pydantic v2 models with `extra="forbid"`. Env-var overrides via `PHOTOCHRON_<SECTION>_<KEY>`. |
| Centralised logging | (lib) | `logging_config.py` | Loguru with an `InterceptHandler` for stdlib loggers (httpx, insightface, onnxruntime). |
| SQLite feature store | (lib) | `store/__init__.py`, `store/schema.py`, `store/queries.py` | Thread-local connections, WAL mode, FK on. Schema migrations are additive. |
| Pipeline orchestration | (lib) | `pipeline/__init__.py` | `PipelineRunner`, `PipelineRegistry` (topo sort + cycle check), `RunContext` (frozen), `PipelineStage` base. |
| Stage 1 — ingestion (hash + EXIF + downsample) | (stage) | `pipeline/stages/ingestion.py` | Parallel via `ThreadPoolExecutor`. |
| Stage 2 — face detection / age | (stage) | `pipeline/stages/face_layer.py` → `face/insightface_wrapper.py` | Backend resolution via `resolve_providers` (CPU / CUDA / CoreML). |
| Stage 3 — context (vision LLM) | (stage) | `pipeline/stages/context_layer.py` → `context/analyzer.py`, `models/ollama_client.py` | Multi-strategy with retries + degraded mode. |
| Stage 4 — anchor application | (stage) | `pipeline/stages/anchor_layer.py` → `anchor/loader.py`, `anchor/models.py` | Loads `anchors.yaml`; validates hard-constraint contradictions. |
| Stage 5 — ranking | (stage) | `pipeline/stages/ranking_engine.py` → `ranking/estimator.py`, `ranking/constraints.py` | Signal fusion + constraint propagation. |
| Stage 6 — output (renames + EXIF + report) | (stage) | `pipeline/stages/output_layer.py` → `output/*.py` | Reads `RunContext.output_dir` + `dry_run`. |

## Data shape

```
┌──────────────────────────────────────────────────────┐
│  SQLite Feature Store  (.photochron/cache.db)        │
├──────────────────────────────────────────────────────┤
│ photos                — one row per ingested image   │
│ faces                 — N per photo                  │
│ persons               — from anchors.yaml + clusters │
│ context               — one per photo, LLM result    │
│ anchor_constraints    — one per run                  │
│ rankings              — one per photo (final)        │
│ pipeline_runs         — one per `photochron run`     │
│ pipeline_stage_runs   — one per (run, stage)         │
│ review_overrides      — one per manually-corrected   │
│                          photo (lazy table)          │
└──────────────────────────────────────────────────────┘
```

## Why `RunContext` and not "just pass kwargs"?

Earlier code mutated the global `Config` singleton (`config.input_dir =
...`) on every run. That poisoned test isolation, made concurrent runs
unsafe, and obscured what was actually a per-run input. The refactor
introduced a frozen `RunContext` dataclass that the runner binds to each
stage via `stage.bind_context(ctx)` before calling `run()`. Stages that
need `input_dir`, `output_dir`, or `dry_run` read them off
`self.context`; everything else still flows through the read-only
`Config`.

## Where to add a new pipeline stage

1. Create `src/photochron/pipeline/stages/<name>.py` with a class
   inheriting `PipelineStage`. Implement `name`, `dependencies`, and
   `run(run_id, config_hash)`.
2. Decorate the class with `@register_stage`.
3. Import it from `pipeline/stages/__init__.py` so the side-effect
   registration fires when the runner imports the package.
4. Add unit tests under `tests/unit/pipeline/stages/`. Run `make
   test-fast` to verify the topological sort still orders correctly.

The runner's `validate_dependencies()` and `get_dependency_order()`
will surface any wiring mistake at startup.
