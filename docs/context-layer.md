# Context Layer

The Context Layer (Stage 3 of the photochron pipeline) analyzes each photo with a local vision LLM through [Ollama](https://ollama.com) and stores structured context data — decade estimate, season, event hint, photo medium — plus per-field confidence scores in the `context` table. All inference runs on-device; images never leave the machine.

## Architecture

```
Photos (without context rows)
        │
        ▼
ContextLayerStage.run()
        │
        ├── _validate_configuration()          # health check, model availability
        │       └── ContextAnalyzer.health_check()
        │
        ├── _get_photos_without_context()      # SQL: photos LEFT JOIN context
        │
        └── for each batch:
              ├── _check_memory_before_batch() # psutil, threshold-based gating
              ├── ContextAnalyzer.analyze()    # OllamaClient → structured JSON
              └── _store_context_result()      # context table insert
```

### Components

- **`ContextLayerStage`** (`src/photochron/pipeline/stages/context_layer.py`) — Pipeline stage that orchestrates photo iteration, batching, memory gating, and DB writes.
- **`ContextAnalyzer`** (`src/photochron/context/analyzer.py`) — Higher-level analyzer that implements analysis strategies (DEFAULT, AGGRESSIVE, CONSERVATIVE, FAST), retries, and fallback model handling.
- **`OllamaClient`** (`src/photochron/models/ollama_client.py`) — Thin wrapper around the Ollama Python client with JSON-schema prompting, timeouts, and retry logic.
- **`ContextCreate`** (`src/photochron/models/`) — Pydantic model describing the row written to the `context` table.

### Data Flow

| Input | Source | Notes |
| :-- | :-- | :-- |
| Downsampled image | `{cache_dir}/thumbs/<hash>.jpg` | Falls back to original file path if missing |
| Configuration | `config.yaml` → `context:` | Validated via `ConfigContext` Pydantic model |
| Models | Local Ollama server | Primary + fallback, discovered at init |

| Output | Destination | Notes |
| :-- | :-- | :-- |
| Structured context | `context` table | One row per photo, unique on `photo_id` |
| Minimal context (on failure) | `context` table | `uncertainty_flag=True`, confidences set to `0.0` |

## Configuration

All options live under `context:` in `config.yaml` (Pydantic model: `ConfigContext` in `src/photochron/config/models.py`).

| Key | Default | Description |
| :-- | :-- | :-- |
| `ollama_host` | `http://localhost:11434` | Base URL of the local Ollama server |
| `ollama_timeout` | `300` | Request timeout in seconds |
| `max_retries` | `3` | Max retry attempts on transient LLM failures |
| `retry_delay` | `2.0` | Seconds between retries (with jitter) |
| `primary_model` | `llava-next:7b` | Preferred vision LLM |
| `fallback_model` | `moondream2` | Used when primary is unavailable or fails |
| `batch_size` | `1` | Photos per batch (sequential inside a batch) |
| `min_decade_confidence` | `0.3` | Decade results below this are flagged uncertain |
| `min_season_confidence` | `0.4` | Season results below this are flagged uncertain |
| `use_fallback_on_failure` | `true` | Enable fallback strategies on analyzer failure |
| `store_minimal_on_complete_failure` | `true` | Insert a minimal row rather than silently drop |
| `memory_warning_threshold_mb` | `100` | Warn if available RAM drops below this |
| `memory_critical_threshold_mb` | `50` | Skip batch and wait when available RAM drops below this |
| `memory_retry_delay_seconds` | `30` | Wait this long after a critical memory event |

A validator enforces `memory_critical_threshold_mb < memory_warning_threshold_mb`.

See `examples/context-config-example.yaml` for a fully commented example.

## Usage Patterns

### Typical full run
```bash
python -m photochron run --input ./photos --output ./photochron_output
```
The context layer runs after the face layer and before the anchor layer.

### Run only the context layer
```bash
python -m photochron run --stages context_layer
```

### Re-run without re-doing earlier stages
```bash
python -m photochron rerun --stage context_layer
```
Useful after changing models, prompts, or confidence thresholds.

### Dry run
```bash
python -m photochron run --dry-run
```
No rows are written; useful for inspecting logs and progress reporting.

## Progress Reporting

The stage emits three levels of progress:

- **Batch start**: `Processing batch {n}/{total} ({pct:.1f}%)`
- **Every 10 photos**: `Processed {p}/{total} photos ({pct:.1f}%)`
- **Completion**: `Context layer stage completed. Processed {p}/{total} photos ({pct:.1f}%), failed: {f}`

All percentages use 1 decimal place. When `total_photos == 0`, percentage is omitted to avoid division by zero.

## Health and Degraded Mode

At initialization `ContextLayerStage._validate_configuration()`:

1. Calls `ContextAnalyzer.health_check()`.
2. Inspects server availability and per-model availability.
3. Sets `_available_models["primary" | "fallback"]`.
4. Rewrites `analyzer.config.model_priority` to only include available models.
5. If neither model is available → enters **degraded mode**.

In degraded mode, `run()` logs a warning, marks the stage complete with `photos_processed=0`, and exits cleanly. Rerunning after fixing Ollama will re-check health.

Access the live status via:
```python
stage = ContextLayerStage()
stage.health_status
# {"is_healthy": True, "degraded_mode": False,
#  "available_models": {"primary": True, "fallback": True}}
```

## Memory Safety

Before each batch, `_check_memory_before_batch()` uses `psutil.virtual_memory().available`:

- `> memory_warning_threshold_mb` → **ok**, continue.
- `< memory_warning_threshold_mb` → **warning**, log but continue.
- `< memory_critical_threshold_mb` → **critical**, skip batch and `sleep(memory_retry_delay_seconds)`.
- `psutil` not installed → **unknown**, log once and continue (never blocks).

This protects long runs on constrained machines where the 7B LLM plus other processes can transiently push the system into swap.

## Error Handling

The analyzer implements layered fallbacks (see `ContextAnalyzer._with_retry` and the `FallbackStrategy` enum):

1. **Retry** on transient Ollama exceptions (`RequestError`, `ResponseError`, timeouts).
2. **Model fallback** from primary to fallback when primary fails or is unavailable.
3. **Strategy fallback** (simple / uncertainty / multi-hypothesis prompts) when structured parsing fails.
4. **Minimal store** when everything fails, if `store_minimal_on_complete_failure=true`.

Per-photo failures are caught inside the batch loop and incremented into the `failed` counter — a single bad image never aborts the stage.

## Output Contract

Example `context` row (JSON view):

```json
{
  "photo_id": 42,
  "decade": "1985-1990",
  "decade_confidence": 0.75,
  "season": "summer",
  "season_confidence": 0.6,
  "event_hint": null,
  "event_confidence": null,
  "photo_medium": "print_scan",
  "photo_medium_confidence": 0.9,
  "visual_evidence": "warm color cast, square format, visible film grain",
  "alternative_decades": ["1980-1985"],
  "uncertainty_flag": false,
  "hypothesis_notes": null,
  "raw_json": "{...}"
}
```

Downstream stages (Ranking Engine) read `decade`, `decade_confidence`, `photo_medium`, and `alternative_decades`.

## Troubleshooting

### "Context layer stage is in degraded mode"
- Is the Ollama server running? `curl http://localhost:11434/api/tags`
- Are the configured models pulled? `ollama list`
- See `docs/ollama-setup.md` for installation and model pulls.

### "Neither primary model nor fallback model are available"
- Pull at least one: `ollama pull llava-next:7b` or `ollama pull moondream2`.
- Verify the names in `config.yaml` match Ollama's model tags exactly.

### "Memory critically low"
- Lower `batch_size` to `1`.
- Raise `memory_critical_threshold_mb` if you want more conservative behavior.
- Switch `primary_model` to `moondream2` for lighter memory footprint.

### "Failed to process photo <id>"
- Verify the downsample file exists at `photo.downsample_path`.
- Rerun ingestion: `python -m photochron rerun --stage ingestion`.
- Check log level: set `LOGLEVEL=DEBUG` for per-photo analyzer output.

### Low confidence scores everywhere
- Try `primary_model: llava-next:7b` over `moondream2` — larger model, better priors.
- Lower `min_decade_confidence` / `min_season_confidence` if you accept more uncertainty.
- Inspect `visual_evidence` on low-confidence rows to understand what the model "saw".

## Testing

- **Unit**: `tests/unit/context/test_analyzer.py`, `tests/unit/models/test_ollama_client.py`
- **Integration**: `tests/integration/test_context_layer.py`
- **Error handling**: `tests/unit/test_error_handling.py`
- **Confidence propagation**: `tests/unit/test_confidence_validation.py`
- **DB integration**: `tests/unit/test_database_integration.py`

Run the full suite:
```bash
pytest -v tests/
```

Run only context-related tests:
```bash
pytest -v tests/unit/context/ tests/unit/models/ tests/integration/test_context_layer.py
```
