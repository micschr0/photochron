## CLAUDE.md

# PhotoChron – CLAUDE.md

What is this project?
PhotoChron is a local-first CLI tool that sorts digitized family photos
without timestamps into chronological order using AI-based age estimation,
visual context analysis, and user-provided anchor data (birthdays, events).
All inference runs fully on-device. No data leaves the machine.

Quick Commands

# Full pipeline run

python -m photochron run --input ./photos --output ./photochron_output

# Dry run (no file writes)

python -m photochron run --input ./photos --dry-run

# Face clustering + person assignment (one-time setup)

python -m photochron cluster --input ./photos

# Re-run single stage without re-inference

python -m photochron rerun --stage ranking

# Show cache stats

python -m photochron status

# Tests

pytest -v tests/

# Lint

ruff check . && mypy src/

Boundaries (read before any change)
See `agent_docs/boundaries.md` – this is mandatory reading.

Short version:

- 🚫 NEVER touch files in the input directory
- 🚫 NEVER call external HTTP APIs for image analysis
- 🚫 NEVER skip writing a confidence score on any result
- ✅ ALWAYS write to copies in the output directory
- ⚠️ ASK before adding new dependencies or changing config schema

Architecture
6-stage pipeline. Each stage reads/writes to SQLite Feature Store only.
Stages are independently re-runnable.

Ingestion → Face Layer → Context Layer → Anchor Layer → Ranking Engine → Output Layer

Full details: `agent_docs/pipeline.md`

### Ingestion Stage Details

The Ingestion stage (Stage 1) reads image files from the configured `input_dir` and performs:

- **File scanning**: Detects image files with extensions `.jpg`, `.jpeg`, `.png`, `.heic`, `.heif`, `.cr2`, `.nef`, `.arw`, `.dng`
- **Perceptual hashing**: Computes pHash for duplicate detection and file identification
- **Downsampling**: Creates resized versions (max 1024px longest edge) stored in cache
- **EXIF extraction**: Extracts timestamp, camera make/model, GPS coordinates (if enabled)
- **Database storage**: Stores metadata in `photos` table with content hash as unique key

**Configuration options** (`config.yaml` → `ingestion:`):
- `max_downsample_size`: Maximum dimension for downsampled images (default: 1024)
- `supported_formats`: List of file extensions to process
- `skip_duplicates`: Skip files with same content hash (default: true)
- `extract_gps`: Extract GPS coordinates from EXIF (default: true)
- `fallback_timestamp`: Source when EXIF missing: `file_mtime`, `file_ctime`, `user_default`

**Non‑destructive operation**: Original files are never modified; downsampled copies are stored in `{cache_dir}/downsampled/`.

### Face Layer Details

The Face layer (Stage 2) detects faces in downsampled photos, computes 512‑dimensional embeddings, estimates ages, and matches faces to known persons (or marks them unknown). All results are stored in the `faces` table with confidence scores.

- **Face detection**: Uses InsightFace with ONNX runtime (CPU or GPU) to detect faces with confidence scores above `detection_threshold`.
- **Embedding extraction**: Computes normalized 512‑dim face embeddings for biometric matching.
- **Age estimation**: Estimates age in years with configurable standard deviation scaling (`age_confidence_scale`).
- **Person matching**: Compares face embeddings against known persons (from `persons` table) using cosine similarity; matches above `matching_threshold` are assigned `person_id`.
- **Database storage**: Stores each face with bounding box, confidence, age estimate, age std dev, embedding (BLOB), and optional person ID.
- **Batch processing**: Photos are processed in configurable batch sizes (currently sequential due to InsightFace limitations).

**Configuration options** (`config.yaml` → `face:`):
- `model_name`: InsightFace model name (`buffalo_l`, `buffalo_s`); default: `buffalo_l`
- `detection_threshold`: Minimum confidence for face detection (0.0–1.0); default: `0.5`
- `matching_threshold`: Cosine similarity threshold for person matching (0.0–1.0); default: `0.6`
- `age_confidence_scale`: Multiplier for age standard deviation; default: `0.1`
- `use_gpu`: Use GPU acceleration if available (requires ONNX Runtime with CUDA); default: `false`
- `batch_size`: Number of photos to process in a batch (future optimization); default: `1`

**Non‑destructive operation**: Only reads downsampled images from cache; never modifies original files.

### Context Layer Details

The Context layer (Stage 3) analyzes each downsampled photo with a local vision LLM (via Ollama/MLX) to estimate the decade it was taken, detect seasonal hints, identify events, and classify the photo medium (e.g. print scan, Polaroid, digital). Results are written to the `context` table with per-field confidence scores.

- **Vision LLM analysis**: Sends downsampled images to Ollama with a structured JSON prompt contract (never free text).
- **Decade estimate**: Returns a decade bucket (e.g. `1985-1990`) plus `decade_confidence` and optional alternative decades.
- **Season / event / medium**: Extracts season hints, event hints, and photo medium with individual confidence scores.
- **Health check & graceful degradation**: At init, checks Ollama reachability and model availability; enters degraded mode (skips stage) if neither primary nor fallback model is usable.
- **Batch processing**: Photos are processed in configurable batch sizes; progress is logged per batch and every 10 photos with percentage completion.
- **Memory guards**: Before each batch, available system memory is checked; batches are skipped and retried if memory falls below the critical threshold.
- **Retry & fallback**: Transient LLM failures are retried with configurable delay; analyzer falls back from primary to fallback model automatically.
- **Minimal-store on failure**: When analysis fails completely, a minimal record with `uncertainty_flag=True` is stored (configurable) so the photo is not silently dropped.

**Typical usage**:
```bash
# Run only the context layer for new photos
python -m photochron run --stages context_layer

# Re-run the context layer from scratch
python -m photochron rerun --stage context_layer
```

**Configuration options** (`config.yaml` → `context:`):
- `ollama_host`: Ollama server URL; default: `http://localhost:11434`
- `ollama_timeout`: Request timeout in seconds; default: `300`
- `max_retries`: Max retry attempts on LLM failure; default: `3`
- `retry_delay`: Delay between retries in seconds; default: `2.0`
- `primary_model`: Primary vision LLM; default: `llava-next:7b`
- `fallback_model`: Fallback vision LLM; default: `moondream2`
- `batch_size`: Number of photos per batch; default: `1`
- `min_decade_confidence`: Minimum confidence for decade results; default: `0.3`
- `min_season_confidence`: Minimum confidence for season results; default: `0.4`
- `use_fallback_on_failure`: Enable fallback analysis strategies; default: `true`
- `store_minimal_on_complete_failure`: Store minimal row when analysis fails; default: `true`
- `memory_warning_threshold_mb`: Warning threshold for low memory; default: `100`
- `memory_critical_threshold_mb`: Critical threshold to skip batch; default: `50`
- `memory_retry_delay_seconds`: Wait time before retrying after critical memory; default: `30`

See `docs/context-layer.md` for architecture details and troubleshooting, and `docs/ollama-setup.md` for installing Ollama and pulling the required models.

**Non‑destructive operation**: Only reads downsampled images from cache; never writes to original files or input directory.

Key Design Decisions (WHY)

- **InsightFace over DeepFace**: better accuracy on low-resolution historical photos
- **Local Vision LLM (Ollama/MLX)**: biometric family data must not leave device
- **SQLite as Feature Store**: cache-first; expensive inference runs only once
- **Copies only, never originals**: non-destructive by design
- **Confidence scores everywhere**: low-confidence photos are flagged, not silently wrong

Sub-Documents

| File | When to read |
| :-- | :-- |
| `agent_docs/boundaries.md` | Before ANY change – hard rules |
| `agent_docs/pipeline.md` | When working on pipeline stages |
| `agent_docs/models.md` | When changing/adding AI models |
| `agent_docs/data_formats.md` | When touching DB schema, EXIF, YAML config |
| `agent_docs/commands.md` | Full CLI spec + expected outputs |
| `agent_docs/SPEC.md` | Full architecture spec (reference document) |

Tech Stack (short)
Python 3.12+, Typer, Rich, InsightFace, ONNX Runtime (CoreML EP),
Ollama (MLX backend), Pillow, piexif, SQLite, PyYAML.