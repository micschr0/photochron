# PhotoChron – Full Architecture Specification

## Project Overview

PhotoChron is a local-first CLI tool that sorts digitized family photos without timestamps into chronological order using AI-based age estimation, visual context analysis, and user-provided anchor data (birthdays, events). All inference runs fully on-device. No data leaves the machine.

## Core Principles

- **Local-first**: All inference runs on-device, no external APIs
- **Non-destructive**: Only works on copies, never modifies originals
- **Confidence-aware**: Every result includes confidence scores
- **Cache-first**: Expensive inference runs only once, results cached in SQLite
- **Privacy-preserving**: Face embeddings never leave the local SQLite Feature Store

## Architecture Overview

6-stage pipeline. Each stage reads/writes to SQLite Feature Store only. Stages are independently re-runnable.

```
Ingestion → Face Layer → Context Layer → Anchor Layer → Ranking Engine → Output Layer
```

## Key Design Decisions

- **InsightFace over DeepFace**: better accuracy on low-resolution historical photos
- **Local Vision LLM (Ollama/MLX)**: biometric family data must not leave device
- **SQLite as Feature Store**: cache-first; expensive inference runs only once
- **Copies only, never originals**: non-destructive by design
- **Confidence scores everywhere**: low-confidence photos are flagged, not silently wrong

## Tech Stack

- Python 3.12+
- Typer (CLI framework)
- Rich (terminal UI)
- InsightFace (face detection + age estimation)
- ONNX Runtime with CoreML Execution Provider (Apple Silicon)
- Ollama with MLX backend (vision LLM)
- Pillow (image processing)
- piexif (EXIF manipulation)
- SQLite (feature store)
- PyYAML (configuration)

## Pipeline Stages

### Stage 1: Ingestion
- MD5 content hash per file (rename-safe cache key)
- Downsample to 1024px longest edge
- Read existing EXIF (DateTimeOriginal, Make, Model)
- Perceptual hash for near-duplicate detection
- Output: `photos` table

### Stage 2: Face Layer
- InsightFace buffalo_l via ONNX Runtime (CoreML EP)
- Detect faces, compute embeddings + age estimates
- Person identity matching via cosine similarity
- Unknown faces go to cluster pool for user assignment
- Output: `faces` table

### Stage 3: Context Layer
- Ollama (llava-next:7b via MLX), fallback moondream2
- Structured JSON prompt for decade, season, event hints, photo medium
- Anchor context passed when person birthdays known
- Output: `context` table

### Stage 4: Anchor Layer
- Load `anchors.yaml` (persons + birthdays, events, known dates)
- Create `AnchorMap` and `ConstraintSet`
- Validate constraints (no contradicting hard constraints)
- Output: in-memory `ConstraintSet` passed to Ranking Engine

### Stage 5: Ranking Engine
- Weighted combination of signals:
  - Face age estimate (45%)
  - LLM decade estimate (30%)
  - Photo medium prior (10%)
  - EXIF date (100% when present, overrides all)
- Apply constraints (hard first, then soft)
- Pairwise LLM comparison for ambiguous pairs (max 500 pairs)
- Topological sort → final `sort_rank`
- Output: `rankings` table

### Stage 6: Output Layer
Two output modes (both active on full run):

1. **Renamed copies**: `{sort_rank:04d}_{estimated_year}-est_{original_name}.jpg`
2. **EXIF-enriched copies**: Original name preserved, EXIF fields added:
   - `DateTimeOriginal`: Estimated date
   - `ImageDescription`: Human-readable summary
   - `UserComment`: Full JSON result blob

Additional outputs:
- `photochron_report.json`
- `photochron_timeline.csv`

## Data Storage

### SQLite Feature Store (`.photochron/cache.db`)
- `photos`: Photo metadata, content hashes, thumbnails
- `faces`: Face detections, embeddings, age estimates, person assignments
- `persons`: Known persons from anchors.yaml + user-assigned clusters
- `context`: LLM analysis results (decade, season, etc.)
- `rankings`: Final chronological ranking with confidence scores
- `pipeline_runs`: Run history with model versions and config hashes

### anchors.yaml Format
```yaml
persons:
  - id: person_mama
    name: "Mama"
    birthday: "1983-03-15"

events:
  - name: "Umzug nach Osnabrück"
    date: "1991-08-01"
    type: hard
    photos_after:
      - "IMG_042.jpg"

known_dates:
  - file: "Weihnachten_gross.jpg"
    month: 12
    type: soft
```

## Hardware & Performance

### Apple Silicon Execution Model
- M3 MacBook Air with 16 GB Unified Memory
- No VRAM copy overhead (CPU, GPU, Neural Engine share memory)
- Models loaded into RAM directly accessible by ANE and GPU

### Model Performance
- **InsightFace buffalo_l**: 100–300ms/image, ~400 MB memory
- **llava-next:7b (Ollama/MLX)**: 2–5s/image, ~4.5 GB memory
- **moondream2 (fallback)**: 0.5–1.5s/image, ~1.5 GB memory

### Memory Budget (16 GB system)
- InsightFace: ~400 MB
- llava-next:7b: ~4.5 GB
- SQLite + Python: ~300 MB
- macOS overhead: ~3 GB
- **Total**: ~8.2 GB
- **Free headroom**: ~7.8 GB

## Development Guidelines

### Testing
```bash
pytest -v tests/
```

### Linting & Type Checking
```bash
ruff check . && mypy src/
```

### Installation
```bash
pip install -e ".[dev]"
```

## Boundaries & Constraints

See `agent_docs/boundaries.md` for detailed MUST, MAY, and MUST NOT rules.

Key hard stops:
- Never modify files in input directory (read-only)
- Never call external HTTP APIs for image analysis
- Never produce results without confidence scores
- Never export face embeddings outside local SQLite
- Never overwrite existing output files silently

## CLI Reference

See `agent_docs/commands.md` for complete CLI specification.

Quick commands:
```bash
# Full pipeline
python -m photochron run --input ./photos --output ./photochron_output

# Dry run
python -m photochron run --input ./photos --dry-run

# Face clustering
python -m photochron cluster --input ./photos

# Re-run single stage
python -m photochron rerun --stage ranking

# Show cache stats
python -m photochron status
```