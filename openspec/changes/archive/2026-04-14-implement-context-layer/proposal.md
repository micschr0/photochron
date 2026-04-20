## Why

The PhotoChron pipeline currently has two completed stages (Ingestion and Face Layer) but cannot analyze visual context for chronological dating. The Context Layer (Stage 3) is critical for extracting decade estimates, seasons, event hints, and photo medium information using a vision LLM. Without this stage, the pipeline lacks the visual context signals needed for accurate chronological ranking, making the system incomplete and unable to produce meaningful chronological outputs.

## What Changes

- **Add Ollama/MLX dependency** for local vision LLM inference (llava-next:7b model with MLX backend)
- **Implement ContextLayerStage** that processes downsampled images from previous stages
- **Integrate vision LLM** to analyze photo context and extract structured information:
  - Decade estimate with confidence (e.g., "1985-1990", 0.75 confidence)
  - Season (spring, summer, autumn, winter)
  - Event hints (wedding, birthday, graduation, etc.)
  - Photo medium (print_scan, digital, polaroid, etc.)
- **Implement structured JSON prompting** to ensure consistent LLM outputs
- **Add retry logic** for JSON parsing failures with fallback strategies
- **Store results** in the existing `context` table with confidence scores
- **Add configuration options** for model selection, prompt templates, and retry behavior
- **Integrate with pipeline progress tracking** and dependency system (depends on `face_layer` stage)

## Capabilities

### New Capabilities
- `context-analysis`: Analyze photo visual context using vision LLM to extract decade estimates, seasons, event hints, and photo medium information with confidence scores. This capability covers the entire context analysis workflow required by Stage 3 of the PhotoChron pipeline.

### Modified Capabilities
<!-- No existing capabilities are being modified at the requirement level. -->

## Impact

- **Dependencies**: Adds `ollama` Python client library and requires Ollama runtime with MLX backend for Apple Silicon optimization. This maintains the local-only principle while adding significant AI capability.
- **Code**: New implementation in `src/photochron/pipeline/stages/context_layer.py` replacing the current placeholder. Additional helper modules for LLM integration, prompt engineering, and JSON parsing.
- **Database**: Uses the existing `context` table schema; no schema changes required.
- **Configuration**: New `models` section in `config.yaml` already exists with `ollama_model` and `fallback_model` settings. May add context-specific configuration for prompt templates and retry behavior.
- **Performance**: Vision LLM inference is computationally intensive (~2-5 seconds per image on 7B model with MLX). Will leverage downsampled images (max 1024px) and implement batch processing where possible.
- **Testing**: Requires unit tests with mock LLM responses, integration tests with sample photos, and verification of confidence-score integrity and JSON parsing robustness.