## Context

The PhotoChron pipeline currently has completed Stages 1 (Ingestion) and 2 (Face Layer), but Stage 3 (Context Layer) is a placeholder. The Context Layer is responsible for analyzing visual context in photos using a vision LLM to extract chronological signals: decade estimates, seasons, event hints, and photo medium information. This stage depends on the Face Layer for person context and produces data for the Ranking Engine (Stage 5).

Key constraints from the project:
- **Local-only principle**: No external API calls for image analysis
- **Non-destructive operation**: Original files never modified
- **Confidence scores everywhere**: Every result must include a confidence score (0.0-1.0)
- **SQLite as feature store**: All inter-stage communication through database
- **Apple Silicon optimization**: MLX backend for Ollama on Apple Silicon

## Goals / Non-Goals

**Goals:**
1. Implement functional ContextLayerStage that analyzes photos using local vision LLM
2. Extract structured information: decade estimate, season, event hints, photo medium
3. Ensure robust JSON parsing with retry logic for LLM outputs
4. Maintain local-only principle with Ollama/MLX integration
5. Propagate confidence scores from LLM analysis to database
6. Integrate with existing pipeline framework and progress tracking

**Non-Goals:**
1. Training custom vision models (use pre-trained Ollama models)
2. Real-time processing (accept ~2-5 seconds per image latency)
3. Support for non-Apple Silicon platforms (MLX backend is Apple Silicon specific)
4. Complex prompt engineering beyond basic structured JSON output
5. Integration with external vision APIs (maintain local-only)

## Decisions

### 1. Ollama Integration Approach
**Decision**: Use `ollama` Python client library with direct API calls to local Ollama server
**Rationale**: 
- Ollama provides a simple HTTP API for model inference
- MLX backend offers Apple Silicon optimization
- Avoids complex model loading/unloading in Python process
- Allows model management (pull, list, etc.) via Ollama CLI
**Alternative considered**: Direct PyTorch/MLX integration - rejected due to complexity and model format compatibility issues

### 2. Model Selection Strategy
**Decision**: Use `llava-next:7b` as primary model with `moondream2` as fallback
**Rationale**:
- `llava-next:7b` provides good balance of accuracy and speed for vision tasks
- `moondream2` is smaller and faster for basic scene understanding
- Fallback ensures robustness if primary model fails
- Both models available via Ollama model registry
**Alternative considered**: Single model only - rejected for robustness

### 3. Prompt Engineering Approach
**Decision**: Structured JSON prompting with schema validation
**Rationale**:
- Ensures consistent output format for parsing
- Reduces LLM hallucination by constraining output space
- Enables automatic validation against expected schema
- Allows retry logic when JSON parsing fails
**Alternative considered**: Free-text output with regex parsing - rejected due to inconsistency risks

### 4. Error Handling Strategy
**Decision**: Two-tier retry with fallback to minimal output
**Rationale**:
1. First attempt: Full analysis with primary model
2. Second attempt: Simplified prompt with same model
3. Fallback: Basic analysis with fallback model
4. Final fallback: Store minimal data with low confidence flag
**Alternative considered**: Single attempt - rejected for production robustness

### 5. Database Integration Pattern
**Decision**: Follow existing pattern from Ingestion and Face Layer stages
**Rationale**:
- Consistency with established pipeline patterns
- Uses existing `context` table schema
- Implements same progress tracking and dependency checking
- Maintains transaction integrity for batch processing
**Alternative considered**: New database pattern - rejected for consistency

### 6. Configuration Approach
**Decision**: Extend existing `config.yaml` with context-specific settings
**Rationale**:
- Maintains single configuration file approach
- Allows environment variable overrides (PHOTOCHRON_CONTEXT_*)
- Consistent with other stages' configuration patterns
**Alternative considered**: Separate config file - rejected for simplicity

## Risks / Trade-offs

**Risk 1**: LLM inference latency (2-5 seconds per image) → **Mitigation**: Use downsampled images (1024px), implement progress reporting, allow user to skip context analysis

**Risk 2**: JSON parsing failures from LLM outputs → **Mitigation**: Implement retry logic, schema validation, fallback to simplified prompts

**Risk 3**: Ollama server dependency and model availability → **Mitigation**: Check Ollama status at startup, provide clear error messages, offer fallback model

**Risk 4**: Memory usage with large photo collections → **Mitigation**: Process photos in configurable batches, implement memory monitoring

**Risk 5**: Inconsistent decade estimates across similar photos → **Mitigation**: Store confidence scores, flag low-confidence results for review, use in ranking engine with appropriate weights

**Trade-off**: Accuracy vs. Speed → **Decision**: Prioritize accuracy for decade estimates (critical for chronology) but accept slower processing

**Trade-off**: Complexity vs. Robustness → **Decision**: Implement retry logic and fallbacks for robustness, accepting increased code complexity