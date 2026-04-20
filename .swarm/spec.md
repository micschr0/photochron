# Specification: Context Layer Implementation

## Feature Description
Implement Stage 3 (Context Layer) of the PhotoChron pipeline to analyze photo visual context using a vision LLM. The context layer extracts decade estimates, seasons, event hints, and photo medium information with confidence scores for chronological dating.

## User Scenarios

### Scenario 1: Single photo context analysis
- **Given** a downsampled photo from the ingestion stage
- **When** the context layer analyzes the photo
- **Then** the system returns structured JSON with decade estimate, season, event hints, photo medium, and confidence scores

### Scenario 2: Batch processing of photos
- **Given** a collection of photos without context data
- **When** the context layer processes the batch
- **Then** all photos are analyzed and results stored in the database with progress tracking

### Scenario 3: Error handling and fallback
- **Given** a photo that causes LLM inference to fail
- **When** the context layer encounters the failure
- **Then** the system implements retry logic, falls back to simpler models, and stores minimal data if all attempts fail

## Functional Requirements

### FR-001: Vision LLM Integration
The system MUST integrate with Ollama vision LLM (llava-next:7b with MLX backend) for local image analysis. The integration MUST include health checks, model validation, and connection handling.

### FR-002: Structured Context Analysis
The system MUST analyze photos to extract: decade estimate with confidence (0.0-1.0), season (spring/summer/autumn/winter) with confidence, event hints with confidence, and photo medium (digital/print_scan/polaroid/film_negative/unknown) with confidence.

### FR-003: JSON Output Validation
The system MUST use structured JSON prompting and validate LLM outputs against a defined schema before storage. Invalid JSON MUST trigger retry logic with simplified prompts.

### FR-004: Retry and Fallback Logic
The system MUST implement retry logic for LLM failures, JSON parsing errors, and timeouts. The system MUST support model fallback (llava-next:7b → moondream2) and MUST prevent infinite retry loops.

### FR-005: Database Integration
The system MUST store context analysis results in the existing `context` table with transaction support. The system MUST query for photos without context data and process them in configurable batches.

### FR-006: Pipeline Integration
The context layer MUST implement the `PipelineStage` abstract base class, declare `face_layer` as a dependency, and integrate with the pipeline's progress tracking system.

### FR-007: Configuration Management
The system MUST load context-specific configuration from config.yaml, including Ollama connection settings, model selection, confidence thresholds, and batch processing settings.

### FR-008: Error Handling
The system MUST implement comprehensive error handling for Ollama failures, network timeouts, JSON parsing failures, and database errors. The system MUST store minimal data with low confidence when analysis fails completely.

### FR-009: Testing
The system MUST include unit tests for the Ollama client wrapper, integration tests with mocked LLM responses, and database transaction tests.

## Success Criteria

### SC-001: Successful Single Photo Analysis
When provided with a downsampled photo, the context layer returns valid structured context data with confidence scores within 5 seconds.

### SC-002: Batch Processing Completion
When processing a batch of 100 photos, the context layer completes analysis and stores all results in the database with accurate progress reporting.

### SC-003: Error Recovery
When LLM inference fails, the system successfully retries and falls back to alternative strategies, storing either valid data or minimal failure records.

### SC-004: Pipeline Integration
The context layer correctly declares dependencies, runs after the face layer completes, and reports progress through the pipeline framework.

### SC-005: Configuration Validation
The context layer validates all required configuration at startup and provides clear error messages for missing or invalid settings.

## Key Entities
- Photo: The image being analyzed
- ContextAnalysisResult: Structured result containing decade, season, event, medium with confidence scores
- OllamaClient: Wrapper for Ollama vision LLM API
- ContextAnalyzer: Orchestrates the analysis pipeline
- ContextLayerStage: Pipeline stage implementation
- Context table: Database table for storing analysis results

## Edge Cases
- Photos with no discernible visual context (abstract images)
- Photos with ambiguous or conflicting decade indicators
- Indoor photos without seasonal indicators
- Photos that could depict multiple events
- Large photo collections requiring graceful degradation
- Network failures during Ollama inference
- Invalid JSON responses from LLM

## Known Constraints
- Local-only principle: No external API calls for image analysis
- Non-destructive operation: Original files never modified
- Confidence scores everywhere: Every result must include a confidence score (0.0-1.0)
- SQLite as feature store: All inter-stage communication through database
- Apple Silicon optimization: MLX backend for Ollama on Apple Silicon
- Processing latency: 2-5 seconds per image acceptable for vision LLM inference