# Changelog

All notable changes to PhotoChron will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Comprehensive Phase 3 testing suite**: Added extensive test coverage for Context Layer implementation:
  - **ContextAnalyzer strategy tests**: Tests for all 4 analysis strategies (DEFAULT, AGGRESSIVE, CONSERVATIVE, FAST)
  - **Integration tests**: Full pipeline integration tests for ContextLayerStage with database
  - **Error handling tests**: Comprehensive tests for retry logic and fallback strategies
  - **Confidence validation tests**: Tests for confidence score validation and propagation
  - **Database integration tests**: Tests for transaction integrity and schema compliance
- **Memory threshold configuration**: Added memory management fields to `ConfigContext`:
  - `memory_warning_threshold_mb`: Warning threshold for available memory (default: `100`)
  - `memory_critical_threshold_mb`: Critical threshold for available memory (default: `50`)
  - `memory_retry_delay_seconds`: Delay when memory is critically low (default: `30`)
- **Memory checking**: Before each batch, checks available system memory against configurable thresholds
  - Logs warning when memory falls below warning threshold
  - Skips batch and waits when memory falls below critical threshold
  - Graceful degradation if psutil unavailable
- **Progress reporting**: Enhanced batch processing with detailed percentage-based progress logging
  - Batch-level progress with percentage complete
  - Photo-level progress every 10 photos
  - Final completion percentage with failure count
- **Comprehensive error handling tests**: Added tests for:
  - JSON parsing fallback scenarios
  - Timeout handling for Ollama requests
  - Model fallback behavior
  - Memory check scenarios (warning, critical, error)
  - Progress reporting format and calculation
- **ConfigContext class**: New configuration model for context layer with comprehensive Ollama integration settings
- **Configuration validation**: Runtime health checks for Ollama server and model availability
- **Graceful degradation**: Automatic fallback to simpler models and degraded mode operation
- **Retry logic**: Configurable retry attempts with exponential backoff for transient failures
- **Health monitoring**: Real-time health status reporting via `health_status` property
- **Database integration tests**: Comprehensive tests for `_get_photos_without_context()` method

### Changed
- **Config class**: Updated to include `context` configuration section with memory management fields
- **ContextLayerStage initialization**: Modified to use `self.config.context` for configuration
- **ContextLayerStage.run()**: Enhanced with memory checking and progress reporting
- **Package structure**: Established complete Python package structure with `__init__.py` files in all directories

### Configuration Changes
- Added `context` section to configuration with the following options:
  - `ollama_host`: Ollama server URL (default: `http://localhost:11434`)
  - `ollama_timeout`: Timeout in seconds for Ollama requests (default: `300`)
  - `max_retries`: Maximum retry attempts for LLM failures (default: `3`)
  - `retry_delay`: Delay between retries in seconds (default: `2.0`)
  - `primary_model`: Primary vision LLM model (default: `llava-next:7b`)
  - `fallback_model`: Fallback vision model (default: `moondream2`)
  - `batch_size`: Batch size for processing images (default: `1`)
  - `min_decade_confidence`: Minimum confidence for decade estimates (default: `0.3`)
  - `min_season_confidence`: Minimum confidence for season estimates (default: `0.4`)
  - `use_fallback_on_failure`: Use fallback strategies on analysis failure (default: `true`)
  - `store_minimal_on_complete_failure`: Store minimal data when analysis completely fails (default: `true`)
  - `memory_warning_threshold_mb`: Memory warning threshold in MB (default: `100`)
  - `memory_critical_threshold_mb`: Memory critical threshold in MB (default: `50`)
  - `memory_retry_delay_seconds`: Delay in seconds when memory is critically low (default: `30`)

### Documentation
- Created comprehensive pipeline documentation in `docs/pipeline.md`
- Added configuration reference in `docs/configuration.md`
- Updated agent documentation to reflect new configuration structure

## [0.1.0] - 2026-04-13

### Added
- Initial project foundation with 6-stage pipeline architecture
- Ingestion stage with EXIF extraction and downsampling
- Face layer with InsightFace integration for face detection and age estimation
- Basic configuration management with Pydantic models
- SQLite feature store for caching inference results
- CLI interface with Typer framework
- Comprehensive test infrastructure

### Configuration
- Basic configuration structure with paths, models, ingestion, face, and pipeline sections
- Environment variable support for configuration overrides
- YAML configuration file support