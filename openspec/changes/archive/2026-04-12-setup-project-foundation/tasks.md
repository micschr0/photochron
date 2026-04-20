## 1. Project Structure & Dependencies

- [x] 1.1 Create `src/photochron/` directory structure with submodules: `cli/`, `pipeline/`, `store/`, `config/`, `models/`, `utils/`
- [x] 1.2 Create `pyproject.toml` with `[project]` section, dependencies (Typer, Rich, Pillow, piexif, PyYAML, pydantic, pytest), and build configuration
- [x] 1.3 Create `src/photochron/__init__.py` with version and package metadata
- [x] 1.4 Create `src/photochron/__main__.py` as CLI entry point
- [x] 1.5 Set up development tools configuration: `ruff.toml`, `.pre-commit-config.yaml`, `.gitignore`
- [x] 1.6 Create `.photochron/` directory for cache and thumbs (gitignored)

## 2. Feature Store (SQLite)

- [x] 2.1 Create `src/photochron/store/__init__.py` with database connection manager
- [x] 2.2 Implement `DatabaseStore` class with context manager for connections
- [x] 2.3 Create SQL schema definition in `src/photochron/store/schema.py` with tables: photos, faces, context, rankings, pipeline_runs, persons
- [x] 2.4 Implement schema creation and migration logic with version tracking in `pipeline_runs` table
- [x] 2.5 Create data models in `src/photochron/models/` (Pydantic models for each table)
- [x] 2.6 Implement query helper functions for common operations (insert photo, get faces by photo_id, etc.)
- [x] 2.7 Add connection pooling and transaction management

## 3. Configuration System

- [x] 3.1 Create `src/photochron/config/__init__.py` with configuration loading logic
- [x] 3.2 Define Pydantic models for config sections in `src/photochron/config/models.py`
- [x] 3.3 Create default `config.yaml` with values from architecture specification
- [x] 3.4 Implement environment variable override system with `PHOTOCHRON_` prefix
- [x] 3.5 Create `anchors.yaml` template with commented examples
- [x] 3.6 Implement anchors parsing and validation in `src/photochron/config/anchors.py`
- [x] 3.7 Add configuration versioning and migration support

## 4. CLI Implementation

- [x] 4.1 Create `src/photochron/cli/__init__.py` with Typer app definition
- [x] 4.2 Implement `run` command with `--input`, `--output`, `--dry-run` options
- [x] 4.3 Implement `cluster` command for face clustering and person assignment
- [x] 4.4 Implement `rerun` command with `--stage` parameter
- [x] 4.5 Implement `status` command showing pipeline progress and cache stats
- [x] 4.6 Add Rich integration for progress bars, tables, and formatted output
- [x] 4.7 Implement parameter validation and helpful error messages
- [x] 4.8 Add `--version` flag showing package version

## 5. Pipeline Foundation

- [x] 5.1 Create `src/photochron/pipeline/__init__.py` with `PipelineStage` abstract base class
- [x] 5.2 Implement stage registry and dependency tracking
- [x] 5.3 Create placeholder modules for all 6 stages in `src/photochron/pipeline/stages/` (ingestion, face_layer, context_layer, anchor_layer, ranking_engine, output_layer)
- [x] 5.4 Implement basic pipeline runner that executes stages in order with SQLite communication
- [x] 5.5 Add stage re-run capability using `pipeline_runs` table for tracking
- [x] 5.6 Implement confidence score propagation between stages

## 6. Test Infrastructure

- [x] 6.1 Create `tests/` directory with `conftest.py` for shared fixtures
- [x] 6.2 Implement database fixture providing temporary SQLite instance
- [x] 6.3 Create mock image generation utilities in `tests/fixtures/images.py`
- [x] 6.4 Implement InsightFace mock returning predictable face detections
- [x] 6.5 Implement Ollama mock returning structured JSON responses
- [x] 6.6 Write unit tests for configuration loading and validation
- [x] 6.7 Write unit tests for database store operations
- [x] 6.8 Write integration test for basic pipeline flow with mocked AI
- [x] 6.9 Set up pytest configuration with coverage reporting
- [x] 6.10 Create test data: 5-10 sample images with metadata for integration testing

## 7. Integration & Validation

- [x] 7.1 Test end-to-end CLI: `photochron run --input tests/data/sample_images --output tests/output --dry-run`
- [x] 7.2 Verify SQLite schema matches specification (all tables and columns present)
- [x] 7.3 Verify configuration defaults match architecture spec values
- [x] 7.4 Run full test suite with coverage report (aim for >80% coverage)
- [x] 7.5 Test stage re-run capability: modify a photo, run specific stage only
- [x] 7.6 Validate non-destructive operation: ensure no input files are modified
- [x] 7.7 Verify confidence scores are present in all database results
- [x] 7.8 Create documentation: update `CLAUDE.md` with new commands and structure