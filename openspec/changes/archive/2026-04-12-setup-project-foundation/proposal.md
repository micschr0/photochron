## Why

PhotoChron is a fully specified but unimplemented greenfield project with comprehensive architecture documentation. We need to establish the foundational codebase structure that respects the architectural correctness priority, sets up proper Python packaging, implements the CLI skeleton, defines the SQLite schema, and creates a testable infrastructure before implementing the AI pipeline stages.

This foundation is critical because:
1. The 6-stage pipeline design requires careful stage isolation and SQLite-based communication
2. Heavy dependencies (InsightFace, Ollama/MLX) need proper integration patterns
3. Non-destructive operation and confidence scoring are core requirements that must be baked in from the start
4. Architectural correctness is prioritized over speed-to-first-run

## What Changes

- Create Python package structure with proper module organization (`src/photochron/`)
- Implement basic CLI using Typer with all commands defined in `agent_docs/commands.md`
- Define and implement SQLite Feature Store schema matching the pipeline stage requirements
- Create configuration system (`config.yaml`) with default values from architecture spec
- Set up test infrastructure with pytest, including test image dataset for validation
- Establish dependency management with `requirements.txt` or `pyproject.toml`
- Implement basic logging and progress reporting using Rich
- Create placeholder modules for all 6 pipeline stages with proper interfaces

**BREAKING**: This introduces the entire codebase structure - there is no existing code to break.

## Capabilities

### New Capabilities

- **project-structure**: Defines the Python package layout, module organization, and import structure for the PhotoChron application. Ensures clean separation between pipeline stages, data models, and CLI.
- **cli-interface**: Implements the Typer-based command-line interface with all commands specified in architecture docs. Handles argument parsing, help text, and command routing.
- **feature-store**: Defines the SQLite database schema for the Feature Store, including tables for photos, faces, context, rankings, and pipeline runs. Manages migrations, cache invalidation, and connection pooling.
- **configuration-management**: Handles loading and validation of `config.yaml` with default values, environment overrides, and schema versioning.
- **test-infrastructure**: Establishes pytest configuration, test utilities, mock data generation, and integration test patterns for the pipeline stages.

### Modified Capabilities

<!-- No existing capabilities to modify -->

## Impact

- **Codebase**: Creates the entire source code structure from scratch in `src/`
- **Dependencies**: Adds Python dependencies (Typer, Rich, Pillow, piexif, PyYAML, pytest, etc.)
- **Database**: Introduces SQLite database file (`.photochron/cache.db`) with specific schema
- **Configuration**: Adds `config.yaml` and `anchors.yaml` (template) to project root
- **Development workflow**: Establishes linting (ruff), type checking (mypy), and testing patterns
- **Build/output**: Creates output directory structure for renamed and EXIF-enriched copies