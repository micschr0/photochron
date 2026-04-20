## Context

PhotoChron is a greenfield project with comprehensive architecture documentation but no implementation code. The project has a well-defined 6-stage pipeline architecture, SQLite Feature Store design, and strict boundaries (local-only inference, non-destructive operation, confidence scoring). This design document addresses the foundational codebase setup needed before implementing the AI pipeline stages.

Current state: Only documentation files exist (`agent_docs/`, `CLAUDE.md`, OpenSpec configuration). No Python source code, dependencies, or project structure.

## Goals / Non-Goals

**Goals:**
1. Establish a clean, maintainable Python package structure that reflects the 6-stage pipeline architecture
2. Implement a fully functional CLI skeleton with all commands defined in architecture docs
3. Create a robust SQLite Feature Store with proper schema, migrations, and connection management
4. Set up configuration management with validation and environment-aware defaults
5. Create a testable infrastructure with proper mocking for heavy dependencies (InsightFace, Ollama)
6. Ensure architectural patterns (stage isolation, cache-first, confidence propagation) are baked in from the start

**Non-Goals:**
1. Implementing actual AI inference (InsightFace, Ollama integration) - this is for later changes
2. Performance optimization of image processing or database queries
3. User interface beyond CLI (no GUI, web UI)
4. Cross-platform compatibility beyond macOS/Apple Silicon (though design should not preclude it)
5. Advanced features like distributed processing or cloud backup

## Decisions

### 1. Package Structure
**Decision**: Use `src/photochron/` layout with clear module separation
**Rationale**: 
- `src/` layout prevents accidental imports from test directories and enforces clean boundaries
- Modules reflect pipeline architecture: `cli/`, `pipeline/`, `store/`, `config/`, `models/`, `utils/`
- Each pipeline stage gets its own module under `pipeline/stages/`
**Alternative considered**: Flat module structure - rejected as it would become unwieldy with 6+ stages

### 2. Dependency Management
**Decision**: Use `pyproject.toml` with `[project]` and `[build-system]` sections
**Rationale**:
- Modern Python standard (PEP 621)
- Single file for dependencies, build configuration, and tool configuration
- Better tooling support (uv, pdm, poetry)
**Alternative considered**: `requirements.txt` + `setup.py` - rejected as legacy approach

### 3. SQLite Interface
**Decision**: Use `sqlite3` standard library with custom context managers and minimal abstraction
**Rationale**:
- No external dependencies for core data persistence
- Simple schema migration via `pipeline_runs` table versioning
- Direct control over connection pooling and transaction management
- Lightweight enough for the Feature Store pattern
**Alternative considered**: SQLAlchemy Core - rejected as overkill for single-database, simple schema app

### 4. Configuration Validation
**Decision**: Use Pydantic for configuration validation
**Rationale**:
- Strong type validation for config values
- Environment variable integration
- JSON schema generation for documentation
- Clean separation between config loading and business logic
**Alternative considered**: Custom validation with `@dataclass` - rejected as more error-prone

### 5. CLI Framework
**Decision**: Use Typer with Rich for progress reporting
**Rationale**:
- Type-annotated CLI with automatic help generation
- Rich integration for beautiful terminal output and progress bars
- Subcommand structure matches the 6-command architecture
**Alternative considered**: Click - rejected due to better type hint integration in Typer

### 6. Stage Interface Pattern
**Decision**: Abstract base class `PipelineStage` with explicit input/output contracts
**Rationale**:
- Enforces stage isolation (only communicates via SQLite)
- Enables re-run capability via stage dependency tracking
- Clear contract for testing and mocking
**Alternative considered**: Function-based stages - rejected as less maintainable for complex dependencies

### 7. Testing Strategy
**Decision**: Pytest with fixture-based test database and image mocks
**Rationale**:
- Industry standard for Python testing
- Fixtures allow clean test isolation
- Mock heavy dependencies (InsightFace, Ollama) to enable CI testing
**Alternative considered**: unittest - rejected due to less expressive syntax and larger boilerplate

## Risks / Trade-offs

**Risk**: Over-engineering the foundation before validating pipeline concepts
**Mitigation**: Keep implementations minimal but correct; use extensive mocking to test integration points

**Risk**: SQLite schema changes requiring migrations
**Mitigation**: Version all schema changes in `pipeline_runs` table; provide migration scripts

**Risk**: Performance of naive SQLite implementation with large photo collections (1000+ photos)
**Mitigation**: Design with indexing from the start; profile and optimize in later changes

**Risk**: Configuration complexity overwhelming early development
**Mitigation**: Start with minimal config, expand as needed; use sensible defaults

**Trade-off**: Pure `sqlite3` vs ORM - choosing simplicity over convenience may increase boilerplate
**Mitigation**: Create reusable query builders and context managers to reduce repetition

**Trade-off**: Early focus on architectural correctness may delay first working prototype
**Mitigation**: Prioritize "working skeleton" approach - get basic pipeline flowing with mocks before full implementation