## ADDED Requirements

### Requirement: SQLite schema
The Feature Store SHALL use SQLite with tables matching the 6-stage pipeline architecture.

#### Scenario: Table structure
- **WHEN** inspecting the database schema
- **THEN** there SHALL be tables: `photos`, `faces`, `context`, `rankings`, `pipeline_runs`, `persons`
- **AND** each table SHALL have columns as defined in the architecture specification

#### Scenario: Photos table columns
- **WHEN** examining the `photos` table
- **THEN** it SHALL contain columns: `id`, `content_hash`, `file_path`, `downsample_path`, `exif_datetime`, `make`, `model`, `perceptual_hash`, `created_at`

### Requirement: Connection management
The Feature Store SHALL manage database connections efficiently and provide context managers for transactions.

#### Scenario: Connection pooling
- **WHEN** multiple pipeline stages access the database concurrently
- **THEN** connections SHALL be reused via connection pooling
- **AND** there SHALL be no connection leaks

#### Scenario: Transaction safety
- **WHEN** a pipeline stage fails during database operations
- **THEN** incomplete transactions SHALL be rolled back
- **AND** the database SHALL remain in a consistent state

### Requirement: Cache invalidation
The Feature Store SHALL support cache invalidation based on content hashes and model versions.

#### Scenario: Content hash changes
- **WHEN** a photo file's content changes (different MD5 hash)
- **THEN** all dependent feature rows (faces, context, rankings) SHALL be marked as invalid
- **AND** subsequent pipeline runs SHALL recompute features for that photo

#### Scenario: Model version changes
- **WHEN** the InsightFace or Ollama model version changes
- **THEN** the `pipeline_runs` table SHALL record the new model version
- **AND** affected feature rows SHALL be flagged for recomputation

### Requirement: Migration support
The Feature Store SHALL support schema migrations without data loss.

#### Scenario: Schema versioning
- **WHEN** checking the database version
- **THEN** the `pipeline_runs` table SHALL contain a `schema_version` column
- **AND** migration scripts SHALL be available for each version increment

#### Scenario: Migration application
- **WHEN** a new schema version is detected
- **THEN** migration scripts SHALL be applied automatically
- **AND** existing data SHALL be preserved where possible