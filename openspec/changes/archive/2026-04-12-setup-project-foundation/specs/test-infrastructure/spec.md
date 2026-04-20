## ADDED Requirements

### Requirement: Test framework
The project SHALL use pytest as the test framework with comprehensive fixture support.

#### Scenario: Test discovery
- **WHEN** running `pytest` from the project root
- **THEN** tests SHALL be discovered in the `tests/` directory
- **AND** test modules SHALL follow the pattern `test_*.py` or `*_test.py`

#### Scenario: Test configuration
- **WHEN** checking the project configuration
- **THEN** there SHALL be a `pytest.ini` or `pyproject.toml` section for pytest
- **AND** it SHALL configure test discovery, logging, and coverage reporting

### Requirement: Test database fixture
Tests SHALL have access to a temporary SQLite database fixture.

#### Scenario: Database isolation
- **WHEN** running tests that use the database
- **THEN** each test SHALL get a fresh database instance
- **AND** database state SHALL not leak between tests

#### Scenario: Schema setup
- **WHEN** using the database fixture
- **THEN** the database SHALL have all required tables created
- **AND** the schema SHALL match the production schema exactly

### Requirement: Mock image data
Tests SHALL have access to mock image data for testing pipeline stages.

#### Scenario: Mock image generation
- **WHEN** running tests that require image input
- **THEN** there SHALL be fixture functions to generate test images
- **AND** these images SHALL have configurable metadata (EXIF, size, format)

#### Scenario: Image cleanup
- **WHEN** tests create temporary image files
- **THEN** these files SHALL be automatically cleaned up after tests complete
- **AND** no leftover files SHALL remain in the filesystem

### Requirement: AI model mocking
Tests SHALL mock heavy AI dependencies (InsightFace, Ollama) to enable CI testing.

#### Scenario: InsightFace mocking
- **WHEN** testing the face detection stage
- **THEN** there SHALL be a mock that returns predictable face detections
- **AND** the mock SHALL support configurable age estimates and confidence scores

#### Scenario: Ollama mocking
- **WHEN** testing the context analysis stage
- **THEN** there SHALL be a mock that returns structured JSON responses
- **AND** the mock SHALL validate prompt format and return appropriate decade estimates

### Requirement: Test coverage
The test infrastructure SHALL support code coverage measurement.

#### Scenario: Coverage reporting
- **WHEN** running tests with coverage enabled
- **THEN** coverage SHALL be measured for all source files
- **AND** a coverage report SHALL be generated in HTML and terminal formats

#### Scenario: Coverage thresholds
- **WHEN** checking test coverage
- **THEN** there SHALL be configurable minimum coverage thresholds
- **AND** the build SHALL fail if thresholds are not met