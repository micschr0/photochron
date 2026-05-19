# Testing Strategy and Test Suite

photochron uses a comprehensive testing strategy to ensure reliability, correctness, and maintainability of the codebase. This document describes the testing approach, test organization, and how to run tests.

## Test Organization

Tests are organized into the following categories:

### Unit Tests (`tests/unit/`)
- **Location**: `tests/unit/`
- **Purpose**: Test individual components in isolation
- **Mocking**: Extensive use of mocks to isolate components
- **Coverage**: Focus on business logic, edge cases, and error handling

### Integration Tests (`tests/integration/`)
- **Location**: `tests/integration/`
- **Purpose**: Test interactions between components
- **Databases**: Use SQLite in-memory databases for database integration tests
- **Mocking**: Minimal mocking, focus on real component interactions

### Test Structure
```
tests/
├── unit/
│   ├── context/
│   │   └── test_analyzer.py          # ContextAnalyzer strategy tests
│   ├── models/
│   │   └── test_ollama_client.py     # OllamaClient tests
│   ├── test_error_handling.py        # Comprehensive error handling
│   ├── test_confidence_validation.py # Confidence score validation
│   └── test_database_integration.py  # Database transaction tests
├── integration/
│   └── test_context_layer.py         # Full pipeline integration
└── conftest.py                       # Shared fixtures
```

## Test Categories

### 1. ContextAnalyzer Strategy Tests
Tests for all 4 analysis strategies implemented in `ContextAnalyzer`:

#### Analysis Strategies Tested:
- **DEFAULT**: Standard analysis with primary model fallback
- **AGGRESSIVE**: Try multiple models and prompts for best results
- **CONSERVATIVE**: Only return results with high confidence
- **FAST**: Use simpler prompts and skip retries for speed

#### Key Test Scenarios:
- Success cases for each strategy
- Fallback behavior when primary model fails
- Confidence threshold validation
- Strategy override at runtime
- Singleton pattern for analyzer instances

### 2. Error Handling Tests
Comprehensive tests for error scenarios and recovery:

#### Error Types Tested:
- **Network errors**: Connection refused, timeout, socket errors
- **Model errors**: Model not found, model unavailable
- **JSON errors**: Malformed JSON, validation errors
- **Resource errors**: Memory constraints, file system errors

#### Recovery Mechanisms Tested:
- **Retry logic**: Exponential backoff with jitter
- **Circuit breakers**: Prevent cascading failures
- **Fallback strategies**: Model fallback, prompt fallback
- **Graceful degradation**: Continue operation in degraded mode

### 3. Confidence Validation Tests
Tests for confidence score validation and propagation:

#### Validation Scenarios:
- **Boundary validation**: Confidence scores 0.0-1.0
- **Threshold validation**: Minimum confidence requirements
- **Field clearing**: Low confidence fields are cleared
- **Overall confidence**: Weighted average calculation

#### Propagation Tests:
- Confidence propagation through retry logic
- Confidence propagation through model fallback
- Confidence propagation across different strategies

### 4. Database Integration Tests
Tests for database operations and transaction integrity:

#### Transaction Tests:
- **Commit on success**: Verify data persistence
- **Rollback on failure**: Verify data consistency
- **Nested transactions**: Proper handling of nested contexts
- **Constraint violations**: Foreign key and unique constraints

#### Query Tests:
- **Context record insertion**: Full and minimal data scenarios
- **Batch queries**: Pagination and offset handling
- **Count operations**: Accurate record counting
- **Schema compliance**: Indexes and constraints

### 5. Integration Tests
Full pipeline integration tests:

#### Integration Scenarios:
- **Basic integration**: Full pipeline with mocked analysis
- **Duplicate processing**: Skip photos with existing context
- **Analysis failure**: Handle complete analysis failure
- **Low confidence**: Handle results below confidence thresholds
- **Degraded mode**: Operation when Ollama is unavailable
- **Batch processing**: Process multiple photos in batches
- **Missing files**: Handle missing downsampled images

## Test Fixtures

### Database Fixtures
```python
@pytest.fixture
def database_store():
    """Create an in-memory SQLite database for testing."""
    store = DatabaseStore(":memory:")
    with store.transaction() as conn:
        create_schema(conn)
    yield store
    store.close()
```

### Mock Fixtures
```python
@pytest.fixture
def mock_ollama_client():
    """Create a mock OllamaClient for unit tests."""
    mock_client = Mock(spec=OllamaClient)
    mock_client.analyze_image_context = Mock()
    mock_client.get_prompt_template = Mock(return_value="Test prompt template")
    mock_client.connect = Mock(return_value=True)
    mock_client.health_check = Mock(
        return_value={"status": "healthy", "server_available": True}
    )
    return mock_client
```

### Configuration Fixtures
```python
@pytest.fixture
def mock_config():
    """Create a mock Config for testing."""
    mock_config = Mock(spec=Config)
    mock_config.context = Mock(spec=ConfigContext)
    # Set configuration values...
    return mock_config
```

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Categories
```bash
# Run unit tests only
pytest tests/unit/

# Run integration tests only
pytest tests/integration/

# Run tests by marker
pytest -m "integration"
```

### Run with Coverage
```bash
# Generate coverage report
pytest --cov=photochron tests/

# Generate HTML coverage report
pytest --cov=photochron --cov-report=html tests/
```

### Run Specific Test Files
```bash
# Run ContextAnalyzer tests
pytest tests/unit/context/test_analyzer.py

# Run error handling tests
pytest tests/unit/test_error_handling.py

# Run integration tests
pytest tests/integration/test_context_layer.py
```

## Test Design Principles

### 1. Isolation
- Unit tests mock external dependencies
- Integration tests use real components with minimal mocking
- Database tests use in-memory SQLite

### 2. Determinism
- Tests produce the same results every time
- Mock responses are predictable
- Database state is reset between tests

### 3. Coverage
- Test success and failure paths
- Test edge cases and boundary conditions
- Test error recovery mechanisms

### 4. Maintainability
- Clear test names and organization
- Reusable fixtures
- Minimal test duplication

## Test Data Patterns

### Mock Responses
```python
mock_context_result = ContextAnalysisResult(
    decade="1985-1990",
    decade_confidence=0.75,
    season="summer",
    event_hint=None,
    photo_medium="print_scan",
    photo_medium_confidence=0.8,
)
```

### Test Sequences
```python
# Test retry logic with sequence of failures then success
mock_ollama_client.analyze_image_context.side_effect = [
    ConnectionError("First failure"),
    ConnectionError("Second failure"),
    mock_context_result,  # Success on third attempt
]
```

### Database Test Data
```python
# Insert test data
with store.transaction() as conn:
    conn.execute(
        "INSERT INTO photos (content_hash, file_path) VALUES (?, ?)",
        ("test_hash", "/test/photo.jpg"),
    )
```

## Continuous Integration

Tests are designed to run in CI environments:
- **No external dependencies**: Tests don't require Ollama server
- **Fast execution**: Tests complete in seconds
- **Deterministic**: No flaky tests
- **Comprehensive**: High code coverage

## Adding New Tests

When adding new tests:
1. Place unit tests in `tests/unit/` with appropriate subdirectory
2. Place integration tests in `tests/integration/`
3. Use existing fixtures when possible
4. Follow naming conventions: `test_<function>_<scenario>`
5. Include docstrings explaining test purpose
6. Test both success and failure paths

## Test Maintenance

Regular test maintenance includes:
- Updating tests when interfaces change
- Adding tests for new features
- Removing obsolete tests
- Improving test performance
- Increasing test coverage
