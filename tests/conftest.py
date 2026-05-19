"""
Pytest fixtures for photochron tests.
"""

import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest
from loguru import logger as _loguru_logger
from PIL import Image

from photochron import config as _config_module
from photochron.config import Config, ConfigModels, ConfigPaths, ConfigPipeline
from photochron.pipeline import reset_registry
from photochron.store import DatabaseStore, close_store
from photochron.store.schema import create_schema


@pytest.fixture
def caplog(caplog):
    """Route loguru records into pytest's caplog handler."""
    handler_id = _loguru_logger.add(
        caplog.handler,
        format="{message}",
        level=0,
        filter=lambda record: record["level"].no >= caplog.handler.level,
    )
    yield caplog
    _loguru_logger.remove(handler_id)


@pytest.fixture
def temp_db_path() -> Generator[Path, None, None]:
    """Create temporary database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    db_path.unlink(missing_ok=True)


@pytest.fixture
def database_store(temp_db_path: Path) -> Generator[DatabaseStore, None, None]:
    """Create database store with temporary database."""
    store = DatabaseStore(temp_db_path)
    with store.transaction() as conn:
        create_schema(conn)
    yield store
    store.close()


@pytest.fixture
def mock_config() -> Config:
    """Create mock configuration for testing."""
    return Config(
        version="1.0",
        paths=ConfigPaths(
            cache_dir=".test_cache",
            thumbs_dir=".test_cache/thumbs",
            output_dir="test_output",
        ),
        models=ConfigModels(
            insightface_version="test_model",
            ollama_model="test_llm",
            fallback_model="test_fallback",
            max_image_size=512,
        ),
        pipeline=ConfigPipeline(
            face_age_weight=0.45,
            llm_decade_weight=0.30,
            photo_medium_weight=0.10,
            min_confidence_threshold=0.5,
            max_pairwise_comparisons=100,
        ),
    )


@pytest.fixture
def sample_image_path(tmp_path: Path) -> Path:
    """Write a tiny synthetic JPEG to ``tmp_path`` and return its path.

    The previous fixture returned a string pointing at a non-existent file,
    which silently broke any test depending on it. Generating an in-memory
    1x1 image gives every test an isolated file without dragging a binary
    fixture into version control.
    """
    image_path = tmp_path / "sample.jpg"
    Image.new("RGB", (8, 8), color=(127, 127, 127)).save(image_path, format="JPEG")
    return image_path


@pytest.fixture
def mock_insightface():
    """Mock InsightFace detector."""

    class MockInsightFace:
        def detect(self, image):
            return [
                {
                    "bbox": [10, 10, 100, 100],
                    "embedding": b"fake_embedding",
                    "age": 25.0,
                    "age_std": 2.0,
                    "confidence": 0.9,
                }
            ]

    return MockInsightFace()


@pytest.fixture
def mock_ollama():
    """Mock Ollama vision LLM."""

    class MockOllama:
        def generate(self, prompt, image):
            return '{"decade": "1990-1995", "decade_confidence": 0.8, "season": "summer", "event_hint": null, "photo_medium": "print_scan", "photo_medium_confidence": 0.7}'

    return MockOllama()


@pytest.fixture(autouse=True)
def cleanup_global_state():
    """Reset every process-global singleton before AND after each test.

    Order-dependent failures used to be hard to spot because only the
    ``DatabaseStore`` was reset; the pipeline registry and the Config
    singleton leaked between tests. Resetting all three here keeps tests
    independent regardless of execution order.
    """
    close_store()
    _config_module._config = None
    reset_registry()
    yield
    close_store()
    _config_module._config = None
    reset_registry()
