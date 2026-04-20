"""
Unit tests for configuration loading and validation.
"""

import tempfile
from pathlib import Path
import yaml
import pytest

from photochron.config import Config, load_config, save_config
from photochron.config.models import (
    ConfigPaths,
    ConfigModels,
    ConfigPipeline,
    ConfigContext,
)


def test_config_defaults():
    """Test that default configuration has correct values."""
    config = Config()

    assert config.version == "1.0"
    assert config.paths.cache_dir == ".photochron"
    assert config.paths.thumbs_dir == ".photochron/thumbs"
    assert config.paths.output_dir == "photochron_output"

    assert config.models.insightface_version == "buffalo_l"
    assert config.models.ollama_model == "llava-next:7b"
    assert config.models.fallback_model == "moondream2"
    assert config.models.max_image_size == 1024

    assert config.pipeline.face_age_weight == 0.45
    assert config.pipeline.llm_decade_weight == 0.30
    assert config.pipeline.photo_medium_weight == 0.10
    assert config.pipeline.min_confidence_threshold == 0.5
    assert config.pipeline.max_pairwise_comparisons == 500


def test_config_validation():
    """Test configuration validation with invalid values."""
    # Test invalid weight (should raise validation error)
    with pytest.raises(ValueError):
        Config(pipeline=ConfigPipeline(face_age_weight=1.5))

    # Test invalid image size
    with pytest.raises(ValueError):
        Config(models=ConfigModels(max_image_size=100))  # below minimum

    # Test valid configuration should not raise
    config = Config(
        models=ConfigModels(max_image_size=2048),
        pipeline=ConfigPipeline(min_confidence_threshold=0.7),
    )
    assert config.models.max_image_size == 2048
    assert config.pipeline.min_confidence_threshold == 0.7


def test_load_config_from_file():
    """Test loading configuration from YAML file."""
    config_data = {
        "version": "1.0",
        "paths": {
            "cache_dir": "/tmp/test_cache",
            "output_dir": "/tmp/test_output",
        },
        "models": {
            "max_image_size": 2048,
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = Path(f.name)

    try:
        config = load_config(config_path)

        assert config.version == "1.0"
        assert config.paths.cache_dir == "/tmp/test_cache"
        assert config.paths.output_dir == "/tmp/test_output"
        assert config.models.max_image_size == 2048

        # Check that defaults are used for unspecified fields
        assert config.models.insightface_version == "buffalo_l"
        assert config.pipeline.face_age_weight == 0.45
    finally:
        config_path.unlink()


def test_save_config():
    """Test saving configuration to file."""
    config = Config(
        paths=ConfigPaths(cache_dir="/custom/cache"),
        models=ConfigModels(max_image_size=2048),
    )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        config_path = Path(f.name)

    try:
        save_config(config, config_path)

        # Load saved config and verify
        with open(config_path, "r") as f:
            saved_data = yaml.safe_load(f)

        assert saved_data["paths"]["cache_dir"] == "/custom/cache"
        assert saved_data["models"]["max_image_size"] == 2048
        # Unspecified fields should not be in output (exclude_defaults=True)
        assert "fallback_model" not in saved_data["models"]
    finally:
        config_path.unlink()


def test_config_migration():
    """Test configuration version migration."""
    # Simulate old config format (version 0.1)
    old_config = {"version": "0.1", "some_old_field": "old_value"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(old_config, f)
        config_path = Path(f.name)

    try:
        # Should migrate to version 1.0
        config = load_config(config_path)

        assert config.version == "1.0"
        # Should have all required sections with defaults
        assert config.paths.cache_dir == ".photochron"
        assert config.models.max_image_size == 1024
    finally:
        config_path.unlink()


def test_config_context_defaults():
    """Test that ConfigContext has correct default values."""
    context = ConfigContext()

    assert context.ollama_host == "http://localhost:11434"
    assert context.ollama_timeout == 300
    assert context.max_retries == 3
    assert context.retry_delay == 2.0
    assert context.primary_model == "llava-next:7b"
    assert context.fallback_model == "moondream2"
    assert context.batch_size == 1
    assert context.min_decade_confidence == 0.3
    assert context.min_season_confidence == 0.4
    assert context.use_fallback_on_failure == True
    assert context.store_minimal_on_complete_failure == True
