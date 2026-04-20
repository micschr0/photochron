"""
Configuration management for PhotoChron.

Loads configuration from config.yaml, environment variables, and provides
default values from the architecture specification.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from .models import (
    Config,
    ConfigModels,
    ConfigPipeline,
    ConfigPaths,
    ConfigIngestion,
    ConfigFace,
    ConfigContext,
    ConfigLogging,
)


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Load configuration from file, environment variables, and defaults.

    Args:
        config_path: Path to config.yaml file. If None, looks for config.yaml
                     in current directory and project root.

    Returns:
        Config: Validated configuration object.
    """
    # Find config file
    if config_path is None:
        possible_paths = [
            Path("config.yaml"),
            Path(__file__).parent.parent.parent / "config.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        else:
            # Create default config if none exists
            return _create_default_config()

    # Load YAML
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f) or {}

    # Migrate to current version if needed
    config_data = migrate_config(config_data)

    # Apply environment variable overrides
    config_data = _apply_env_overrides(config_data)

    # Validate with Pydantic
    return Config.model_validate(config_data)


def _create_default_config() -> Config:
    """Create default configuration based on architecture specification."""
    return Config(
        version="1.0",
        paths=ConfigPaths(
            cache_dir=".photochron",
            thumbs_dir=".photochron/thumbs",
            output_dir="photochron_output",
        ),
        models=ConfigModels(
            insightface_version="buffalo_l",
            ollama_model="llava-next:7b",
            fallback_model="moondream2",
            max_image_size=1024,
        ),
        ingestion=ConfigIngestion(
            max_downsample_size=1024,
            supported_formats=[
                ".jpg",
                ".jpeg",
                ".png",
                ".heic",
                ".heif",
                ".cr2",
                ".nef",
                ".arw",
                ".dng",
            ],
            skip_duplicates=True,
            extract_gps=True,
            fallback_timestamp="file_mtime",
        ),
        face=ConfigFace(
            model_name="buffalo_l",
            detection_threshold=0.5,
            matching_threshold=0.6,
            age_confidence_scale=0.1,
            use_gpu=False,
            batch_size=1,
        ),
        pipeline=ConfigPipeline(
            face_age_weight=0.45,
            llm_decade_weight=0.30,
            photo_medium_weight=0.10,
            min_confidence_threshold=0.5,
            max_pairwise_comparisons=500,
        ),
        context=ConfigContext(),
    )


def _apply_env_overrides(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Environment variables follow the pattern:
    PHOTOCHRON_<SECTION>_<KEY> (uppercase with underscores)

    Example: PHOTOCHRON_MODELS_MAX_IMAGE_SIZE=2048
    """
    prefix = "PHOTOCHRON_"

    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        # Convert PHOTOCHRON_MODELS_MAX_IMAGE_SIZE -> models.max_image_size
        parts = env_key[len(prefix) :].lower().split("_")

        # Navigate through nested structure
        current = config_data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set value with type conversion
        final_key = parts[-1]
        try:
            # Try to convert to int
            env_value = int(env_value)
        except ValueError:
            try:
                # Try to convert to float
                env_value = float(env_value)
            except ValueError:
                # Try to convert to boolean
                if env_value.lower() in ("true", "yes", "1"):
                    env_value = True
                elif env_value.lower() in ("false", "no", "0"):
                    env_value = False
                # Otherwise keep as string

        current[final_key] = env_value

    return config_data


def migrate_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate configuration data to current version.

    Args:
        config_data: Raw configuration data from YAML

    Returns:
        Migrated configuration data
    """
    version = config_data.get("version", "0.1")  # Assume old version if missing

    if version == "0.1":
        # Migration from 0.1 to 1.0
        # Remove any unknown fields (only keep version, paths, models, ingestion, face, pipeline, context)
        known_keys = {
            "version",
            "paths",
            "models",
            "ingestion",
            "face",
            "pipeline",
            "context",
            "logging",
        }
        keys_to_remove = [k for k in config_data.keys() if k not in known_keys]
        for key in keys_to_remove:
            del config_data[key]

        # Add version field if missing
        config_data["version"] = "1.0"

        # Ensure all sections exist with defaults
        if "paths" not in config_data:
            config_data["paths"] = {}
        if "models" not in config_data:
            config_data["models"] = {}
        if "ingestion" not in config_data:
            config_data["ingestion"] = {}
        if "face" not in config_data:
            config_data["face"] = {}
        if "pipeline" not in config_data:
            config_data["pipeline"] = {}
        if "context" not in config_data:
            config_data["context"] = {}
        if "logging" not in config_data:
            config_data["logging"] = {}

    # Add future migrations here as elif version == "1.0": ...

    return config_data


def save_config(config: Config, config_path: Path) -> None:
    """Save configuration to YAML file."""
    config_dict = config.model_dump(exclude_defaults=True, by_alias=True)
    with open(config_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
