"""
Configuration management for photochron.

Loads configuration from config.yaml, environment variables, and provides
default values from the architecture specification.
"""

import os
from pathlib import Path
from typing import Any

import yaml

from .models import (
    Config,
    ConfigContext,
    ConfigFace,
    ConfigIngestion,
    ConfigLogging,
    ConfigModels,
    ConfigPaths,
    ConfigPipeline,
)


def load_config(config_path: Path | None = None) -> Config:
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
    with open(config_path) as f:
        config_data = yaml.safe_load(f) or {}

    # Migrate to current version if needed
    config_data = migrate_config(config_data)

    # Apply environment variable overrides
    config_data = _apply_env_overrides(config_data)

    # Validate with Pydantic
    return Config.model_validate(config_data)


def _create_default_config() -> Config:
    """Create default configuration when no config.yaml exists.

    Model names are intentionally left empty – users must opt in by
    uncommenting model entries in config.yaml (with license verification).
    """
    return Config(
        version="1.0",
        paths=ConfigPaths(
            cache_dir=".photochron",
            thumbs_dir=".photochron/thumbs",
            output_dir="photochron_output",
        ),
        models=ConfigModels(max_image_size=1024),
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
            extract_gps=False,
            fallback_timestamp="file_mtime",
            workers=4,
        ),
        face=ConfigFace(
            detection_threshold=0.5,
            matching_threshold=0.6,
            age_confidence_scale=0.1,
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


# Top-level config sections (used for env-var parsing to correctly split
# section names from multi-word keys like `ollama_host`).
_CONFIG_SECTIONS = (
    "paths",
    "models",
    "ingestion",
    "face",
    "pipeline",
    "context",
    "logging",
)


def _apply_env_overrides(config_data: dict[str, Any]) -> dict[str, Any]:
    """
    Apply environment variable overrides to configuration.

    Environment variables follow the pattern:
    PHOTOCHRON_<SECTION>_<KEY> where <KEY> may contain underscores.

    Examples:
        PHOTOCHRON_MODELS_MAX_IMAGE_SIZE=2048  -> models.max_image_size
        PHOTOCHRON_CONTEXT_OLLAMA_HOST=http://x -> context.ollama_host
        PHOTOCHRON_FACE_USE_GPU=true           -> face.use_gpu
    """
    prefix = "PHOTOCHRON_"

    for env_key, raw_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        rest = env_key[len(prefix) :].lower()

        # Match known section prefix so multi-word keys stay intact.
        section = next((s for s in _CONFIG_SECTIONS if rest.startswith(s + "_")), None)
        if section is None:
            # Unknown section – skip silently rather than corrupting config_data.
            continue
        key = rest[len(section) + 1 :]
        if not key:
            continue

        converted: Any = raw_value
        try:
            converted = int(raw_value)
        except ValueError:
            try:
                converted = float(raw_value)
            except ValueError:
                if raw_value.lower() in ("true", "yes", "1"):
                    converted = True
                elif raw_value.lower() in ("false", "no", "0"):
                    converted = False

        config_data.setdefault(section, {})[key] = converted

    return config_data


def migrate_config(config_data: dict[str, Any]) -> dict[str, Any]:
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
_config: Config | None = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
