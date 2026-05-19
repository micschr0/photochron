"""
Pydantic models for photochron configuration.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigPaths(BaseModel):
    """Path configuration."""

    cache_dir: str = Field(".photochron", description="Directory for cache and database")
    thumbs_dir: str = Field(".photochron/thumbs", description="Directory for downsampled thumbnails")
    output_dir: str = Field("photochron_output", description="Directory for output files")


class ConfigModels(BaseModel):
    """AI model configuration.

    Model names are not hardcoded – users must explicitly configure them in
    config.yaml and verify model licenses for their intended use.
    """

    insightface_version: str = Field("", description="InsightFace model version (opt-in, see config.yaml)")
    ollama_model: str = Field("", description="Ollama vision LLM model (opt-in, see config.yaml)")
    fallback_model: str = Field("", description="Fallback vision model (opt-in, see config.yaml)")
    max_image_size: int = Field(
        1024,
        ge=256,
        le=4096,
        description="Maximum image size for processing (longest edge)",
    )


class ConfigPipeline(BaseModel):
    """Pipeline configuration."""

    face_age_weight: float = Field(0.45, ge=0.0, le=1.0, description="Weight for face age estimates in ranking")
    llm_decade_weight: float = Field(0.30, ge=0.0, le=1.0, description="Weight for LLM decade estimates in ranking")
    photo_medium_weight: float = Field(0.10, ge=0.0, le=1.0, description="Weight for photo medium priors in ranking")
    min_confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for results to be considered reliable",
    )
    max_pairwise_comparisons: int = Field(
        500,
        ge=0,
        le=10000,
        description="Maximum number of pairwise LLM comparisons per run",
    )


class ConfigIngestion(BaseModel):
    """Ingestion stage configuration."""

    max_downsample_size: int = Field(
        1024,
        ge=256,
        le=4096,
        description="Maximum size for downsampled images (longest edge in pixels)",
    )
    supported_formats: list[str] = Field(
        default_factory=lambda: [
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
        description="Supported image file extensions (case-insensitive)",
    )
    skip_duplicates: bool = Field(
        True,
        description="Whether to skip duplicate files (same content hash)",
    )
    extract_gps: bool = Field(
        False,
        description=(
            "Whether to extract GPS coordinates from EXIF. "
            "Default False: family photos are intended as private and GPS "
            "can de-anonymize locations when reports are shared."
        ),
    )
    fallback_timestamp: str = Field(
        "file_mtime",
        description="Fallback timestamp source when EXIF missing",
    )
    workers: int = Field(
        4,
        ge=1,
        le=32,
        description=(
            "Number of concurrent worker threads used to decode images, "
            "compute hashes, and extract EXIF metadata. Ingestion is I/O- and "
            "decode-bound – Pillow and imagehash release the GIL during their "
            "C-level work so threads give near-linear speedup on 4–8 cores. "
            "Set to 1 to disable parallelism (useful when debugging)."
        ),
    )


FaceBackend = Literal["auto", "cpu", "cuda", "coreml"]


class ConfigFace(BaseModel):
    """Face layer configuration."""

    model_name: str = Field(
        "",
        description="InsightFace model name (opt-in, see config.yaml; verify license)",
    )
    detection_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for face detection (0.0-1.0)",
    )
    matching_threshold: float = Field(
        0.6,
        ge=0.0,
        le=1.0,
        description="Cosine similarity threshold for person matching (0.0-1.0)",
    )
    age_confidence_scale: float = Field(
        0.1,
        ge=0.0,
        le=1.0,
        description="Scale factor for age estimation standard deviation",
    )
    backend: FaceBackend = Field(
        "auto",
        description=(
            "ONNX Runtime execution backend for InsightFace. "
            "'auto' picks CoreML on arm64 macOS and CPU elsewhere. "
            "'coreml' uses the Apple Neural Engine; 'cuda' needs an NVIDIA GPU; "
            "'cpu' is always available."
        ),
    )
    use_gpu: bool | None = Field(
        None,
        description=(
            "Deprecated. Prefer 'backend'. If set to true and 'backend' is 'auto', "
            "backend is upgraded to 'cuda' for backward compatibility."
        ),
    )
    batch_size: int = Field(
        1,
        ge=1,
        le=64,
        description="Batch size for face detection (higher values may improve GPU utilization)",
    )

    @model_validator(mode="after")
    def _migrate_use_gpu(self) -> "ConfigFace":
        """Soft-migration of legacy ``use_gpu: true`` configs to ``backend: 'cuda'``."""
        if self.use_gpu is True and self.backend == "auto":
            self.backend = "cuda"
        # Keep the value around so users can still read their config back; we
        # only normalise it when it would conflict with an explicit backend.
        return self


class ConfigContext(BaseModel):
    """Context and runtime configuration."""

    ollama_host: str = Field(
        "http://localhost:11434",
        description="Ollama server URL",
    )
    ollama_timeout: int = Field(
        300,
        ge=1,
        description="Timeout in seconds for Ollama requests",
    )
    max_retries: int = Field(
        3,
        ge=0,
        description="Maximum retry attempts for LLM failures",
    )
    retry_delay: float = Field(
        2.0,
        ge=0.0,
        description="Delay between retries in seconds",
    )
    primary_model: str = Field(
        "",
        description="Primary vision LLM model (opt-in, see config.yaml; verify license)",
    )
    fallback_model: str = Field(
        "",
        description="Fallback vision model (opt-in, see config.yaml; verify license)",
    )
    batch_size: int = Field(
        1,
        ge=1,
        description="Batch size for processing images",
    )
    min_decade_confidence: float = Field(
        0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for decade estimates",
    )
    min_season_confidence: float = Field(
        0.4,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for season estimates",
    )
    use_fallback_on_failure: bool = Field(
        True,
        description="Use fallback strategies on analysis failure",
    )
    store_minimal_on_complete_failure: bool = Field(
        True,
        description="Store minimal data when analysis completely fails",
    )
    keep_alive: str = Field(
        "30m",
        description=(
            "Ollama `keep_alive` — how long to keep the model in memory between "
            "requests. Accepts a duration string ('30s', '10m', '1h') or '-1' "
            "for 'forever'. Setting this prevents expensive model reloads "
            "between photos on resource-constrained machines."
        ),
    )
    num_ctx: int = Field(
        2048,
        ge=512,
        le=32768,
        description=(
            "Ollama `options.num_ctx` — context window size (tokens). "
            "Lowering reduces Metal/GPU memory pressure and speeds up inference "
            "on Apple Silicon; raise only if prompts/outputs get truncated."
        ),
    )
    num_gpu: int = Field(
        -1,
        ge=-1,
        description=(
            "Ollama `options.num_gpu` — number of layers to offload to GPU. "
            "-1 = auto (Ollama decides). On Apple Silicon this means all layers "
            "on Metal. Set 0 to force CPU."
        ),
    )
    model_options: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description=(
            "Per-model override of Ollama options. Keys match the model name "
            "(e.g. 'moondream2'); values are partial options dicts that shadow "
            "the globals above. Supports 'keep_alive' as a special key."
        ),
    )
    memory_warning_threshold_mb: int = Field(
        100,
        ge=10,
        le=10000,
        description="Memory warning threshold in MB. Log warning if available memory falls below this.",
    )
    memory_critical_threshold_mb: int = Field(
        50,
        ge=10,
        le=10000,
        description="Memory critical threshold in MB. Skip batch processing if available memory falls below this.",
    )
    memory_retry_delay_seconds: int = Field(
        30,
        ge=1,
        le=300,
        description="Delay in seconds to wait when memory is critically low before retrying batch processing.",
    )

    @model_validator(mode="after")
    def validate_memory_thresholds(self) -> "ConfigContext":
        """Validate that memory_critical_threshold_mb < memory_warning_threshold_mb."""
        if self.memory_critical_threshold_mb >= self.memory_warning_threshold_mb:
            raise ValueError(
                f"memory_critical_threshold_mb ({self.memory_critical_threshold_mb}) "
                f"must be less than memory_warning_threshold_mb ({self.memory_warning_threshold_mb})"
            )
        return self


class ConfigLogging(BaseModel):
    """Logging configuration."""

    level: str = Field(
        "INFO",
        description="Log level for console output (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    file_path: str | None = Field(
        ".photochron/logs/photochron.log",
        description="Path to log file. Set to null to disable file logging.",
    )
    file_level: str = Field(
        "DEBUG",
        description="Log level for file output (typically more verbose than console)",
    )
    rotation: str = Field(
        "10 MB",
        description="Log file rotation trigger (e.g. '10 MB', '1 day', '00:00')",
    )
    retention: str = Field(
        "7 days",
        description="How long to keep rotated log files",
    )
    json_format: bool = Field(
        False,
        description="Emit logs as JSON (machine-readable). Applies to file sink.",
    )
    backtrace: bool = Field(
        True,
        description="Show extended tracebacks with variable values on exceptions",
    )
    diagnose: bool = Field(
        False,
        description="Show variable values in tracebacks. Set False in production to avoid leaking sensitive data.",
    )

    @model_validator(mode="after")
    def validate_levels(self) -> "ConfigLogging":
        """Validate log level strings."""
        valid = {"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"}
        for field in ("level", "file_level"):
            value = getattr(self, field).upper()
            if value not in valid:
                raise ValueError(f"{field} must be one of {sorted(valid)}, got '{value}'")
            setattr(self, field, value)
        return self


class Config(BaseModel):
    """Root configuration model."""

    version: str = Field("1.0", description="Configuration schema version")
    paths: ConfigPaths = Field(default_factory=ConfigPaths)
    models: ConfigModels = Field(default_factory=ConfigModels)
    ingestion: ConfigIngestion = Field(default_factory=ConfigIngestion)
    face: ConfigFace = Field(default_factory=ConfigFace)
    pipeline: ConfigPipeline = Field(default_factory=ConfigPipeline)
    context: ConfigContext = Field(default_factory=ConfigContext)
    logging: ConfigLogging = Field(default_factory=ConfigLogging)

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
    )

    @property
    def cache_dir(self) -> str:
        """Convenience accessor; delegates to paths.cache_dir."""
        return self.paths.cache_dir

    @property
    def output_dir(self) -> str:
        """Convenience accessor; delegates to paths.output_dir."""
        return self.paths.output_dir
