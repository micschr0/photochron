"""
Ollama client wrapper for vision LLM integration.

This module provides a client for interacting with the local Ollama server
for vision LLM analysis of photos.
"""

import base64
import json
import math
import os
import random
import re
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

import ollama
from loguru import logger
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

# Maximum file size for base64 encoding (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB in bytes

# Maximum base64 string size (base64 encoding adds ~33% overhead)
MAX_BASE64_SIZE = int(MAX_FILE_SIZE * 4 / 3) + 100  # Add padding for safety


class ModelType(StrEnum):
    """Supported vision LLM models."""

    LLAVA_NEXT_7B = "llava-next:7b"
    MOONDREAM2 = "moondream2"


class ContextAnalysisResult(BaseModel):
    """Structured result from context analysis."""

    decade: str | None = Field(None, description="Estimated decade range (e.g., '1985-1990')")
    decade_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in decade estimate (0.0-1.0)")
    season: Literal["spring", "summer", "autumn", "winter"] | None = Field(
        None, description="Season: 'spring', 'summer', 'autumn', 'winter'"
    )
    event_hint: str | None = Field(None, description="Event hint (e.g., 'wedding', 'birthday', 'graduation')")
    photo_medium: Literal["digital", "print_scan", "polaroid", "film_negative", "unknown"] = Field(
        "digital",
        description="Photo medium: 'digital', 'print_scan', 'polaroid', 'film_negative', 'unknown'",
    )
    photo_medium_confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in photo medium estimate (0.0-1.0)",
    )
    visual_evidence: list[str] | None = Field(
        None, description="List of specific visual cues that informed the analysis"
    )
    season_confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence in season estimate (0.0-1.0)")
    event_confidence: float | None = Field(None, ge=0.0, le=1.0, description="Confidence in event hint (0.0-1.0)")
    alternative_decades: list[str] | None = Field(None, description="Alternative decade possibilities when uncertain")
    uncertainty_flag: bool | None = Field(None, description="Flag indicating high uncertainty in analysis")
    hypothesis_notes: str | None = Field(None, description="Explanation when multiple hypotheses exist")

    @field_validator("decade")
    @classmethod
    def validate_decade_format(cls, v: str | None) -> str | None:
        """Validate decade format matches pattern like '1985-1990'."""
        if v is None:
            return v

        # Check format: YYYY-YYYY where both are 4-digit years
        pattern = r"^\d{4}-\d{4}$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid decade format: '{v}'. Expected format: 'YYYY-YYYY' (e.g., '1985-1990')")

        # Validate year ranges
        try:
            start_year, end_year = map(int, v.split("-"))
            if start_year >= end_year:
                raise ValueError(f"Invalid decade range: '{v}'. Start year must be less than end year")
            if end_year - start_year > 20:  # Allow some flexibility but not too much
                raise ValueError(f"Invalid decade range: '{v}'. Range should be reasonable (max 20 years)")
            if start_year < 1800 or end_year > 2100:  # Reasonable bounds for photography
                raise ValueError(f"Invalid decade range: '{v}'. Years should be between 1800-2100")
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError(f"Invalid decade format: '{v}'. Years must be numeric")
            raise

        return v

    @field_validator("alternative_decades")
    @classmethod
    def validate_alternative_decades(cls, v: list[str] | None, info) -> list[str] | None:
        """Validate alternative decades format."""
        if v is None:
            return v

        validated_decades = []
        for decade in v:
            if decade is None:
                continue
            # Reuse the decade validator logic
            try:
                # Check format
                pattern = r"^\d{4}-\d{4}$"
                if not re.match(pattern, decade):
                    raise ValueError(f"Invalid alternative decade format: '{decade}'. Expected format: 'YYYY-YYYY'")

                # Validate year ranges
                start_year, end_year = map(int, decade.split("-"))
                if start_year >= end_year:
                    raise ValueError(
                        f"Invalid alternative decade range: '{decade}'. Start year must be less than end year"
                    )
                if end_year - start_year > 20:
                    raise ValueError(
                        f"Invalid alternative decade range: '{decade}'. Range should be reasonable (max 20 years)"
                    )
                if start_year < 1800 or end_year > 2100:
                    raise ValueError(f"Invalid alternative decade range: '{decade}'. Years should be between 1800-2100")

                validated_decades.append(decade)
            except ValueError as e:
                # Log warning but don't fail entire validation for one bad alternative
                logger.warning(f"Skipping invalid alternative decade '{decade}': {e}")

        return validated_decades if validated_decades else None

    @field_validator("season_confidence")
    @classmethod
    def validate_season_confidence(cls, v: float | None, info) -> float | None:
        """Validate season confidence is provided when season is set."""
        if info.data.get("season") is not None and v is None:
            logger.warning("Season is set but season_confidence is None. Consider providing confidence score.")
        elif info.data.get("season") is None and v is not None:
            logger.warning("season_confidence is set but season is None. Confidence without season may be ignored.")
        return v

    @field_validator("event_confidence")
    @classmethod
    def validate_event_confidence(cls, v: float | None, info) -> float | None:
        """Validate event confidence is provided when event_hint is set."""
        if info.data.get("event_hint") is not None and v is None:
            logger.warning("event_hint is set but event_confidence is None. Consider providing confidence score.")
        elif info.data.get("event_hint") is None and v is not None:
            logger.warning(
                "event_confidence is set but event_hint is None. Confidence without event hint may be ignored."
            )
        return v

    @field_validator("photo_medium_confidence")
    @classmethod
    def validate_photo_medium_confidence(cls, v: float | None, info) -> float | None:
        """Validate photo medium confidence is provided when photo_medium is not 'unknown'."""
        photo_medium = info.data.get("photo_medium", "digital")
        if photo_medium != "unknown" and v is None:
            logger.warning(
                f"photo_medium is '{photo_medium}' but photo_medium_confidence is None. "
                "Consider providing confidence score."
            )
        elif photo_medium == "unknown" and v is not None:
            logger.warning(
                "photo_medium_confidence is set but photo_medium is 'unknown'. "
                "Confidence for 'unknown' medium may be ignored."
            )
        return v

    @model_validator(mode="after")
    def validate_cross_field_logic(self) -> "ContextAnalysisResult":
        """Validate cross-field logic and clean up uncertain fields."""
        # Clear season if confidence is too low or None
        if self.season is not None:
            if self.season_confidence is None or self.season_confidence < 0.3:
                logger.debug(
                    f"Clearing season '{self.season}' due to "
                    f"{'missing' if self.season_confidence is None else 'low'} "
                    f"confidence: {self.season_confidence}"
                )
                self.season = None
                self.season_confidence = None

        # Clear event_hint if confidence is too low or None
        if self.event_hint is not None:
            if self.event_confidence is None or self.event_confidence < 0.3:
                logger.debug(
                    f"Clearing event_hint '{self.event_hint}' due to "
                    f"{'missing' if self.event_confidence is None else 'low'} "
                    f"confidence: {self.event_confidence}"
                )
                self.event_hint = None
                self.event_confidence = None

        # Set photo_medium to "unknown" if confidence is too low or None
        if self.photo_medium != "unknown":
            if self.photo_medium_confidence is None or self.photo_medium_confidence < 0.3:
                logger.debug(
                    f"Setting photo_medium to 'unknown' due to "
                    f"{'missing' if self.photo_medium_confidence is None else 'low'} "
                    f"confidence: {self.photo_medium_confidence}"
                )
                self.photo_medium = "unknown"
                self.photo_medium_confidence = None

        # Set uncertainty flag based on low confidence
        if self.decade_confidence < 0.4:
            self.uncertainty_flag = True
            # If decade confidence is very low, consider clearing decade
            if self.decade_confidence < 0.2 and self.decade is not None:
                logger.debug(f"Clearing decade '{self.decade}' due to very low confidence: {self.decade_confidence}")
                self.decade = None
                self.alternative_decades = None

        # Ensure visual_evidence is a list if not None
        if self.visual_evidence is not None and not isinstance(self.visual_evidence, list):
            self.visual_evidence = [self.visual_evidence] if self.visual_evidence else None

        # Ensure alternative_decades is a list if not None
        if self.alternative_decades is not None and not isinstance(self.alternative_decades, list):
            self.alternative_decades = [self.alternative_decades] if self.alternative_decades else None

        return self


@dataclass
class OllamaConfig:
    """Configuration for Ollama client."""

    host: str = "http://localhost:11434"
    timeout: int = 300  # 5 minutes for vision LLM
    max_retries: int = 3
    retry_delay: float = 2.0  # seconds
    jitter_percentage: float = 0.2  # ±20% random variation for retry delays
    primary_model: ModelType = ModelType.LLAVA_NEXT_7B
    fallback_model: ModelType = ModelType.MOONDREAM2
    # Performance / Apple-Silicon tuning – mirrored from ConfigContext.
    keep_alive: str = "30m"
    num_ctx: int = 2048
    num_gpu: int = -1
    model_options: dict[str, dict[str, Any]] = field(default_factory=dict)
    heartbeat_interval: float = 5.0  # seconds between "still working" log lines


@contextmanager
def _heartbeat(message: str, interval: float) -> Iterator[None]:
    """Emit periodic "still working" log lines while a blocking call runs.

    The Ollama Python SDK's `generate` is synchronous – on Apple Silicon a
    single llava-next:7b call can take several seconds. Without feedback the
    CLI appears frozen. A daemon thread ticks every ``interval`` seconds and
    reports the elapsed time so the user can see progress continuously.
    """
    if interval <= 0:
        yield
        return
    stop = threading.Event()
    start = time.monotonic()

    def _tick() -> None:
        while not stop.wait(interval):
            elapsed = time.monotonic() - start
            logger.info(f"{message} (still working, {elapsed:.1f}s elapsed)")

    thread = threading.Thread(target=_tick, name="ollama-heartbeat", daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval + 0.2)


class OllamaClient:
    """Client for interacting with Ollama server."""

    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()
        self._client = None
        self._available_models: list[str] = []

    def _resolve_generate_kwargs(self, model_name: str) -> tuple[dict[str, Any], str]:
        """Merge global Ollama defaults with per-model overrides.

        Returns the `options` dict suitable for ``ollama.generate`` together
        with the resolved ``keep_alive`` value (which is a top-level kwarg on
        the Ollama SDK, not an options field).
        """
        per_model = self.config.model_options.get(model_name, {})
        options: dict[str, Any] = {
            "temperature": 0.1,  # low for deterministic JSON
            "num_predict": 500,
            "num_ctx": self.config.num_ctx,
            "num_gpu": self.config.num_gpu,
        }
        # Per-model override shadows globals. 'keep_alive' is pulled out
        # because Ollama takes it at the top level.
        keep_alive = self.config.keep_alive
        for key, value in per_model.items():
            if key == "keep_alive":
                keep_alive = value
            else:
                options[key] = value
        return options, keep_alive

    def connect(self) -> bool:
        """Connect to Ollama server and check availability."""
        try:
            # Test connection by listing models
            response = ollama.list()
            self._available_models = [model["name"] for model in response.get("models", [])]

            logger.info(f"Connected to Ollama at {self.config.host}")
            logger.info(f"Available models: {self._available_models}")

            # Check if required models are available
            primary_available = self.config.primary_model.value in self._available_models
            fallback_available = self.config.fallback_model.value in self._available_models

            if not primary_available:
                logger.warning(f"Primary model {self.config.primary_model.value} not found")
            if not fallback_available:
                logger.warning(f"Fallback model {self.config.fallback_model.value} not found")

            return primary_available or fallback_available

        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.config.host}: {e}")
            return False

    def health_check(self) -> dict[str, Any]:
        """
        Perform a health check on the Ollama server.

        Returns:
            Dictionary with health status and details
        """
        try:
            # Get server info
            info = ollama.show(self.config.primary_model.value)

            health_status = {
                "status": "healthy",
                "server_available": True,
                "primary_model_available": self.is_model_available(self.config.primary_model.value),
                "fallback_model_available": self.is_model_available(self.config.fallback_model.value),
                "available_models": self._available_models,
                "model_details": {
                    "primary": {
                        "name": self.config.primary_model.value,
                        "available": self.is_model_available(self.config.primary_model.value),
                    },
                    "fallback": {
                        "name": self.config.fallback_model.value,
                        "available": self.is_model_available(self.config.fallback_model.value),
                    },
                },
            }

            # Add model info if available
            if info:
                health_status["model_info"] = {
                    "size": info.get("size", "unknown"),
                    "modified_at": info.get("modified_at", "unknown"),
                    "digest": info.get("digest", "unknown")[:16] + "..." if info.get("digest") else "unknown",
                }

            return health_status

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "unhealthy",
                "server_available": False,
                "error": str(e),
                "available_models": self._available_models,
            }

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        return model_name in self._available_models

    def validate_model(self, model_name: str) -> bool:
        """
        Validate that a model exists and can be loaded.

        Args:
            model_name: Name of the model to validate

        Returns:
            True if model is valid and can be loaded
        """
        if not self.is_model_available(model_name):
            return False

        try:
            # Try to get model info to verify it can be loaded
            ollama.show(model_name)
            return True
        except Exception as e:
            logger.warning(f"Model {model_name} exists but cannot be loaded: {e}")
            return False

    def _is_base64_encoded(self, data: str) -> bool:
        """
        Check if a string is base64 encoded.

        Args:
            data: String to check

        Returns:
            True if the string appears to be base64 encoded
        """
        try:
            # Check if string is not empty
            if not data:
                return False

            # Remove whitespace and newlines that might be added during transmission
            data = data.strip()

            # Check length before processing to prevent memory issues
            if len(data) > MAX_BASE64_SIZE:
                logger.debug(f"Base64 string too large: {len(data)} chars > {MAX_BASE64_SIZE} max")
                return False

            # Check for valid base64 characters (A-Z, a-z, 0-9, +, /, =, -, _)
            # URL-safe base64 uses - instead of + and _ instead of /
            # Base64 strings should only contain these characters
            # Note: - must be at beginning or end of character class to avoid being interpreted as range
            base64_pattern = re.compile(r"^[A-Za-z0-9+/\-_]*={0,2}$")
            if not base64_pattern.match(data):
                return False

            # Check length: base64 length should be multiple of 4 (after removing padding)
            # But we need to be careful with padding - actual data length can be different
            # The most reliable check is to try decoding
            # Try standard base64 first, then URL-safe if that fails
            try:
                base64.b64decode(data, validate=True)
                return True
            except (ValueError, TypeError, base64.binascii.Error):
                # Try URL-safe base64
                try:
                    base64.urlsafe_b64decode(data)
                    return True
                except (ValueError, TypeError, base64.binascii.Error):
                    return False
        except (ValueError, TypeError, base64.binascii.Error):
            return False

    def _prepare_image_input(self, image_input: str, use_base64: bool = False) -> str:
        """
        Prepare image input for Ollama API.

        Args:
            image_input: Either a file path or base64 encoded image string
            use_base64: If True, force encoding as base64 even for file paths

        Returns:
            Prepared image input (file path or base64 string)

        Raises:
            ValueError: If image_input is invalid or file doesn't exist
            IOError: If file cannot be read
        """
        # If already base64 encoded and not forcing re-encoding
        if not use_base64 and self._is_base64_encoded(image_input):
            logger.debug(f"Input is already base64 encoded (length: {len(image_input)} chars)")
            return image_input

        # If it's a file path
        if os.path.isfile(image_input):
            # Additional file validation
            try:
                # Check if file is readable
                if not os.access(image_input, os.R_OK):
                    raise OSError(f"File {image_input} is not readable")

                # Check file size before reading (best effort)
                file_size = os.path.getsize(image_input)
                if file_size == 0:
                    raise ValueError(f"File {image_input} is empty")
                if file_size > MAX_FILE_SIZE * 10:  # Allow some buffer for base64 overhead check
                    raise ValueError(
                        f"File {image_input} is too large ({file_size} bytes > {MAX_FILE_SIZE * 10} bytes max)"
                    )
            except (OSError, ValueError) as e:
                # Enhance error message while preserving original exception type and attributes
                # Create a new exception with enhanced message, chained to the original
                # This preserves all attributes of the original exception through the chain
                error_msg = f"Invalid image file {image_input}: {e}"
                if isinstance(e, OSError):
                    # For OSError and subclasses, create new exception with same type
                    new_exc = type(e)(error_msg)
                else:
                    # For ValueError, create new ValueError
                    new_exc = ValueError(error_msg)
                raise new_exc from e

            if use_base64:
                # Read file and encode as base64
                try:
                    with open(image_input, "rb") as f:
                        image_data = f.read()

                    # Check file size after reading to avoid race condition
                    file_size = len(image_data)
                    if file_size > MAX_FILE_SIZE:
                        raise ValueError(
                            f"File {image_input} is too large ({file_size} bytes > {MAX_FILE_SIZE} bytes max). "
                            f"Consider using file path instead of base64 encoding for large files."
                        )

                    logger.debug(f"Encoding file {image_input} ({file_size} bytes) as base64")
                    return base64.b64encode(image_data).decode("utf-8")
                except OSError as e:
                    raise OSError(f"Cannot read image file {image_input}: {e}")
            else:
                # Return file path as-is
                logger.debug(f"Using file path directly: {image_input}")
                return image_input

        # Check if it's base64 encoded when use_base64=True
        if use_base64 and self._is_base64_encoded(image_input):
            logger.debug(f"Input is already base64 encoded (length: {len(image_input)} chars), using as-is")
            return image_input

        # Check if it might be base64 but validation failed
        if use_base64:
            # Try to provide more helpful error message
            if self._is_base64_encoded(image_input):
                # This shouldn't happen, but just in case
                raise ValueError("Input validation inconsistent: detected as base64 but failed to process")
            else:
                # Safely get first 100 chars or less
                preview = image_input[:100] if len(image_input) > 100 else image_input
                raise ValueError(
                    f"Input is not valid base64 when use_base64=True. First {len(preview)} chars: {preview}..."
                )

        # Not a file and not base64
        raise ValueError(f"Invalid image input: {image_input}. Must be a valid file path or base64 encoded string.")

    def analyze_image_context(
        self,
        image_input: str,
        model_name: str | None = None,
        prompt_template: str | None = None,
        use_base64: bool = False,
    ) -> ContextAnalysisResult | None:
        """
        Analyze image context using vision LLM.

        Args:
            image_input: Path to the image file or base64 encoded image string
            model_name: Name of the model to use (defaults to primary model)
            prompt_template: Custom prompt template (uses default if None)
            use_base64: If True, encode image as base64 before sending to Ollama

        Returns:
            ContextAnalysisResult if successful, None otherwise
        """
        model_name = model_name or self.config.primary_model.value

        if not self.is_model_available(model_name):
            logger.warning(f"Model {model_name} not available, trying fallback")
            model_name = self.config.fallback_model.value

            if not self.is_model_available(model_name):
                logger.error(f"Fallback model {model_name} also not available")
                return None

        # Prepare image input (file path or base64)
        try:
            prepared_image = self._prepare_image_input(image_input, use_base64)
        except (OSError, ValueError) as e:
            logger.error(f"Failed to prepare image input: {e}")
            return None

        prompt = prompt_template or self._get_default_prompt()

        # Create log message for image info
        log_image_info = f"image (base64: {use_base64})" if use_base64 else f"image {image_input}"

        # Simple circuit breaker state
        consecutive_timeouts = 0
        max_consecutive_timeouts = 2

        options, keep_alive = self._resolve_generate_kwargs(model_name)

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Analyzing {log_image_info} with model {model_name} (attempt {attempt + 1})")

                with _heartbeat(
                    f"Analyzing {log_image_info} via {model_name}",
                    interval=self.config.heartbeat_interval,
                ):
                    response = ollama.generate(
                        model=model_name,
                        prompt=prompt,
                        images=[prepared_image],
                        options=options,
                        keep_alive=keep_alive,
                    )

                result_text = response.get("response", "").strip()
                logger.debug(f"Raw LLM response: {result_text}")

                # Try to parse JSON from response
                result = self._parse_llm_response(result_text)
                if result:
                    logger.info(f"Successfully analyzed {log_image_info}")
                    # Reset circuit breaker on success
                    consecutive_timeouts = 0
                    return result

                logger.warning(f"Failed to parse JSON from LLM response on attempt {attempt + 1}")

            except (
                TimeoutError,
                ConnectionError,
                ConnectionRefusedError,
                OSError,
            ) as e:
                # Network/timeout specific error handling
                # OSError catches socket.timeout and other OS-level errors
                consecutive_timeouts += 1
                logger.warning(f"Network/timeout error analyzing image on attempt {attempt + 1}: {e}")

                # Check circuit breaker
                if consecutive_timeouts >= max_consecutive_timeouts:
                    logger.error(
                        f"Circuit breaker triggered: {consecutive_timeouts} consecutive timeouts. "
                        f"Aborting retries for {log_image_info}"
                    )
                    break

            except Exception as e:
                # Other exceptions (non-network/timeout)
                logger.error(f"Error analyzing image on attempt {attempt + 1}: {e}")
                # Reset circuit breaker for non-timeout errors
                consecutive_timeouts = 0

            # Exponential backoff for retries with jitter
            if attempt < self.config.max_retries - 1 and consecutive_timeouts < max_consecutive_timeouts:
                base_delay = self.config.retry_delay * (2**attempt)

                # Add jitter: ±jitter_percentage random variation
                jitter_range = base_delay * self.config.jitter_percentage
                jitter = random.uniform(-jitter_range, jitter_range)
                delay = max(0.1, base_delay + jitter)  # Ensure minimum delay of 0.1 seconds

                logger.debug(f"Retrying in {delay:.2f} seconds (base: {base_delay:.2f}, jitter: {jitter:+.2f})...")
                time.sleep(delay)

        logger.error(f"Failed to analyze {log_image_info} after {self.config.max_retries} attempts")
        return None

    def _get_default_prompt(self) -> str:
        """Get the default structured prompt for context analysis."""
        return self._get_prompt_template("default")

    def _get_prompt_template(self, template_name: str) -> str:
        """Get a specific prompt template by name."""
        templates = {
            "default": """Analyze this photo and provide structured JSON output. Focus on visual anchors:

VISUAL ANCHORS TO CONSIDER:
1. Fashion: Clothing styles, hairstyles, accessories
2. Technology: Vehicles, electronics, appliances
3. Architecture: Building styles, interior design, furniture
4. Media quality: Photo grain, color palette, aspect ratio
5. Cultural markers: Logos, brands, signage, publications

Provide JSON with these fields:
- decade: Estimated decade range (e.g., "1985-1990", "1990-1995"). Use null if uncertain.
- decade_confidence: Confidence score 0.0-1.0 based on clarity of visual anchors
- season: "spring", "summer", "autumn", "winter", or null
- event_hint: "wedding", "birthday", "graduation", "holiday", "family_gathering", or null
- photo_medium: "digital", "print_scan", "polaroid", "film_negative", "unknown"
- photo_medium_confidence: Confidence in photo medium estimate 0.0-1.0 (optional)
- visual_evidence: List of specific visual anchors that informed your analysis
- season_confidence: Confidence in season estimate 0.0-1.0 (optional)
- event_confidence: Confidence in event hint 0.0-1.0 (optional)

Provide ONLY valid JSON output with no additional text. Example:
{
  "decade": "1985-1990",
  "decade_confidence": 0.82,
  "season": "summer",
  "season_confidence": 0.7,
  "event_hint": null,
  "event_confidence": null,
  "photo_medium": "print_scan",
  "photo_medium_confidence": 0.8,
  "visual_evidence": ["bell-bottom jeans", "large collar shirt", "1970s car model"]
}""",
            "detailed_decade": """Analyze this historical photo and estimate the decade it was taken with high precision.

Consider these visual cues:
1. Fashion and clothing styles
2. Technology and vehicles
3. Architecture and interior design
4. Photo quality and color palette
5. Hairstyles and accessories

Provide structured JSON output with:
- decade: Estimated decade range (e.g., "1975-1980", "1985-1990")
- decade_confidence: Confidence score 0.0-1.0 based on clarity of visual cues
- visual_evidence: List of specific visual cues that informed your estimate
- season: "spring", "summer", "autumn", "winter", or null
- season_confidence: Confidence in season estimate 0.0-1.0 (optional)
- photo_medium: "digital", "print_scan", "polaroid", "film_negative", "unknown"
- photo_medium_confidence: Confidence in photo medium estimate 0.0-1.0 (optional)

Example:
{
  "decade": "1978-1982",
  "decade_confidence": 0.82,
  "visual_evidence": ["bell-bottom jeans", "large collar shirts", "1970s car model"],
  "season": "summer",
  "season_confidence": 0.6,
  "photo_medium": "print_scan",
  "photo_medium_confidence": 0.7
}""",
            "season_focused": """Analyze this photo to determine the season depicted.

Look for:
- Vegetation: blooming flowers (spring), green leaves (summer), colorful leaves (autumn), bare trees (winter)
- Weather: snow (winter), rain (spring/autumn), bright sun (summer)
- Clothing: heavy coats (winter), light clothing (summer), transitional wear (spring/autumn)
- Activities: beach (summer), skiing (winter), leaf raking (autumn)

Provide structured JSON output with:
- season: "spring", "summer", "autumn", "winter", or null if indoor/uncertain
- season_confidence: Confidence 0.0-1.0
- visual_evidence: List of specific visual cues for season
- decade: Estimated decade if discernible, otherwise null
- decade_confidence: Confidence in decade estimate 0.0-1.0 (optional)
- photo_medium: Photo type
- photo_medium_confidence: Confidence in photo medium estimate 0.0-1.0 (optional)

Example:
{
  "season": "winter",
  "season_confidence": 0.9,
  "visual_evidence": ["snow on ground", "heavy winter coats", "bare trees"],
  "decade": "1990-1995",
  "decade_confidence": 0.7,
  "photo_medium": "digital",
  "photo_medium_confidence": 0.8
}""",
            "event_detection": """Analyze this photo for event indicators.

Look for:
- Special clothing: wedding dresses, graduation gowns, birthday hats
- Decorations: balloons, banners, cake, flowers
- Group activities: ceremonies, parties, celebrations
- Props: gifts, trophies, certificates

Provide structured JSON output with:
- event_hint: "wedding", "birthday", "graduation", "holiday", "family_gathering", or null
- event_confidence: Confidence 0.0-1.0
- visual_evidence: List of specific visual indicators for event
- decade: Estimated decade if discernible
- decade_confidence: Confidence in decade estimate 0.0-1.0 (optional)
- season: Season if discernible
- season_confidence: Confidence in season estimate 0.0-1.0 (optional)
- photo_medium: Photo type
- photo_medium_confidence: Confidence in photo medium estimate 0.0-1.0 (optional)

Example:
{
  "event_hint": "wedding",
  "event_confidence": 0.85,
  "visual_evidence": ["white wedding dress", "bouquet", "formal attire"],
  "decade": "1985-1990",
  "decade_confidence": 0.8,
  "season": "summer",
  "season_confidence": 0.7,
  "photo_medium": "print_scan",
  "photo_medium_confidence": 0.9
}""",
            "simple_fallback": """Analyze this photo briefly.

Provide JSON with:
- decade: Rough estimate or null
- decade_confidence: 0.0 if null, otherwise low confidence
- season: null
- event_hint: null
- photo_medium: Guess or "unknown"
- photo_medium_confidence: Confidence in photo medium estimate 0.0-1.0 (optional)
- visual_evidence: null or empty list (optional)
- season_confidence: null (optional)
- event_confidence: null (optional)

Keep it simple. Example:
{
  "decade": null,
  "decade_confidence": 0.0,
  "season": null,
  "event_hint": null,
  "photo_medium": "unknown",
  "photo_medium_confidence": null,
  "visual_evidence": null,
  "season_confidence": null,
  "event_confidence": null
}""",
            "uncertainty_handling": """Analyze this photo when visual evidence is ambiguous or unclear.

Use this template when:
- Image quality is poor (blurry, dark, low resolution)
- Key visual anchors are obscured or missing
- Multiple time periods could be plausible
- You have low confidence in your estimates

Provide JSON with:
- decade: Best estimate or null if too uncertain
- decade_confidence: Low confidence score (0.0-0.4 range expected)
- uncertainty_flag: true (indicates high uncertainty)
- visual_evidence: List of any visual anchors you can identify, or null
- alternative_decades: List of alternative decade possibilities
- season: null or best guess with low confidence
- season_confidence: Low if provided
- photo_medium: "unknown" or best guess
- photo_medium_confidence: Confidence in photo medium estimate 0.0-1.0 (optional)

Example for uncertain case:
{
  "decade": "1990-1995",
  "decade_confidence": 0.3,
  "uncertainty_flag": true,
  "visual_evidence": ["blurry clothing", "indoor setting"],
  "alternative_decades": ["1985-1990", "1995-2000"],
  "season": null,
  "season_confidence": null,
  "event_hint": null,
  "event_confidence": null,
  "photo_medium": "unknown",
  "photo_medium_confidence": 0.2
}""",
            "multi_hypothesis": """Analyze this photo and provide multiple decade hypotheses when evidence supports multiple possibilities.

Use this when:
- Fashion/styles span multiple decades
- Mixed technology from different eras
- Architecture has transitional features
- Photo has ambiguous time markers

Provide JSON with:
- decade: Primary/best estimate (most likely)
- decade_confidence: Confidence in primary estimate
- alternative_decades: List of alternative decade possibilities
- visual_evidence: List of visual anchors supporting each hypothesis
- hypothesis_notes: Brief explanation of why multiple hypotheses exist
- season: Best estimate
- photo_medium: Best estimate
- photo_medium_confidence: Confidence in photo medium estimate 0.0-1.0 (optional)

Example with multiple hypotheses:
{
  "decade": "1975-1980",
  "decade_confidence": 0.6,
  "alternative_decades": ["1970-1975", "1980-1985"],
  "visual_evidence": ["transitional fashion", "mixed furniture styles", "ambiguous car model"],
  "hypothesis_notes": "Fashion suggests late 70s but technology could be early 80s",
  "season": "summer",
  "season_confidence": 0.7,
  "event_hint": null,
  "photo_medium": "print_scan",
  "photo_medium_confidence": 0.8
}""",
        }

        return templates.get(template_name, templates["default"])

    def get_available_prompts(self) -> list[str]:
        """Get list of available prompt template names."""
        return [
            "default",
            "detailed_decade",
            "season_focused",
            "event_detection",
            "simple_fallback",
            "uncertainty_handling",
            "multi_hypothesis",
        ]

    def get_prompt_template(self, template_name: str) -> str:
        """Get a specific prompt template by name.

        Args:
            template_name: Name of the prompt template to retrieve.

        Returns:
            The prompt template string. Returns the default template if the
            requested template is not found.

        Raises:
            ValueError: If template_name is empty or None.
        """
        if not template_name or not template_name.strip():
            logger.error("Cannot get prompt template: template_name is empty")
            raise ValueError("template_name cannot be empty")

        template = self._get_prompt_template(template_name)

        # Check if we got the default template (which means the requested template wasn't found)
        default_template = self._get_prompt_template("default")
        if template == default_template and template_name != "default":
            logger.warning(f"Prompt template '{template_name}' not found, using default template")
        else:
            logger.debug(f"Using prompt template: {template_name}")

        return template

    def _parse_llm_response(self, response_text: str) -> ContextAnalysisResult | None:
        """Parse LLM response into structured result with enhanced error handling."""
        if not response_text or not response_text.strip():
            logger.warning("Empty LLM response received")
            return None

        response_text = response_text.strip()

        try:
            # Try to extract JSON from response (LLM might add extra text)
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]

                # Try to parse JSON with better error messages
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    # Provide more helpful error message
                    error_context = json_str[max(0, e.pos - 50) : min(len(json_str), e.pos + 50)]
                    logger.warning(f"JSON decode error at position {e.pos}: {e.msg}. Context: ...{error_context}...")

                    # Try to fix common JSON issues
                    fixed_json = self._attempt_json_fix(json_str)
                    if fixed_json:
                        try:
                            data = json.loads(fixed_json)
                            logger.debug("Successfully fixed JSON parsing issue")
                        except json.JSONDecodeError:
                            return None
                    else:
                        return None

                # Validate with Pydantic model
                try:
                    result = ContextAnalysisResult(**data)
                    logger.debug("Successfully parsed and validated LLM response")
                    return result
                except ValidationError as e:
                    # Log detailed validation errors
                    error_details = []
                    for error in e.errors():
                        field = ".".join(str(loc) for loc in error.get("loc", []))
                        msg = error.get("msg", "Unknown error")
                        error_type = error.get("type", "validation")
                        error_details.append(f"{field}: {msg} ({error_type})")

                    logger.warning(f"Validation failed for LLM response: {', '.join(error_details)}. Raw data: {data}")

                    # Try to create a fallback result with cleaned data
                    return self._create_fallback_result(data)
            else:
                logger.warning(f"No JSON found in response. First 200 chars: {response_text[:200]}...")
                return None

        except Exception as e:
            logger.error(f"Unexpected error parsing LLM response: {e}")
            return None

    def _attempt_json_fix(self, json_str: str) -> str | None:
        """Attempt to fix common JSON issues in LLM responses."""
        if not json_str:
            return None

        fixed = json_str

        # Fix 1: Remove trailing commas
        fixed = re.sub(r",\s*}", "}", fixed)
        fixed = re.sub(r",\s*]", "]", fixed)

        # Fix 2: Fix unquoted keys (common LLM mistake)
        # Match unquoted keys at start of object or after comma
        fixed = re.sub(r"([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:", r'\1"\2":', fixed)

        # Fix 3: Fix single quotes to double quotes
        fixed = re.sub(r"'([^']*)'", r'"\1"', fixed)

        # Fix 4: Fix escaped quotes inside strings - handle common LLM errors
        # LLMs sometimes output unescaped quotes inside strings like: "hello "world""
        # We need to escape the inner quotes but not the outer ones
        # This is a simplified heuristic approach
        try:
            # First, try to parse as-is to see if it's already valid
            json.loads(fixed)
            return fixed  # Already valid, no fix needed
        except json.JSONDecodeError:
            # Try to fix common quote escaping issues
            # Pattern: quote followed by non-whitespace, not preceded by backslash or colon/brace/comma
            # This is a heuristic and won't catch all cases
            pass

        # Fix 5: Remove control characters (except whitespace)
        fixed = "".join(char for char in fixed if ord(char) >= 32 or char in "\n\r\t")

        # Fix 6: Ensure proper null values
        fixed = re.sub(r":\s*null\b", ": null", fixed, flags=re.IGNORECASE)

        # Fix 7: Try a simple quote escaping fix for the most common case
        # Look for patterns like: "text "more text"" and escape the inner quotes
        # This regex looks for a quote that's inside a string (has a quote before it in the same string)
        # It's not perfect but handles many common LLM errors
        # We'll use a state machine approach instead of complex regex
        fixed = self._fix_unescaped_quotes(fixed)

        return fixed if fixed != json_str else None

    def _fix_unescaped_quotes(self, json_str: str) -> str:
        """
        Fix unescaped quotes inside JSON strings.

        This handles the common LLM error where quotes inside strings are not escaped.
        Example: "hello "world"" -> "hello \"world\""

        Args:
            json_str: JSON string to fix

        Returns:
            Fixed JSON string
        """
        if not json_str:
            return json_str

        result = []
        i = 0
        in_string = False
        escape_next = False

        while i < len(json_str):
            char = json_str[i]

            if escape_next:
                # Current character is escaped, add it as-is
                result.append(char)
                escape_next = False
                i += 1
                continue

            if char == "\\":
                # Next character is escaped
                result.append(char)
                escape_next = True
                i += 1
                continue

            if char == '"':
                if in_string:
                    # Check if this is the end of the string or an unescaped quote inside
                    # Look ahead to see what comes after
                    j = i + 1
                    while j < len(json_str) and json_str[j].isspace():
                        j += 1

                    if j < len(json_str) and json_str[j] in ",:[]{}":
                        # This is likely the end of the string
                        result.append(char)
                        in_string = False
                    else:
                        # This might be an unescaped quote inside the string
                        # Escape it
                        result.append("\\")
                        result.append(char)
                        # Stay in string
                else:
                    # Start of a string
                    result.append(char)
                    in_string = True
            else:
                result.append(char)

            i += 1

        return "".join(result)

    def _create_fallback_result(self, data: dict[str, Any]) -> ContextAnalysisResult | None:
        """Create a fallback result when validation fails."""
        try:
            # Clean the data before creating fallback
            cleaned_data = {}

            # Handle decade field
            decade = data.get("decade")
            if decade and isinstance(decade, str):
                # Try to validate decade format
                try:
                    if re.match(r"^\d{4}-\d{4}$", decade):
                        cleaned_data["decade"] = decade
                except Exception:
                    pass

            # Handle decade_confidence
            decade_confidence = data.get("decade_confidence")
            if isinstance(decade_confidence, (int, float)):
                confidence_value = float(decade_confidence)
                # Check for NaN using math.isnan()
                if math.isnan(confidence_value):
                    pass  # Skip NaN values
                else:
                    cleaned_data["decade_confidence"] = max(0.0, min(1.0, confidence_value))
            elif isinstance(decade_confidence, str):
                try:
                    confidence_value = float(decade_confidence)
                    # Check for NaN using math.isnan()
                    if math.isnan(confidence_value):
                        pass  # Skip NaN values
                    else:
                        cleaned_data["decade_confidence"] = max(0.0, min(1.0, confidence_value))
                except (ValueError, TypeError):
                    pass  # Skip invalid string values

            # Handle season (only allow valid values)
            season = data.get("season")
            if season in ["spring", "summer", "autumn", "winter"]:
                cleaned_data["season"] = season

            # Handle season_confidence
            season_confidence = data.get("season_confidence")
            if isinstance(season_confidence, (int, float)):
                confidence_value = float(season_confidence)
                # Check for NaN using math.isnan()
                if math.isnan(confidence_value):
                    pass  # Skip NaN values
                else:
                    cleaned_data["season_confidence"] = max(0.0, min(1.0, confidence_value))
            elif isinstance(season_confidence, str):
                try:
                    confidence_value = float(season_confidence)
                    # Check for NaN using math.isnan()
                    if math.isnan(confidence_value):
                        pass  # Skip NaN values
                    else:
                        cleaned_data["season_confidence"] = max(0.0, min(1.0, confidence_value))
                except (ValueError, TypeError):
                    pass  # Skip invalid string values

            # Handle event_hint
            event_hint = data.get("event_hint")
            if event_hint and isinstance(event_hint, str):
                cleaned_data["event_hint"] = event_hint

            # Handle event_confidence
            event_confidence = data.get("event_confidence")
            if isinstance(event_confidence, (int, float)):
                confidence_value = float(event_confidence)
                # Check for NaN using math.isnan()
                if math.isnan(confidence_value):
                    pass  # Skip NaN values
                else:
                    cleaned_data["event_confidence"] = max(0.0, min(1.0, confidence_value))
            elif isinstance(event_confidence, str):
                try:
                    confidence_value = float(event_confidence)
                    # Check for NaN using math.isnan()
                    if math.isnan(confidence_value):
                        pass  # Skip NaN values
                    else:
                        cleaned_data["event_confidence"] = max(0.0, min(1.0, confidence_value))
                except (ValueError, TypeError):
                    pass  # Skip invalid string values

            # Handle photo_medium (only allow valid values)
            photo_medium = data.get("photo_medium")
            if photo_medium in [
                "digital",
                "print_scan",
                "polaroid",
                "film_negative",
                "unknown",
            ]:
                cleaned_data["photo_medium"] = photo_medium

            # Handle photo_medium_confidence
            photo_medium_confidence = data.get("photo_medium_confidence")
            if isinstance(photo_medium_confidence, (int, float)):
                confidence_value = float(photo_medium_confidence)
                # Check for NaN using math.isnan()
                if math.isnan(confidence_value):
                    pass  # Skip NaN values
                else:
                    cleaned_data["photo_medium_confidence"] = max(0.0, min(1.0, confidence_value))
            elif isinstance(photo_medium_confidence, str):
                try:
                    confidence_value = float(photo_medium_confidence)
                    # Check for NaN using math.isnan()
                    if math.isnan(confidence_value):
                        pass  # Skip NaN values
                    else:
                        cleaned_data["photo_medium_confidence"] = max(0.0, min(1.0, confidence_value))
                except (ValueError, TypeError):
                    pass  # Skip invalid string values

            # Handle visual_evidence
            visual_evidence = data.get("visual_evidence")
            if visual_evidence:
                if isinstance(visual_evidence, list):
                    cleaned_data["visual_evidence"] = [str(item) for item in visual_evidence if item]
                elif isinstance(visual_evidence, str):
                    cleaned_data["visual_evidence"] = [visual_evidence]

            # Handle alternative_decades
            alternative_decades = data.get("alternative_decades")
            if alternative_decades:
                if isinstance(alternative_decades, list):
                    valid_decades = []
                    for decade in alternative_decades:
                        if decade and isinstance(decade, str) and re.match(r"^\d{4}-\d{4}$", decade):
                            valid_decades.append(decade)
                    if valid_decades:
                        cleaned_data["alternative_decades"] = valid_decades

            # Handle uncertainty_flag
            uncertainty_flag = data.get("uncertainty_flag")
            if isinstance(uncertainty_flag, bool):
                cleaned_data["uncertainty_flag"] = uncertainty_flag

            # Handle hypothesis_notes
            hypothesis_notes = data.get("hypothesis_notes")
            if hypothesis_notes and isinstance(hypothesis_notes, str):
                cleaned_data["hypothesis_notes"] = hypothesis_notes

            # Create fallback result with cleaned data
            if cleaned_data:
                logger.info(f"Created fallback result from cleaned data: {cleaned_data}")
                return ContextAnalysisResult(**cleaned_data)
            else:
                logger.warning("No valid data found for fallback result")
                return None

        except Exception as e:
            logger.error(f"Failed to create fallback result: {e}")
            return None

    def get_available_models(self) -> list[str]:
        """Get list of available models."""
        return self._available_models.copy()


# Default client instance
_default_client: OllamaClient | None = None


def get_ollama_client(config: OllamaConfig | None = None) -> OllamaClient:
    """Get or create default Ollama client."""
    global _default_client
    if _default_client is None:
        _default_client = OllamaClient(config)
        if not _default_client.connect():
            logger.warning("Ollama client failed to connect on initialization")
    return _default_client
