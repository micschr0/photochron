"""
Context layer stage: Analyze photo context using vision LLM.
"""

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

from photochron.config import get_config
from photochron.context.analyzer import ContextAnalyzer, ContextAnalyzerConfig
from photochron.models import ContextCreate
from photochron.models.ollama_client import ModelType, OllamaConfig
from photochron.pipeline import PipelineStage, register_stage
from photochron.store import get_store


@register_stage
class ContextLayerStage(PipelineStage):
    """Stage 3: Visual context analysis using LLM."""

    def __init__(self) -> None:
        """Initialize context layer stage with configuration."""
        self.config = get_config()
        self.context_config = self.config.context

        # Convert string model names to ModelType enum values
        try:
            primary_model_type = self._get_model_type(self.context_config.primary_model)
            fallback_model_type = self._get_model_type(self.context_config.fallback_model)
        except ValueError as e:
            logger.error(f"Invalid model configuration: {e}")
            # Set degraded mode immediately
            self._is_healthy = False
            self._degraded_mode = True
            # Use defaults for OllamaConfig to avoid initialization errors
            primary_model_type = ModelType.LLAVA_NEXT_7B
            fallback_model_type = ModelType.MOONDREAM2

        # Create Ollama configuration from context settings. The tuning
        # fields (keep_alive / num_ctx / num_gpu / model_options) feed the
        # Apple-Silicon-Metal path in ollama.generate(); without them llama.cpp
        # reloads the model between photos and ignores available Metal layers.
        ollama_config = OllamaConfig(
            host=self.context_config.ollama_host,
            timeout=self.context_config.ollama_timeout,
            max_retries=self.context_config.max_retries,
            retry_delay=self.context_config.retry_delay,
            primary_model=primary_model_type,
            fallback_model=fallback_model_type,
            keep_alive=self.context_config.keep_alive,
            num_ctx=self.context_config.num_ctx,
            num_gpu=self.context_config.num_gpu,
            model_options=dict(self.context_config.model_options),
        )

        # Create ContextAnalyzer configuration with model priority based on availability
        # We'll set model_priority after health check
        analyzer_config = ContextAnalyzerConfig(
            min_decade_confidence=self.context_config.min_decade_confidence,
            min_season_confidence=self.context_config.min_season_confidence,
            enable_retries=True,
            max_retries=self.context_config.max_retries,
        )

        # Initialize the context analyzer
        self.analyzer = ContextAnalyzer(
            ollama_config=ollama_config,
            config=analyzer_config,
        )

        # Initialize health status flags
        self._is_healthy = False
        self._degraded_mode = False
        self._available_models = {
            "primary": False,
            "fallback": False,
        }

        # Perform configuration validation and health check
        self._validate_configuration()

        logger.info("Initialized ContextLayerStage")

    def _get_model_type(self, model_name: str) -> ModelType:
        """
        Convert string model name to ModelType enum.

        Args:
            model_name: String model name (e.g., "llava-next:7b", "moondream2")

        Returns:
            ModelType enum value

        Raises:
            ValueError: If model name doesn't match any ModelType enum value
        """
        try:
            return ModelType(model_name)
        except ValueError:
            # Try case-insensitive match
            for model_type in ModelType:
                if model_type.value.lower() == model_name.lower():
                    return model_type
            raise ValueError(f"Unknown model name: {model_name}. Available models: {[m.value for m in ModelType]}")

    def _validate_configuration(self) -> None:
        """
        Validate configuration and check Ollama availability.

        Performs health check and model availability validation.
        Sets appropriate flags for graceful degradation.
        """
        try:
            # Perform health check
            health = self.analyzer.health_check()

            # Check Ollama server availability
            ollama_healthy = health.get("status") == "healthy"
            server_available = health.get("ollama_health", {}).get("server_available", False)

            if not ollama_healthy or not server_available:
                logger.warning(
                    f"Ollama server unavailable or unhealthy. "
                    f"Health status: {health.get('status')}, "
                    f"Server available: {server_available}"
                )
                self._degraded_mode = True
                self._is_healthy = False
                return

            # Check model availability from health check
            model_details = health.get("ollama_health", {}).get("model_details", {})
            primary_available = model_details.get("primary", {}).get("available", False)
            fallback_available = model_details.get("fallback", {}).get("available", False)

            self._available_models["primary"] = primary_available
            self._available_models["fallback"] = fallback_available

            # Determine overall health status and set model priority
            if primary_available or fallback_available:
                self._is_healthy = True
                self._degraded_mode = False

                # Update analyzer config model_priority based on availability
                model_priority = []

                if primary_available:
                    primary_model_type = self._get_model_type(self.context_config.primary_model)
                    model_priority.append(primary_model_type)
                    logger.info(f"Primary model '{self.context_config.primary_model}' is available")

                if fallback_available:
                    fallback_model_type = self._get_model_type(self.context_config.fallback_model)
                    # Only add fallback if it's not already in the list (in case primary == fallback)
                    if fallback_model_type not in model_priority:
                        model_priority.append(fallback_model_type)
                    logger.info(f"Fallback model '{self.context_config.fallback_model}' is available")

                # Update analyzer configuration
                self.analyzer.config.model_priority = model_priority

                if not primary_available and fallback_available:
                    logger.info(
                        f"Primary model '{self.context_config.primary_model}' not available, "
                        f"but fallback model '{self.context_config.fallback_model}' is available. "
                        f"Will use fallback model as primary."
                    )
                elif primary_available and not fallback_available:
                    logger.info(
                        f"Primary model '{self.context_config.primary_model}' available, "
                        f"but fallback model '{self.context_config.fallback_model}' not available."
                    )
                else:
                    logger.info(
                        f"Both primary model '{self.context_config.primary_model}' and "
                        f"fallback model '{self.context_config.fallback_model}' are available."
                    )
            else:
                logger.warning(
                    f"Neither primary model '{self.context_config.primary_model}' nor "
                    f"fallback model '{self.context_config.fallback_model}' are available. "
                    f"Entering degraded mode."
                )
                self._is_healthy = False
                self._degraded_mode = True

        except Exception as e:
            logger.warning(f"Configuration validation failed: {e}")
            logger.warning("Entering degraded mode due to validation failure")
            self._is_healthy = False
            self._degraded_mode = True

    @property
    def name(self) -> str:
        return "context_layer"

    @property
    def dependencies(self) -> list[str]:
        return ["face_layer"]  # Can use face info as context

    @property
    def health_status(self) -> dict[str, Any]:
        """Get current health status of the context analyzer."""
        return {
            "is_healthy": self._is_healthy,
            "degraded_mode": self._degraded_mode,
            "available_models": self._available_models,
        }

    def _check_memory_before_batch(self) -> dict[str, Any]:
        """
        Check available memory before processing a batch.

        Returns:
            Dict with keys:
            - status: "ok", "warning", "critical", "error", or "unknown"
            - available_mb: Available memory in MB (if psutil available)
            - message: Human-readable message (optional)
        """
        if not PSUTIL_AVAILABLE:
            logger.debug("psutil not available, skipping memory check")
            return {
                "status": "unknown",
                "available_mb": None,
                "message": "psutil not available",
            }

        try:
            available_memory = psutil.virtual_memory().available
            available_memory_mb = available_memory / (1024 * 1024)

            if available_memory_mb < self.context_config.memory_critical_threshold_mb:
                return {
                    "status": "critical",
                    "available_mb": available_memory_mb,
                    "message": (
                        f"Memory critically low: {available_memory_mb:.1f}MB < "
                        f"{self.context_config.memory_critical_threshold_mb}MB"
                    ),
                }
            elif available_memory_mb < self.context_config.memory_warning_threshold_mb:
                return {
                    "status": "warning",
                    "available_mb": available_memory_mb,
                    "message": (
                        f"Low memory: {available_memory_mb:.1f}MB < {self.context_config.memory_warning_threshold_mb}MB"
                    ),
                }
            else:
                return {
                    "status": "ok",
                    "available_mb": available_memory_mb,
                    "message": f"Memory OK: {available_memory_mb:.1f}MB",
                }
        except Exception as e:
            logger.debug(f"Memory check failed: {e}")
            return {
                "status": "error",
                "available_mb": None,
                "message": f"Memory check failed: {e}",
            }

    def run(self, run_id: str, config_hash: str) -> None:
        """
        Analyze photo context using vision LLM.

        For each photo without context data:
        1. Load downsampled image
        2. Run Ollama/MLX vision LLM with structured prompt
        3. Extract decade estimate, season, event hints, photo medium
        4. Store in context table with confidence scores
        """
        logger.info("Starting context layer stage")
        try:
            # Check if we're in degraded mode
            if self._degraded_mode:
                logger.warning(
                    "Context layer stage is in degraded mode. "
                    "Skipping analysis due to Ollama unavailability or model issues."
                )
                # Mark stage as complete with zero photos processed
                self.mark_complete(run_id, photos_processed=0)
                return

            # Optional: perform a fresh health check at runtime
            if not self._is_healthy:
                logger.warning("Context analyzer is not healthy. Attempting to re-check health status...")
                self._validate_configuration()

                if self._degraded_mode or not self._is_healthy:
                    logger.warning("Context analyzer still not healthy after re-check. Skipping analysis.")
                    self.mark_complete(run_id, photos_processed=0)
                    return

            # Get photos without context analysis
            photos = self._get_photos_without_context()
            if not photos:
                logger.info("No photos without context data; stage complete")
                self.mark_complete(run_id, photos_processed=0)
                return

            total_photos = len(photos)
            logger.info(f"Found {total_photos} photos without context data")

            processed = 0
            failed = 0
            batch_size = self.context_config.batch_size

            # Validate batch_size is at least 1
            if batch_size <= 0:
                logger.warning(f"Invalid batch_size {self.context_config.batch_size}, using 1 instead")
                batch_size = 1

            # Calculate total batches for progress reporting
            total_batches = (total_photos + batch_size - 1) // batch_size if total_photos > 0 else 0

            for i in range(0, total_photos, batch_size):
                batch = photos[i : i + batch_size]
                batch_number = i // batch_size + 1

                # Log batch-level progress with percentage
                if total_photos > 0:
                    batch_percentage = (i / total_photos) * 100
                    logger.info(f"Processing batch {batch_number}/{total_batches} ({batch_percentage:.1f}%)")

                # Memory check before processing batch
                memory_check_result = self._check_memory_before_batch()
                if memory_check_result["status"] == "critical":
                    # Memory is critically low, skip this batch and wait
                    batch_percentage = (i / total_photos) * 100 if total_photos > 0 else 0
                    logger.warning(
                        f"Memory critically low ({memory_check_result['available_mb']:.1f}MB < "
                        f"{self.context_config.memory_critical_threshold_mb}MB threshold). "
                        f"Skipping batch {batch_number}/{total_batches} ({batch_percentage:.1f}%) "
                        f"and waiting {self.context_config.memory_retry_delay_seconds}s."
                    )
                    time.sleep(self.context_config.memory_retry_delay_seconds)
                    continue  # Skip this batch
                elif memory_check_result["status"] == "warning":
                    # Memory is low but not critical, log warning but continue
                    logger.warning(
                        f"Low memory available: {memory_check_result['available_mb']:.1f}MB (< "
                        f"{self.context_config.memory_warning_threshold_mb}MB threshold) before processing batch"
                    )
                elif memory_check_result["status"] in ["error", "unknown"]:
                    # Memory check failed or psutil unavailable, log warning but continue
                    logger.warning(
                        f"Memory check returned '{memory_check_result['status']}' status: "
                        f"{memory_check_result.get('message', 'No message')}. "
                        f"Continuing with batch processing."
                    )

                for photo in batch:
                    try:
                        self._process_photo(photo)
                        processed += 1
                        if processed % 10 == 0:
                            # Calculate percentage with 1 decimal place
                            if total_photos > 0:
                                percentage = (processed / total_photos) * 100
                                logger.info(f"Processed {processed}/{total_photos} photos ({percentage:.1f}%)")
                            else:
                                logger.info(f"Processed {processed}/{total_photos} photos")
                    except Exception as e:
                        logger.warning(f"Failed to process photo {photo.id}: {e}")
                        failed += 1
                        continue

            # Calculate final percentage for completion log
            if total_photos > 0:
                final_percentage = (processed / total_photos) * 100
                logger.info(
                    f"Context layer stage completed. Processed {processed}/{total_photos} photos "
                    f"({final_percentage:.1f}%), failed: {failed}"
                )
            else:
                logger.info(
                    f"Context layer stage completed. Processed {processed}/{total_photos} photos, failed: {failed}"
                )
            self.mark_complete(run_id, photos_processed=processed)

        except Exception as e:
            logger.error(f"Context layer stage failed: {e}")
            raise

    def _get_photos_without_context(self) -> list[Any]:
        """Query photos that have no context records yet."""
        store = get_store()
        with store.transaction() as conn:
            helper = store.get_query_helper(conn)
            return helper.get_photos_without_context()

    def _process_photo(self, photo: Any) -> None:
        """
        Process a single photo for context analysis.

        Args:
            photo: Photo object from database
        """
        logger.debug(f"Processing photo {photo.id}: {photo.file_path}")

        # Check if downsample path exists
        downsample_path = Path(photo.downsample_path)
        if not downsample_path.exists():
            logger.warning(f"Downsample path does not exist for photo {photo.id}: {downsample_path}")
            # Try to use original file path as fallback
            original_path = Path(photo.file_path)
            if not original_path.exists():
                logger.error(f"Neither downsample nor original file exists for photo {photo.id}")
                return
            image_path = str(original_path)
        else:
            image_path = str(downsample_path)

        # Analyze the image
        result = self.analyzer.analyze(image_path)

        if result is None:
            logger.warning(f"Context analysis failed for photo {photo.id}")
            # Optionally store minimal data with low confidence
            if self.context_config.store_minimal_on_complete_failure:
                self._store_minimal_context(photo.id)
            return

        # Store the result
        self._store_context_result(photo.id, result)
        logger.debug(f"Stored context analysis for photo {photo.id}")

    def _store_context_result(self, photo_id: int, result: Any) -> None:
        """
        Store context analysis result in database.

        Args:
            photo_id: ID of the photo
            result: ContextAnalysisResult object
        """
        # Convert result to ContextCreate model
        context_data = ContextCreate(
            photo_id=photo_id,
            decade=result.decade,
            decade_confidence=result.decade_confidence,
            season=result.season,
            season_confidence=result.season_confidence,
            event_hint=result.event_hint,
            event_confidence=result.event_confidence,
            photo_medium=result.photo_medium,
            photo_medium_confidence=result.photo_medium_confidence,
            visual_evidence=result.visual_evidence,
            alternative_decades=result.alternative_decades,
            uncertainty_flag=result.uncertainty_flag,
            hypothesis_notes=result.hypothesis_notes,
            raw_json=result.model_dump_json()
            if hasattr(result, "model_dump_json")
            else json.dumps({"error": "No model_dump_json method available"}),
        )

        store = get_store()
        with store.transaction() as conn:
            helper = store.get_query_helper(conn)
            helper.insert_context(context_data)

    def _store_minimal_context(self, photo_id: int) -> None:
        """
        Store minimal context data when analysis completely fails.

        Args:
            photo_id: ID of the photo
        """
        # Create minimal JSON for failed analysis
        minimal_json = {
            "status": "failed",
            "error": "Analysis failed completely",
            "photo_id": photo_id,
            "timestamp": "1970-01-01T00:00:00Z",  # Placeholder timestamp
        }

        context_data = ContextCreate(
            photo_id=photo_id,
            decade=None,
            decade_confidence=0.0,
            season=None,
            season_confidence=None,
            event_hint=None,
            event_confidence=None,
            photo_medium="unknown",
            photo_medium_confidence=0.0,
            visual_evidence=None,
            alternative_decades=None,
            uncertainty_flag=True,
            hypothesis_notes="Analysis failed completely",
            raw_json=json.dumps(minimal_json),
        )

        store = get_store()
        with store.transaction() as conn:
            helper = store.get_query_helper(conn)
            helper.insert_context(context_data)
