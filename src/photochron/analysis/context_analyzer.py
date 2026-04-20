"""
Context analyzer for photo analysis using vision LLM.

This module provides the ContextAnalyzer class which orchestrates the analysis
of photos for decade estimation, season detection, event hints, and photo medium
identification using the OllamaClient.
"""

import time
import random
from typing import Optional, Dict, Any, List, Callable, TypeVar
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

T = TypeVar("T")

from photochron.models.ollama_client import (
    OllamaClient,
    OllamaConfig,
    ContextAnalysisResult,
    ModelType,
)


@dataclass
class ContextAnalyzerConfig:
    """Configuration for ContextAnalyzer."""

    min_decade_confidence: float = 0.3
    """Minimum confidence threshold for decade estimates (0.0-1.0)."""

    min_season_confidence: float = 0.4
    """Minimum confidence threshold for season estimates (0.0-1.0)."""

    min_event_confidence: float = 0.5
    """Minimum confidence threshold for event hints (0.0-1.0)."""

    min_photo_medium_confidence: float = 0.4
    """Minimum confidence threshold for photo medium identification (0.0-1.0)."""

    enable_retries: bool = True
    """Whether to enable retries on analysis failures."""

    max_retries: int = 2
    """Maximum number of retry attempts."""

    use_base64: bool = False
    """Whether to encode images as base64 for Ollama."""

    model_priority: List[ModelType] = field(
        default_factory=lambda: [
            ModelType.LLAVA_NEXT_7B,
            ModelType.MOONDREAM2,
        ]
    )
    """Ordered list of models to try."""


class ContextAnalyzer:
    """
    Analyzer for photo context using vision LLM.

    This class orchestrates the analysis of photos for:
    - Decade estimation with confidence scores
    - Season detection (spring, summer, autumn, winter)
    - Event hints (wedding, birthday, graduation, etc.)
    - Photo medium identification (digital, print_scan, polaroid, etc.)

    The analyzer implements a main analysis pipeline with confidence scoring
    and handles fallback analysis for failed or low-confidence results.
    """

    def __init__(
        self,
        ollama_client: Optional[OllamaClient] = None,
        config: Optional[ContextAnalyzerConfig] = None,
        ollama_config: Optional[OllamaConfig] = None,
    ) -> None:
        """
        Initialize the context analyzer.

        Args:
            ollama_client: Pre-configured OllamaClient instance. If None,
                a new client will be created.
            config: ContextAnalyzer configuration. If None, defaults are used.
            ollama_config: OllamaClient configuration. Only used if
                ollama_client is None.
        """
        self.config = config or ContextAnalyzerConfig()

        # Initialize Ollama client
        if ollama_client is None:
            ollama_config = ollama_config or OllamaConfig()
            # Sync retry settings from analyzer config to ollama config
            if not self.config.enable_retries:
                ollama_config.max_retries = 0  # Disable retries in OllamaClient
            else:
                ollama_config.max_retries = self.config.max_retries
            self.ollama_client = OllamaClient(ollama_config)
            # Try to connect
            if not self.ollama_client.connect():
                logger.warning("Ollama client failed to connect on initialization")
        else:
            self.ollama_client = ollama_client

        logger.info(f"Initialized ContextAnalyzer")

    def _with_retry(
        self, operation: Callable[[], Optional[T]], operation_name: str
    ) -> Optional[T]:
        """
        Execute an operation with retry logic based on analyzer configuration.

        Args:
            operation: The operation to execute (returns Optional[T]).
            operation_name: Name of the operation for logging.

        Returns:
            The result of the operation, or None if all retries failed.
        """
        if not self.config.enable_retries:
            # Execute without retries
            return operation()

        last_exception = None
        for attempt in range(self.config.max_retries + 1):  # +1 for initial attempt
            try:
                result = operation()
                if result is not None:
                    if attempt > 0:
                        logger.info(
                            f"{operation_name} succeeded on attempt {attempt + 1}"
                        )
                    return result
                else:
                    # Operation returned None (not an exception)
                    logger.debug(
                        f"{operation_name} returned None on attempt {attempt + 1}"
                    )

            except Exception as e:
                last_exception = e
                logger.warning(f"{operation_name} failed on attempt {attempt + 1}: {e}")

            # Check if we should retry
            if attempt < self.config.max_retries:
                # Exponential backoff with jitter
                base_delay = 2.0 * (2**attempt)  # 2, 4, 8 seconds
                jitter = random.uniform(-0.5, 0.5)  # ±0.5 seconds jitter
                delay = max(0.5, base_delay + jitter)  # Minimum 0.5 seconds

                logger.debug(f"Retrying {operation_name} in {delay:.2f} seconds...")
                time.sleep(delay)

        # All retries exhausted
        if last_exception:
            logger.error(
                f"{operation_name} failed after {self.config.max_retries + 1} attempts: {last_exception}"
            )
        else:
            logger.error(
                f"{operation_name} returned None after {self.config.max_retries + 1} attempts"
            )

        return None

    def analyze(
        self,
        image_path: str,
    ) -> Optional[ContextAnalysisResult]:
        """
        Analyze a photo for context information.

        This is the main entry point for photo context analysis. It implements
        the main analysis pipeline with confidence scoring and handles fallback
        analysis for failed or low-confidence results.

        Args:
            image_path: Path to the image file to analyze.

        Returns:
            ContextAnalysisResult with analysis results, or None if analysis
            failed completely.
        """
        logger.info(f"Analyzing image: {image_path}")

        # Validate image path
        if not Path(image_path).exists():
            logger.error(f"Image file does not exist: {image_path}")
            return None

        # Run main analysis pipeline
        result = self._run_main_analysis_pipeline(image_path)

        # Apply fallback if needed
        if result is None:
            result = self._apply_fallback_analysis(image_path)

        # Validate and clean result
        if result is not None:
            result = self._validate_and_clean_result(result)
            logger.info(
                f"Analysis complete for {image_path}: "
                f"decade={result.decade} (conf={result.decade_confidence:.2f}), "
                f"season={result.season}, medium={result.photo_medium}"
                f"{f' (conf={result.photo_medium_confidence:.2f})' if result.photo_medium_confidence is not None else ''}"
            )

        return result

    def _run_main_analysis_pipeline(
        self, image_path: str
    ) -> Optional[ContextAnalysisResult]:
        """
        Run the main analysis pipeline.

        This implements the core analysis logic:
        1. Try primary model with default prompt
        2. If low confidence or failure, try fallback model
        3. Return the best result

        Args:
            image_path: Path to the image file.

        Returns:
            Analysis result, or None if analysis fails.
        """
        logger.debug("Running main analysis pipeline")

        # Get primary model name
        primary_model = self._get_primary_model_name()
        if primary_model is None:
            logger.error("Cannot perform analysis: no primary model available")
            return None

        # Try primary model with default prompt
        result = self._with_retry(
            lambda: self.ollama_client.analyze_image_context(
                image_input=image_path,
                model_name=primary_model,
                prompt_template=self.ollama_client.get_prompt_template("default"),
                use_base64=self.config.use_base64,
            ),
            "Primary model analysis",
        )

        # If result is None or has very low confidence, try fallback model
        if (
            result is None
            or result.decade_confidence < self.config.min_decade_confidence
        ):
            fallback_model = self._get_fallback_model_name()
            if fallback_model:
                logger.debug("Trying fallback model due to low confidence")
                result = self._with_retry(
                    lambda: self.ollama_client.analyze_image_context(
                        image_input=image_path,
                        model_name=fallback_model,
                        prompt_template=self.ollama_client.get_prompt_template(
                            "default"
                        ),
                        use_base64=self.config.use_base64,
                    ),
                    "Fallback model analysis",
                )

        return result

    def _get_primary_model_name(self) -> Optional[str]:
        """
        Safely get the primary model name from model_priority list.

        Returns:
            Model name string, or None if model_priority list is empty.
        """
        if not self.config.model_priority:
            logger.error("model_priority list is empty")
            return None
        return self.config.model_priority[0].value

    def _get_fallback_model_name(self) -> Optional[str]:
        """
        Safely get the fallback model name from model_priority list.

        Returns:
            Model name string, or None if model_priority list has less than 2 items.
        """
        if len(self.config.model_priority) < 2:
            logger.debug("No fallback model available in model_priority list")
            return None
        return self.config.model_priority[1].value

    def _apply_fallback_analysis(
        self, image_path: str
    ) -> Optional[ContextAnalysisResult]:
        """
        Apply fallback analysis when main pipeline fails.

        Args:
            image_path: Path to the image file.

        Returns:
            Fallback analysis result, or None if fallback also fails.
        """
        logger.info("Applying fallback analysis")

        primary_model = self._get_primary_model_name()
        if primary_model is None:
            logger.error("Cannot apply fallback analysis: no primary model available")
            return None

        # Try simple fallback prompt first
        result = self._with_retry(
            lambda: self.ollama_client.analyze_image_context(
                image_input=image_path,
                model_name=primary_model,
                prompt_template=self.ollama_client.get_prompt_template(
                    "simple_fallback"
                ),
                use_base64=self.config.use_base64,
            ),
            "Simple fallback analysis",
        )

        # If still no result, try uncertainty handling
        if result is None:
            logger.debug("Trying uncertainty handling fallback")
            result = self._with_retry(
                lambda: self.ollama_client.analyze_image_context(
                    image_input=image_path,
                    model_name=primary_model,
                    prompt_template=self.ollama_client.get_prompt_template(
                        "uncertainty_handling"
                    ),
                    use_base64=self.config.use_base64,
                ),
                "Uncertainty handling fallback",
            )

        return result

    def _validate_and_clean_result(
        self,
        result: ContextAnalysisResult,
    ) -> ContextAnalysisResult:
        """
        Validate and clean analysis result based on configuration.

        Args:
            result: Raw analysis result.

        Returns:
            Cleaned and validated result.
        """
        # Clear fields below confidence thresholds
        if result.season is not None and result.season_confidence is not None:
            if result.season_confidence < self.config.min_season_confidence:
                logger.debug(
                    f"Clearing season '{result.season}' due to low confidence: "
                    f"{result.season_confidence:.2f} < {self.config.min_season_confidence:.2f}"
                )
                result.season = None
                result.season_confidence = None

        if result.event_hint is not None and result.event_confidence is not None:
            if result.event_confidence < self.config.min_event_confidence:
                logger.debug(
                    f"Clearing event hint '{result.event_hint}' due to low confidence: "
                    f"{result.event_confidence:.2f} < {self.config.min_event_confidence:.2f}"
                )
                result.event_hint = None
                result.event_confidence = None

        # Clear decade if confidence is too low
        if result.decade is not None:
            if result.decade_confidence < self.config.min_decade_confidence:
                logger.debug(
                    f"Clearing decade '{result.decade}' due to low confidence: "
                    f"{result.decade_confidence:.2f} < {self.config.min_decade_confidence:.2f}"
                )
                result.decade = None
                result.decade_confidence = 0.0
                result.alternative_decades = None

        # Set photo_medium to "unknown" if confidence is too low
        if result.photo_medium_confidence is not None:
            if result.photo_medium_confidence < self.config.min_photo_medium_confidence:
                logger.debug(
                    f"Setting photo_medium to 'unknown' due to low confidence: "
                    f"{result.photo_medium_confidence:.2f} < {self.config.min_photo_medium_confidence:.2f}"
                )
                result.photo_medium = "unknown"
                result.photo_medium_confidence = None

        return result

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the analyzer and underlying Ollama client.

        Returns:
            Dictionary with health status and details.
        """
        ollama_health = self.ollama_client.health_check()

        health_status = {
            "status": "healthy"
            if ollama_health.get("status") == "healthy"
            else "unhealthy",
            "analyzer_config": {
                "min_decade_confidence": self.config.min_decade_confidence,
                "min_season_confidence": self.config.min_season_confidence,
                "min_event_confidence": self.config.min_event_confidence,
                "min_photo_medium_confidence": self.config.min_photo_medium_confidence,
            },
            "ollama_health": ollama_health,
        }

        return health_status


# Default analyzer instance
_default_analyzer: Optional[ContextAnalyzer] = None


def get_context_analyzer(
    ollama_client: Optional[OllamaClient] = None,
    config: Optional[ContextAnalyzerConfig] = None,
    ollama_config: Optional[OllamaConfig] = None,
) -> ContextAnalyzer:
    """
    Get or create default ContextAnalyzer instance.

    Args:
        ollama_client: Pre-configured OllamaClient instance.
        config: ContextAnalyzer configuration.
        ollama_config: OllamaClient configuration.

    Returns:
        ContextAnalyzer instance.
    """
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = ContextAnalyzer(
            ollama_client=ollama_client,
            config=config,
            ollama_config=ollama_config,
        )
    return _default_analyzer
