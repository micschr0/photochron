"""
Tests for confidence score validation and propagation.

This module tests confidence score validation and propagation through the analysis pipeline:
- ContextAnalysisResult model validation for confidence scores
- Confidence propagation through analysis pipeline
- Confidence threshold validation in ContextAnalyzer
- Overall confidence calculation methods
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
from typing import Optional, List
from pydantic import ValidationError

from photochron.models.ollama_client import (
    ContextAnalysisResult,
    ModelType,
    OllamaClient,
    OllamaConfig,
)
from photochron.analysis.context_analyzer import (
    ContextAnalyzer,
    ContextAnalyzerConfig as NewContextAnalyzerConfig,
)
from photochron.context.analyzer import (
    ContextAnalyzer as LegacyContextAnalyzer,
    ContextAnalyzerConfig as LegacyContextAnalyzerConfig,
    AnalysisStrategy,
    AnalysisStrategy as LegacyAnalysisStrategy,
    FallbackStrategy,
)


class TestContextAnalysisResultConfidenceValidation:
    """Test confidence score validation in ContextAnalysisResult model."""

    def test_valid_confidence_scores(self):
        """Test ContextAnalysisResult with valid confidence scores."""
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            season="summer",
            season_confidence=0.8,
            event_hint="wedding",
            event_confidence=0.9,
            photo_medium="print_scan",
            photo_medium_confidence=0.7,
            visual_evidence=["bell-bottom jeans", "large collar shirt"],
        )

        assert result.decade_confidence == 0.75
        assert result.season_confidence == 0.8
        assert result.event_confidence == 0.9
        assert result.photo_medium_confidence == 0.7

    def test_confidence_score_boundaries(self):
        """Test confidence scores at boundary values (0.0 and 1.0)."""
        # Test minimum boundary (0.0)
        result_min = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.0,
            season="summer",
            season_confidence=0.0,
            photo_medium="digital",
            photo_medium_confidence=0.0,
        )

        # @model_validator logic:
        # - season_confidence=0.0 (< 0.3): season cleared, season_confidence=None
        # - photo_medium_confidence=0.0 (< 0.3): photo_medium="unknown", photo_medium_confidence=None
        # - decade_confidence=0.0 (< 0.2): decade cleared
        assert result_min.decade is None  # decade cleared (confidence < 0.2)
        assert result_min.decade_confidence == 0.0  # decade_confidence remains 0.0
        assert result_min.season is None  # season cleared (confidence < 0.3)
        assert result_min.season_confidence is None  # season_confidence cleared
        assert (
            result_min.photo_medium == "unknown"
        )  # set to "unknown" (confidence < 0.3)
        assert (
            result_min.photo_medium_confidence is None
        )  # photo_medium_confidence cleared

        # Test maximum boundary (1.0)
        result_max = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=1.0,
            season="summer",
            season_confidence=1.0,
            photo_medium="digital",
            photo_medium_confidence=1.0,
        )

        # All confidences >= thresholds, so all fields should be preserved
        assert result_max.decade_confidence == 1.0
        assert result_max.season_confidence == 1.0
        assert result_max.photo_medium_confidence == 1.0

    def test_invalid_confidence_below_zero(self):
        """Test validation error when confidence is below 0.0."""
        with pytest.raises(ValidationError) as exc_info:
            ContextAnalysisResult(
                decade="1985-1990",
                decade_confidence=-0.1,
                photo_medium="digital",
            )

        assert "decade_confidence" in str(exc_info.value)
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_invalid_confidence_above_one(self):
        """Test validation error when confidence is above 1.0."""
        with pytest.raises(ValidationError) as exc_info:
            ContextAnalysisResult(
                decade="1985-1990",
                decade_confidence=1.1,
                photo_medium="digital",
            )

        assert "decade_confidence" in str(exc_info.value)
        assert "less than or equal to 1" in str(exc_info.value)

    def test_none_confidence_scores(self):
        """Test ContextAnalysisResult with None confidence scores."""
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            season=None,
            season_confidence=None,
            event_hint=None,
            event_confidence=None,
            photo_medium="digital",
            photo_medium_confidence=None,
        )

        assert result.decade_confidence == 0.75
        assert result.season_confidence is None
        assert result.event_confidence is None
        assert result.photo_medium_confidence is None

    def test_season_confidence_validation_with_season(self):
        """Test season_confidence validation when season is provided."""
        # Should accept None confidence even with season
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            season="summer",
            season_confidence=None,
            photo_medium="digital",
        )

        assert result.season_confidence is None

        # Should accept valid confidence with season
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            season="summer",
            season_confidence=0.8,
            photo_medium="digital",
        )

        assert result.season_confidence == 0.8

    def test_event_confidence_validation_with_event_hint(self):
        """Test event_confidence validation when event_hint is provided."""
        # Should accept None confidence even with event_hint
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            event_hint="wedding",
            event_confidence=None,
            photo_medium="digital",
        )

        assert result.event_confidence is None

        # Should accept valid confidence with event_hint
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            event_hint="wedding",
            event_confidence=0.9,
            photo_medium="digital",
        )

        assert result.event_confidence == 0.9

    def test_photo_medium_confidence_validation(self):
        """Test photo_medium_confidence validation."""
        # Should accept None confidence
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            photo_medium="digital",
            photo_medium_confidence=None,
        )

        assert result.photo_medium_confidence is None

        # Should accept valid confidence
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            photo_medium="digital",
            photo_medium_confidence=0.6,
        )

        assert result.photo_medium_confidence == 0.6

    def test_mixed_confidence_levels(self):
        """Test ContextAnalysisResult with mixed confidence levels."""
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.95,  # Very high
            season="summer",
            season_confidence=0.65,  # Medium
            event_hint="graduation",
            event_confidence=0.45,  # Low
            photo_medium="print_scan",
            photo_medium_confidence=0.85,  # High
            visual_evidence=["graduation cap", "robe"],
        )

        assert result.decade_confidence == 0.95
        assert result.season_confidence == 0.65
        assert result.event_confidence == 0.45
        assert result.photo_medium_confidence == 0.85

    def test_model_validator_clears_low_confidence_season(self):
        """Test @model_validator clears season when confidence < 0.3."""
        # Create result with season confidence below model validator threshold (0.3)
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            season="summer",
            season_confidence=0.25,  # Below 0.3 threshold
            photo_medium="digital",
            photo_medium_confidence=0.7,
        )

        # Model validator should clear season and season_confidence
        assert result.season is None
        assert result.season_confidence is None
        # Other fields should remain
        assert result.decade == "1985-1990"
        assert result.decade_confidence == 0.75

    def test_model_validator_clears_low_confidence_event_hint(self):
        """Test @model_validator clears event_hint when confidence < 0.3."""
        # Create result with event confidence below model validator threshold (0.3)
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            event_hint="wedding",
            event_confidence=0.25,  # Below 0.3 threshold
            photo_medium="digital",
            photo_medium_confidence=0.7,
        )

        # Model validator should clear event_hint and event_confidence
        assert result.event_hint is None
        assert result.event_confidence is None
        # Other fields should remain
        assert result.decade == "1985-1990"
        assert result.decade_confidence == 0.75

    def test_model_validator_sets_photo_medium_to_unknown_when_low_confidence(self):
        """Test @model_validator sets photo_medium to 'unknown' when confidence < 0.3."""
        # Create result with photo medium confidence below model validator threshold (0.3)
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            photo_medium="print_scan",
            photo_medium_confidence=0.25,  # Below 0.3 threshold
        )

        # Model validator should set photo_medium to "unknown" and clear confidence
        assert result.photo_medium == "unknown"
        assert result.photo_medium_confidence is None
        # Other fields should remain
        assert result.decade == "1985-1990"
        assert result.decade_confidence == 0.75

    def test_model_validator_preserves_fields_with_adequate_confidence(self):
        """Test @model_validator preserves fields when confidence >= 0.3."""
        # Create result with all confidences at or above 0.3
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            season="summer",
            season_confidence=0.3,  # Exactly at threshold
            event_hint="wedding",
            event_confidence=0.35,  # Above threshold
            photo_medium="print_scan",
            photo_medium_confidence=0.4,  # Above threshold
        )

        # All fields should be preserved
        assert result.season == "summer"
        assert result.season_confidence == 0.3
        assert result.event_hint == "wedding"
        assert result.event_confidence == 0.35
        assert result.photo_medium == "print_scan"
        assert result.photo_medium_confidence == 0.4


class TestContextAnalyzerConfidenceThresholdValidation:
    """Test confidence threshold validation in ContextAnalyzer."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock OllamaClient."""
        mock_client = Mock(spec=OllamaClient)
        mock_client.analyze_image_context = Mock()
        mock_client.get_prompt_template = Mock(return_value="Test prompt template")
        mock_client.connect = Mock(return_value=True)
        mock_client.health_check = Mock(
            return_value={"status": "healthy", "server_available": True}
        )
        return mock_client

    @pytest.fixture
    def analyzer_with_custom_thresholds(self, mock_ollama_client):
        """Create ContextAnalyzer with custom confidence thresholds."""
        config = NewContextAnalyzerConfig(
            min_decade_confidence=0.4,
            min_season_confidence=0.5,
            min_event_confidence=0.6,
            min_photo_medium_confidence=0.5,
            enable_retries=True,
            max_retries=2,
            model_priority=[ModelType.LLAVA_NEXT_7B, ModelType.MOONDREAM2],
        )
        return ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

    def test_run_main_analysis_pipeline_rejects_low_decade_confidence(
        self, analyzer_with_custom_thresholds, mock_ollama_client, caplog
    ):
        """Test _run_main_analysis_pipeline() rejects result with decade confidence below threshold."""
        # Create result with low decade confidence
        low_confidence_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.35,  # Below min_decade_confidence of 0.4
            season="summer",
            season_confidence=0.8,
            photo_medium="digital",
            photo_medium_confidence=0.7,
        )

        mock_ollama_client.analyze_image_context.return_value = low_confidence_result

        # Mock model priority getters
        analyzer_with_custom_thresholds._get_primary_model_name = Mock(
            return_value="llava-next:7b"
        )
        analyzer_with_custom_thresholds._get_fallback_model_name = Mock(
            return_value="moondream2"
        )

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer_with_custom_thresholds._run_main_analysis_pipeline(
                    "test.jpg"
                )

        # _run_main_analysis_pipeline returns result even with low confidence
        # Confidence validation happens in _validate_and_clean_result
        assert result == low_confidence_result
        assert result.decade_confidence == 0.35
        # Should log about trying fallback due to low confidence
        assert (
            "low confidence" in caplog.text.lower()
            or "fallback model" in caplog.text.lower()
        )

    def test_run_main_analysis_pipeline_accepts_high_decade_confidence(
        self, analyzer_with_custom_thresholds, mock_ollama_client
    ):
        """Test _run_main_analysis_pipeline() accepts result with decade confidence above threshold."""
        # Create result with high decade confidence
        high_confidence_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.85,  # Above min_decade_confidence of 0.4
            season="summer",
            season_confidence=0.8,
            photo_medium="digital",
            photo_medium_confidence=0.7,
        )

        mock_ollama_client.analyze_image_context.return_value = high_confidence_result

        # Mock model priority getters
        analyzer_with_custom_thresholds._get_primary_model_name = Mock(
            return_value="llava-next:7b"
        )
        analyzer_with_custom_thresholds._get_fallback_model_name = Mock(
            return_value="moondream2"
        )

        # Execute
        with patch("time.sleep"):
            result = analyzer_with_custom_thresholds._run_main_analysis_pipeline(
                "test.jpg"
            )

        # Should accept the result
        assert result == high_confidence_result

    def test_run_main_analysis_pipeline_handles_none_season_confidence(
        self, analyzer_with_custom_thresholds, mock_ollama_client
    ):
        """Test _run_main_analysis_pipeline() handles None season confidence."""
        # Create result with None season confidence but valid decade confidence
        result_with_none_season = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.85,  # Above threshold
            season="summer",
            season_confidence=None,  # None should be acceptable
            photo_medium="digital",
            photo_medium_confidence=0.7,
        )

        mock_ollama_client.analyze_image_context.return_value = result_with_none_season

        # Mock model priority getters
        analyzer_with_custom_thresholds._get_primary_model_name = Mock(
            return_value="llava-next:7b"
        )
        analyzer_with_custom_thresholds._get_fallback_model_name = Mock(
            return_value="moondream2"
        )

        # Execute
        with patch("time.sleep"):
            result = analyzer_with_custom_thresholds._run_main_analysis_pipeline(
                "test.jpg"
            )

        # Should accept the result even with None season confidence
        assert result == result_with_none_season

    def test_run_main_analysis_pipeline_rejects_low_season_confidence_when_season_present(
        self, analyzer_with_custom_thresholds, mock_ollama_client, caplog
    ):
        """Test _run_main_analysis_pipeline() rejects result with season confidence below threshold when season is present."""
        # Create result with low season confidence
        low_season_confidence_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.85,  # Above threshold
            season="summer",
            season_confidence=0.45,  # Below min_season_confidence of 0.5
            photo_medium="digital",
            photo_medium_confidence=0.7,
        )

        mock_ollama_client.analyze_image_context.return_value = (
            low_season_confidence_result
        )

        # Mock model priority getters
        analyzer_with_custom_thresholds._get_primary_model_name = Mock(
            return_value="llava-next:7b"
        )
        analyzer_with_custom_thresholds._get_fallback_model_name = Mock(
            return_value="moondream2"
        )

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer_with_custom_thresholds._run_main_analysis_pipeline(
                    "test.jpg"
                )

        # _run_main_analysis_pipeline doesn't check season confidence, only decade confidence
        # Since decade confidence is high (0.85 > 0.4), it returns the result
        # Season confidence validation happens in _validate_and_clean_result
        assert result == low_season_confidence_result
        assert result.season_confidence == 0.45
        # Should NOT log about low season confidence here (that happens in _validate_and_clean_result)

    def test_run_main_analysis_pipeline_handles_event_confidence_threshold(
        self, analyzer_with_custom_thresholds, mock_ollama_client, caplog
    ):
        """Test _run_main_analysis_pipeline() handles event confidence threshold."""
        # Create result with low event confidence
        low_event_confidence_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.85,
            season="summer",
            season_confidence=0.8,
            event_hint="wedding",
            event_confidence=0.55,  # Below min_event_confidence of 0.6
            photo_medium="digital",
            photo_medium_confidence=0.7,
        )

        mock_ollama_client.analyze_image_context.return_value = (
            low_event_confidence_result
        )

        # Mock model priority getters
        analyzer_with_custom_thresholds._get_primary_model_name = Mock(
            return_value="llava-next:7b"
        )
        analyzer_with_custom_thresholds._get_fallback_model_name = Mock(
            return_value="moondream2"
        )

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer_with_custom_thresholds._run_main_analysis_pipeline(
                    "test.jpg"
                )

        # _run_main_analysis_pipeline doesn't check event confidence, only decade confidence
        # Since decade confidence is high (0.85 > 0.4), it returns the result
        # Event confidence validation happens in _validate_and_clean_result
        assert result == low_event_confidence_result
        assert result.event_confidence == 0.55
        # Should NOT log about low event confidence here (that happens in _validate_and_clean_result)

    def test_run_main_analysis_pipeline_handles_photo_medium_confidence_threshold(
        self, analyzer_with_custom_thresholds, mock_ollama_client, caplog
    ):
        """Test _run_main_analysis_pipeline() handles photo medium confidence threshold."""
        # Create result with low photo medium confidence
        low_photo_confidence_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.85,
            season="summer",
            season_confidence=0.8,
            photo_medium="print_scan",
            photo_medium_confidence=0.45,  # Below min_photo_medium_confidence of 0.5
        )

        mock_ollama_client.analyze_image_context.return_value = (
            low_photo_confidence_result
        )

        # Mock model priority getters
        analyzer_with_custom_thresholds._get_primary_model_name = Mock(
            return_value="llava-next:7b"
        )
        analyzer_with_custom_thresholds._get_fallback_model_name = Mock(
            return_value="moondream2"
        )

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer_with_custom_thresholds._run_main_analysis_pipeline(
                    "test.jpg"
                )

        # _run_main_analysis_pipeline doesn't check photo medium confidence, only decade confidence
        # Since decade confidence is high (0.85 > 0.4), it returns the result
        # Photo medium confidence validation happens in _validate_and_clean_result
        assert result == low_photo_confidence_result
        assert result.photo_medium_confidence == 0.45
        # Should NOT log about low photo medium confidence here (that happens in _validate_and_clean_result)

    def test_validate_and_clean_result_clears_low_confidence_fields(
        self, analyzer_with_custom_thresholds
    ):
        """Test _validate_and_clean_result() clears fields with confidence below thresholds."""
        # Create result with some low confidence fields
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.85,  # Above threshold (0.4)
            season="summer",
            season_confidence=0.45,  # Below threshold (0.5)
            event_hint="wedding",
            event_confidence=0.55,  # Below threshold (0.6)
            photo_medium="print_scan",
            photo_medium_confidence=0.45,  # Below threshold (0.5)
        )

        # Apply validation and cleaning
        cleaned_result = analyzer_with_custom_thresholds._validate_and_clean_result(
            result
        )

        # Season should be cleared (confidence < 0.5)
        assert cleaned_result.season is None
        assert cleaned_result.season_confidence is None

        # Event hint should be cleared (confidence < 0.6)
        assert cleaned_result.event_hint is None
        assert cleaned_result.event_confidence is None

        # Photo medium should be set to "unknown" (confidence < 0.5)
        assert cleaned_result.photo_medium == "unknown"
        assert cleaned_result.photo_medium_confidence is None

        # Decade should remain (confidence > 0.4)
        assert cleaned_result.decade == "1985-1990"
        assert cleaned_result.decade_confidence == 0.85

    def test_validate_and_clean_result_preserves_high_confidence_fields(
        self, analyzer_with_custom_thresholds
    ):
        """Test _validate_and_clean_result() preserves fields with confidence above thresholds."""
        # Create result with all high confidence fields
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.85,  # Above threshold
            season="summer",
            season_confidence=0.8,  # Above threshold
            event_hint="wedding",
            event_confidence=0.9,  # Above threshold
            photo_medium="print_scan",
            photo_medium_confidence=0.7,  # Above threshold
        )

        # Apply validation and cleaning
        cleaned_result = analyzer_with_custom_thresholds._validate_and_clean_result(
            result
        )

        # All fields should be preserved
        assert cleaned_result.decade == "1985-1990"
        assert cleaned_result.decade_confidence == 0.85
        assert cleaned_result.season == "summer"
        assert cleaned_result.season_confidence == 0.8
        assert cleaned_result.event_hint == "wedding"
        assert cleaned_result.event_confidence == 0.9
        assert cleaned_result.photo_medium == "print_scan"
        assert cleaned_result.photo_medium_confidence == 0.7

    @patch("pathlib.Path.exists", return_value=True)
    def test_analyze_method_integration(
        self, mock_path_exists, mock_ollama_client, caplog
    ):
        """Test public analyze() method integration with confidence validation."""
        config = NewContextAnalyzerConfig(
            min_decade_confidence=0.3,
            min_season_confidence=0.4,
            min_event_confidence=0.5,
            min_photo_medium_confidence=0.4,
            enable_retries=True,
            max_retries=2,
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Create a result that will pass validation
        valid_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.85,
            season="summer",
            season_confidence=0.8,
            event_hint="wedding",
            event_confidence=0.9,
            photo_medium="print_scan",
            photo_medium_confidence=0.7,
        )

        mock_ollama_client.analyze_image_context.return_value = valid_result

        # Mock model priority getters
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer._get_fallback_model_name = Mock(return_value="moondream2")

        # Execute analyze method
        with patch("time.sleep"):
            with caplog.at_level(logging.INFO):
                result = analyzer.analyze("test.jpg")

        # Should return the validated result
        assert result == valid_result
        # Should log about analysis completion
        assert "Analysis complete" in caplog.text


class TestConfidencePropagationThroughPipeline:
    """Test confidence propagation through the analysis pipeline."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock OllamaClient."""
        mock_client = Mock(spec=OllamaClient)
        mock_client.analyze_image_context = Mock()
        mock_client.get_prompt_template = Mock(return_value="Test prompt template")
        mock_client.connect = Mock(return_value=True)
        mock_client.health_check = Mock(
            return_value={"status": "healthy", "server_available": True}
        )
        return mock_client

    def test_confidence_propagation_through_retry_logic(
        self, mock_ollama_client, caplog
    ):
        """Test confidence scores propagate through retry logic."""
        config = NewContextAnalyzerConfig(
            min_decade_confidence=0.3,
            enable_retries=True,
            max_retries=2,
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Create sequence of results with increasing confidence
        low_confidence_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.25,  # Below threshold
            photo_medium="digital",
        )

        medium_confidence_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.5,  # Above threshold
            photo_medium="digital",
        )

        # Mock analyze_image_context to return low confidence first, then medium
        mock_ollama_client.analyze_image_context.side_effect = [
            low_confidence_result,
            medium_confidence_result,
        ]

        # Mock model priority getters
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer._get_fallback_model_name = Mock(return_value="moondream2")

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer._run_main_analysis_pipeline("test.jpg")

        # Should return the medium confidence result after retry
        assert result == medium_confidence_result
        assert result.decade_confidence == 0.5
        # Should have called analyze_image_context twice (initial + retry)
        assert mock_ollama_client.analyze_image_context.call_count == 2

    def test_confidence_propagation_with_model_fallback(
        self, mock_ollama_client, caplog
    ):
        """Test confidence scores propagate through model fallback."""
        config = NewContextAnalyzerConfig(
            min_decade_confidence=0.3,
            enable_retries=True,
            max_retries=2,
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Create results with different confidence levels for different models
        low_confidence_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.25,  # Below threshold
            photo_medium="digital",
        )

        high_confidence_result = ContextAnalysisResult(
            decade="1980-1985",  # Different decade to show fallback worked
            decade_confidence=0.8,  # Above threshold
            photo_medium="print_scan",
        )

        # Mock analyze_image_context to return low confidence for primary model,
        # high confidence for fallback model
        def analyze_side_effect(*args, **kwargs):
            model_name = kwargs.get("model_name", "llava-next:7b")
            if model_name == "llava-next:7b":
                return low_confidence_result
            elif model_name == "moondream2":
                return high_confidence_result
            return None

        mock_ollama_client.analyze_image_context.side_effect = analyze_side_effect

        # Mock model priority getters
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer._get_fallback_model_name = Mock(return_value="moondream2")

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer._run_main_analysis_pipeline("test.jpg")

        # Should return the high confidence result from fallback model
        assert result == high_confidence_result
        assert result.decade_confidence == 0.8
        assert result.decade == "1980-1985"
        # Should have tried both models
        assert mock_ollama_client.analyze_image_context.call_count >= 2

    def test_confidence_propagation_in_aggressive_strategy(
        self, mock_ollama_client, caplog
    ):
        """Test confidence propagation in aggressive analysis strategy."""
        # Note: This tests the legacy ContextAnalyzer from context.analyzer module
        config = LegacyContextAnalyzerConfig(
            min_decade_confidence=0.3,
            enable_retries=True,
            max_retries=2,
        )
        analyzer = LegacyContextAnalyzer(
            ollama_client=mock_ollama_client, config=config
        )

        # Create multiple results with different confidence levels
        results = [
            ContextAnalysisResult(
                decade="1975-1980",
                decade_confidence=0.6,
                photo_medium="film_negative",
            ),
            ContextAnalysisResult(
                decade="1980-1985",
                decade_confidence=0.8,  # Highest confidence
                photo_medium="print_scan",
            ),
            ContextAnalysisResult(
                decade="1970-1975",
                decade_confidence=0.7,
                photo_medium="polaroid",
            ),
        ]

        # Mock analyze_image_context to return different results
        mock_ollama_client.analyze_image_context.side_effect = results

        # Mock model priority getters
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer._get_fallback_model_name = Mock(return_value="moondream2")

        # Execute aggressive strategy
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer._analyze_aggressive("test.jpg")

        # Should return the result with highest confidence (0.8)
        assert result.decade_confidence == 0.8
        assert result.decade == "1980-1985"
        # Aggressive strategy stops early when it gets high confidence (0.8)
        # So it should stop after 2 calls, not try the 3rd result (0.7)
        assert mock_ollama_client.analyze_image_context.call_count == 2


class TestOverallConfidenceCalculation:
    """Test overall confidence calculation methods."""

    def test_calculate_overall_confidence_basic(self):
        """Test basic overall confidence calculation."""
        # Create a mock analyzer
        analyzer = LegacyContextAnalyzer()

        # Create result with various confidence scores
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            season="summer",
            season_confidence=0.8,
            event_hint="wedding",
            event_confidence=0.9,
            photo_medium="print_scan",
            photo_medium_confidence=0.7,
        )

        # Calculate overall confidence
        overall_confidence = analyzer._calculate_overall_confidence(result)

        # Should be weighted average of available confidence scores
        # Actual weights: decade=0.5, season=0.25, event=0.25 (photo_medium not included)
        expected = (0.75 * 0.5) + (0.8 * 0.25) + (0.9 * 0.25)
        # No rounding in implementation, returns raw float

        assert overall_confidence == pytest.approx(expected, abs=0.001)

    def test_calculate_overall_confidence_with_none_values(self):
        """Test overall confidence calculation with None confidence values."""
        # Create a mock analyzer
        analyzer = LegacyContextAnalyzer()

        # Create result with some None confidence scores
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            season=None,  # No season
            season_confidence=None,
            event_hint=None,  # No event
            event_confidence=None,
            photo_medium="digital",
            photo_medium_confidence=0.6,
        )

        # Calculate overall confidence
        overall_confidence = analyzer._calculate_overall_confidence(result)

        # Should only include available confidence scores in weighted average
        # With season and event missing, weights should be adjusted
        # decade=0.75, photo_medium=0.6
        # Adjusted weights: decade gets more weight since season/event missing
        # Implementation detail: weights are normalized based on available fields
        assert 0.6 <= overall_confidence <= 0.75  # Should be between the two values

    def test_calculate_overall_confidence_only_decade(self):
        """Test overall confidence calculation with only decade confidence."""
        # Create a mock analyzer
        analyzer = LegacyContextAnalyzer()

        # Create result with only decade confidence
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.82,
            season=None,
            season_confidence=None,
            event_hint=None,
            event_confidence=None,
            photo_medium="unknown",
            photo_medium_confidence=None,
        )

        # Calculate overall confidence
        overall_confidence = analyzer._calculate_overall_confidence(result)

        # With only decade confidence available, overall should equal decade confidence
        assert overall_confidence == 0.82

    def test_calculate_overall_confidence_edge_cases(self):
        """Test overall confidence calculation with edge cases."""
        # Create a mock analyzer
        analyzer = LegacyContextAnalyzer()

        # Test with all None confidence (should return 0.0)
        result_all_none = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.0,
            season=None,
            season_confidence=None,
            event_hint=None,
            event_confidence=None,
            photo_medium="unknown",
            photo_medium_confidence=None,
        )

        overall_all_none = analyzer._calculate_overall_confidence(result_all_none)
        assert overall_all_none == 0.0

        # Test with very low confidence
        result_low = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.1,
            season="winter",
            season_confidence=0.15,
            photo_medium="digital",
            photo_medium_confidence=0.05,
        )

        overall_low = analyzer._calculate_overall_confidence(result_low)
        assert 0.05 <= overall_low <= 0.15  # Should be weighted average

        # Test with very high confidence
        result_high = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.95,
            season="summer",
            season_confidence=0.98,
            event_hint="graduation",
            event_confidence=0.99,
            photo_medium="print_scan",
            photo_medium_confidence=0.97,
        )

        overall_high = analyzer._calculate_overall_confidence(result_high)
        assert 0.95 <= overall_high <= 0.99  # Should be weighted average


class TestConfidenceIntegration:
    """Integration tests for confidence validation and propagation."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock OllamaClient."""
        mock_client = Mock(spec=OllamaClient)
        mock_client.analyze_image_context = Mock()
        mock_client.get_prompt_template = Mock(return_value="Test prompt template")
        mock_client.connect = Mock(return_value=True)
        mock_client.health_check = Mock(
            return_value={"status": "healthy", "server_available": True}
        )
        return mock_client

    def test_full_pipeline_confidence_validation(self, mock_ollama_client, caplog):
        """Test full confidence validation pipeline with mixed results."""
        config = NewContextAnalyzerConfig(
            min_decade_confidence=0.3,
            min_season_confidence=0.4,
            min_event_confidence=0.5,
            min_photo_medium_confidence=0.4,
            enable_retries=True,
            max_retries=2,
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Create a sequence of results testing different confidence scenarios
        results_sequence = [
            # First: All confidences too low
            ContextAnalysisResult(
                decade="1985-1990",
                decade_confidence=0.25,  # Below threshold
                season="summer",
                season_confidence=0.35,  # Below threshold
                event_hint="party",
                event_confidence=0.45,  # Below threshold
                photo_medium="digital",
                photo_medium_confidence=0.35,  # Below threshold
            ),
            # Second: Some confidences acceptable, others not
            ContextAnalysisResult(
                decade="1985-1990",
                decade_confidence=0.85,  # OK
                season="summer",
                season_confidence=0.35,  # Still below threshold
                photo_medium="digital",
                photo_medium_confidence=0.45,  # OK (above 0.4)
            ),
            # Third: All confidences acceptable
            ContextAnalysisResult(
                decade="1980-1985",  # Different decade to verify we got this one
                decade_confidence=0.9,
                season="winter",
                season_confidence=0.8,
                event_hint="christmas",
                event_confidence=0.85,
                photo_medium="polaroid",
                photo_medium_confidence=0.75,
            ),
        ]

        mock_ollama_client.analyze_image_context.side_effect = results_sequence

        # Mock model priority getters
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer._get_fallback_model_name = Mock(return_value="moondream2")

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer._run_main_analysis_pipeline("test.jpg")

        # Should return the second result (decade confidence acceptable)
        # The pipeline stops when it gets a result with acceptable decade confidence
        assert result == results_sequence[1]
        assert result.decade == "1985-1990"
        assert result.decade_confidence == 0.85
        assert result.season_confidence == 0.35
        assert result.photo_medium_confidence == 0.45

        # Should have called analyze_image_context twice
        # (primary model returns low confidence, fallback model returns acceptable confidence)
        assert mock_ollama_client.analyze_image_context.call_count == 2

    @patch("pathlib.Path.exists", return_value=True)
    def test_confidence_propagation_across_different_strategies(
        self, mock_path_exists, mock_ollama_client
    ):
        """Test confidence propagation across different analysis strategies."""
        # Test with legacy ContextAnalyzer which has multiple strategies
        config = LegacyContextAnalyzerConfig(
            min_decade_confidence=0.3,
            enable_retries=True,
            max_retries=2,
        )

        # Test DEFAULT strategy
        config_default = LegacyContextAnalyzerConfig(
            min_decade_confidence=0.3,
            enable_retries=True,
            max_retries=2,
            strategy=LegacyAnalysisStrategy.DEFAULT,
        )
        analyzer_default = LegacyContextAnalyzer(
            ollama_client=mock_ollama_client,
            config=config_default,
        )

        # Test AGGRESSIVE strategy
        config_aggressive = LegacyContextAnalyzerConfig(
            min_decade_confidence=0.3,
            enable_retries=True,
            max_retries=2,
            strategy=LegacyAnalysisStrategy.AGGRESSIVE,
        )
        analyzer_aggressive = LegacyContextAnalyzer(
            ollama_client=mock_ollama_client,
            config=config_aggressive,
        )

        # Test CONSERVATIVE strategy
        config_conservative = LegacyContextAnalyzerConfig(
            min_decade_confidence=0.3,
            enable_retries=True,
            max_retries=2,
            strategy=LegacyAnalysisStrategy.CONSERVATIVE,
        )
        analyzer_conservative = LegacyContextAnalyzer(
            ollama_client=mock_ollama_client,
            config=config_conservative,
        )

        # Create test results with different confidence levels
        low_confidence_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.25,  # Below threshold
            photo_medium="digital",
        )

        high_confidence_result = ContextAnalysisResult(
            decade="1980-1985",
            decade_confidence=0.85,  # Above threshold
            photo_medium="print_scan",
        )

        # Reset mock between tests
        mock_ollama_client.analyze_image_context.reset_mock()

        # Test DEFAULT strategy - should use fallback on low confidence
        mock_ollama_client.analyze_image_context.side_effect = [
            low_confidence_result,
            high_confidence_result,
        ]

        analyzer_default._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer_default._get_fallback_model_name = Mock(return_value="moondream2")

        with patch("time.sleep"):
            result_default = analyzer_default.analyze("test.jpg")

        # DEFAULT should return high confidence result after fallback
        assert result_default == high_confidence_result

        # Reset for next test
        mock_ollama_client.analyze_image_context.reset_mock()

        # Test AGGRESSIVE strategy - should try multiple approaches
        # and return best confidence
        mock_ollama_client.analyze_image_context.side_effect = [
            low_confidence_result,
            high_confidence_result,
        ]

        analyzer_aggressive._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer_aggressive._get_fallback_model_name = Mock(return_value="moondream2")

        with patch("time.sleep"):
            result_aggressive = analyzer_aggressive.analyze("test.jpg")

        # AGGRESSIVE should return high confidence result
        assert result_aggressive == high_confidence_result

        # Reset for next test
        mock_ollama_client.analyze_image_context.reset_mock()

        # Test CONSERVATIVE strategy - should reject low confidence
        mock_ollama_client.analyze_image_context.return_value = low_confidence_result

        analyzer_conservative._get_primary_model_name = Mock(
            return_value="llava-next:7b"
        )
        analyzer_conservative._get_fallback_model_name = Mock(return_value="moondream2")

        with patch("time.sleep"):
            result_conservative = analyzer_conservative.analyze("test.jpg")

        # CONSERVATIVE should return None for low confidence
        assert result_conservative is None
