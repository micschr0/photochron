"""
Tests for ContextAnalyzer analysis strategies.

This module tests the ContextAnalyzer class with all analysis strategies:
- DEFAULT: Standard analysis with primary model fallback
- AGGRESSIVE: Try multiple models and prompts for best results
- CONSERVATIVE: Only return results with high confidence
- FAST: Use simpler prompts and skip retries for speed

Tests use mocked LLM responses to verify strategy behavior without requiring
a real Ollama server.
"""

import logging
from unittest.mock import Mock, patch

import pytest

from photochron.context.analyzer import (
    AnalysisStrategy,
    ContextAnalyzer,
    ContextAnalyzerConfig,
    FallbackStrategy,
    get_context_analyzer,
)
from photochron.models.ollama_client import (
    ContextAnalysisResult,
    ModelType,
    OllamaClient,
)


class TestContextAnalyzerStrategies:
    """Test suite for ContextAnalyzer analysis strategies."""

    @pytest.fixture
    def mock_ollama_client(self):
        """Create a mock OllamaClient."""
        mock_client = Mock(spec=OllamaClient)
        mock_client.analyze_image_context = Mock()
        mock_client.get_prompt_template = Mock(return_value="Test prompt template")
        mock_client.connect = Mock(return_value=True)
        mock_client.health_check = Mock(return_value={"status": "healthy", "server_available": True})
        return mock_client

    @pytest.fixture
    def mock_context_result(self):
        """Create a mock ContextAnalysisResult with good confidence."""
        return ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.75,
            season="summer",
            event_hint=None,
            photo_medium="print_scan",
            photo_medium_confidence=0.8,
            visual_evidence=None,
            season_confidence=None,
            event_confidence=None,
            alternative_decades=None,
            uncertainty_flag=None,
            hypothesis_notes=None,
        )

    @pytest.fixture
    def mock_context_result_low_confidence(self):
        """Create a mock ContextAnalysisResult with low confidence."""
        return ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.2,  # Below default min_decade_confidence of 0.3
            season="summer",
            event_hint=None,
            photo_medium="print_scan",
            photo_medium_confidence=0.8,
            visual_evidence=None,
            season_confidence=None,
            event_confidence=None,
            alternative_decades=None,
            uncertainty_flag=None,
            hypothesis_notes=None,
        )

    @pytest.fixture
    def mock_context_result_high_confidence(self):
        """Create a mock ContextAnalysisResult with high confidence."""
        return ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.9,
            season="summer",
            season_confidence=0.85,
            event_hint="wedding",
            event_confidence=0.8,
            photo_medium="print_scan",
            photo_medium_confidence=0.9,
            visual_evidence=["vintage clothing", "old car"],
            alternative_decades=["1975-1980", "1990-1995"],
            uncertainty_flag=False,
            hypothesis_notes=None,
        )

    def test_analyze_default_strategy_success(self, mock_ollama_client, mock_context_result):
        """Test DEFAULT strategy succeeds with primary model."""
        # Setup - disable retries for predictable call count
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.DEFAULT,
            enable_retries=False,  # Disable retries for predictable call count
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock the model priority getters
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer._get_fallback_model_name = Mock(return_value="moondream2")

        # Mock successful analysis
        mock_ollama_client.analyze_image_context.return_value = mock_context_result

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result == mock_context_result
        mock_ollama_client.analyze_image_context.assert_called_once_with(
            image_input="test.jpg",
            model_name="llava-next:7b",
            prompt_template="Test prompt template",
            use_base64=False,
        )

    def test_analyze_default_strategy_fallback_on_low_confidence(
        self,
        mock_ollama_client,
        mock_context_result_low_confidence,
        mock_context_result,
    ):
        """Test DEFAULT strategy uses fallback model when primary returns low confidence."""
        # Setup - disable retries for predictable call count
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.DEFAULT,
            enable_retries=False,  # Disable retries for predictable call count
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock the model priority getters
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer._get_fallback_model_name = Mock(return_value="moondream2")

        # Mock primary returns low confidence, fallback returns good confidence
        mock_ollama_client.analyze_image_context.side_effect = [
            mock_context_result_low_confidence,  # Primary model
            mock_context_result,  # Fallback model
        ]

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result == mock_context_result  # Should return fallback result
        assert mock_ollama_client.analyze_image_context.call_count == 2

        # Verify calls
        calls = mock_ollama_client.analyze_image_context.call_args_list
        assert calls[0].kwargs["model_name"] == "llava-next:7b"
        assert calls[1].kwargs["model_name"] == "moondream2"

    def test_analyze_default_strategy_fallback_on_primary_failure(self, mock_ollama_client, mock_context_result):
        """Test DEFAULT strategy uses fallback model when primary fails."""
        # Setup - disable retries for predictable call count
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.DEFAULT,
            enable_retries=False,  # Disable retries for predictable call count
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock the model priority getters
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer._get_fallback_model_name = Mock(return_value="moondream2")

        # Mock primary fails, fallback succeeds
        mock_ollama_client.analyze_image_context.side_effect = [
            None,  # Primary model returns None
            mock_context_result,  # Fallback model succeeds
        ]

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result == mock_context_result
        assert mock_ollama_client.analyze_image_context.call_count == 2

    def test_analyze_default_strategy_no_fallback_available(
        self, mock_ollama_client, mock_context_result_low_confidence
    ):
        """Test DEFAULT strategy when no fallback model is available."""
        # Setup
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.DEFAULT,
            model_priority=[ModelType.LLAVA_NEXT_7B],  # Only one model
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock low confidence result
        mock_ollama_client.analyze_image_context.return_value = mock_context_result_low_confidence

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        # Should return cleaned result with decade=None, decade_confidence=0.0
        assert result is not None
        assert result.decade is None
        assert result.decade_confidence == 0.0
        mock_ollama_client.analyze_image_context.assert_called_once()

    def test_analyze_aggressive_strategy_tries_multiple_models_prompts(self, mock_ollama_client, mock_context_result):
        """Test AGGRESSIVE strategy tries multiple models and prompts."""
        # Setup - disable retries to avoid 6 combinations × 3 retries = 18 calls
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.AGGRESSIVE,
            model_priority=[ModelType.LLAVA_NEXT_7B, ModelType.MOONDREAM2],
            prompt_templates=["default", "detailed_decade", "season_focused"],
            enable_retries=False,  # Disable retries for predictable call count
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock analysis attempts - aggressive strategy tries all combinations
        # 2 models × 3 prompts = 6 combinations
        # Returns result on third attempt, but continues trying all combinations
        # to find better confidence (only stops early if confidence >= 0.8)
        mock_ollama_client.analyze_image_context.side_effect = [
            None,  # First model, first prompt (default)
            None,  # First model, second prompt (detailed_decade)
            mock_context_result,  # First model, third prompt (season_focused) - confidence=0.75
            None,  # Second model, first prompt (default)
            None,  # Second model, second prompt (detailed_decade)
            None,  # Second model, third prompt (season_focused)
        ]

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result == mock_context_result
        # Should have tried all 6 combinations (2 models * 3 prompts)
        assert mock_ollama_client.analyze_image_context.call_count == 6

        # Verify get_prompt_template was called with different templates
        prompt_calls = mock_ollama_client.get_prompt_template.call_args_list
        assert len(prompt_calls) >= 6

    def test_analyze_aggressive_strategy_returns_best_confidence(self, mock_ollama_client):
        """Test AGGRESSIVE strategy returns result with highest overall confidence."""
        # Setup - disable retries for predictable call count
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.AGGRESSIVE,
            enable_retries=False,  # Disable retries for predictable call count
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Create results with different confidence levels
        low_conf_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.4,
            season="summer",
            season_confidence=0.3,
            photo_medium="print_scan",
            photo_medium_confidence=0.8,
        )

        high_conf_result = ContextAnalysisResult(
            decade="1975-1980",
            decade_confidence=0.85,
            season="winter",
            season_confidence=0.9,
            photo_medium="film_negative",
            photo_medium_confidence=0.9,
        )

        # Mock multiple attempts returning different results
        mock_ollama_client.analyze_image_context.side_effect = [
            low_conf_result,
            high_conf_result,
            low_conf_result,  # Another low confidence
        ]

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result == high_conf_result  # Should return highest confidence result
        # Should stop early when confidence >= 0.8 (high_conf_result has decade_confidence=0.85)
        # So only 2 calls should be made (low_conf_result, then high_conf_result)
        assert mock_ollama_client.analyze_image_context.call_count == 2

    def test_analyze_aggressive_strategy_early_exit_on_high_confidence(
        self, mock_ollama_client, mock_context_result_high_confidence
    ):
        """Test AGGRESSIVE strategy stops early when high confidence is achieved."""
        # Setup - disable retries for predictable call count
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.AGGRESSIVE,
            enable_retries=False,  # Disable retries for predictable call count
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock high confidence result on first attempt
        mock_ollama_client.analyze_image_context.return_value = mock_context_result_high_confidence

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result == mock_context_result_high_confidence
        mock_ollama_client.analyze_image_context.assert_called_once()  # Should stop after first high confidence

    def test_analyze_conservative_strategy_discards_low_confidence(
        self, mock_ollama_client, mock_context_result_low_confidence
    ):
        """Test CONSERVATIVE strategy discards results below confidence thresholds."""
        # Setup
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.CONSERVATIVE,
            min_decade_confidence=0.3,
            min_season_confidence=0.4,
            min_event_confidence=0.5,
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock low confidence result (decade_confidence=0.2 < 0.3)
        mock_ollama_client.analyze_image_context.return_value = mock_context_result_low_confidence

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        # Conservative strategy discards low confidence results via _validate_and_clean_result
        # which returns a cleaned result with decade=None, decade_confidence=0.0
        # not None
        assert result is not None
        assert result.decade is None
        assert result.decade_confidence == 0.0

    def test_analyze_conservative_strategy_accepts_high_confidence(
        self, mock_ollama_client, mock_context_result_high_confidence
    ):
        """Test CONSERVATIVE strategy accepts results meeting confidence thresholds."""
        # Setup
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.CONSERVATIVE,
            min_decade_confidence=0.3,
            min_season_confidence=0.4,
            min_event_confidence=0.5,
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock high confidence result (all confidences above thresholds)
        mock_ollama_client.analyze_image_context.return_value = mock_context_result_high_confidence

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result == mock_context_result_high_confidence  # Should accept high confidence result

    def test_analyze_conservative_strategy_tries_uncertainty_handling(
        self,
        mock_ollama_client,
        mock_context_result_low_confidence,
        mock_context_result,
    ):
        """Test CONSERVATIVE strategy tries uncertainty handling when default fails."""
        # Setup
        config = ContextAnalyzerConfig(strategy=AnalysisStrategy.CONSERVATIVE)
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock the model priority getters
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer._get_fallback_model_name = Mock(return_value=None)  # No fallback model

        # Mock default analysis returns low confidence (gets discarded),
        # then uncertainty handling succeeds
        mock_ollama_client.analyze_image_context.side_effect = [
            mock_context_result_low_confidence,  # Default analysis (discarded)
            mock_context_result,  # Uncertainty handling
        ]

        # Mock get_prompt_template to return different templates
        # The conservative strategy calls get_prompt_template with "uncertainty_handling"
        # for the uncertainty handling fallback
        def get_prompt_template_side_effect(template_name):
            if template_name == "uncertainty_handling":
                return "Uncertainty handling prompt"
            return "Default prompt"

        mock_ollama_client.get_prompt_template.side_effect = get_prompt_template_side_effect

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result == mock_context_result
        assert mock_ollama_client.analyze_image_context.call_count == 2

        # Verify uncertainty handling prompt was used
        calls = mock_ollama_client.analyze_image_context.call_args_list
        assert calls[1].kwargs["prompt_template"] == "Uncertainty handling prompt"

    def test_analyze_fast_strategy_uses_simple_prompt(self, mock_ollama_client, mock_context_result):
        """Test FAST strategy uses simple fallback prompt for speed."""
        # Setup - disable retries for predictable call count
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.FAST,
            enable_retries=False,  # Disable retries for predictable call count
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock the model priority getter
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")

        # Mock successful analysis
        mock_ollama_client.analyze_image_context.return_value = mock_context_result

        # Mock get_prompt_template to return simple_fallback for FAST strategy
        def get_prompt_template_side_effect(template_name):
            if template_name == "simple_fallback":
                return "Simple fallback prompt"
            return "Default prompt"

        mock_ollama_client.get_prompt_template.side_effect = get_prompt_template_side_effect

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result == mock_context_result
        mock_ollama_client.analyze_image_context.assert_called_once_with(
            image_input="test.jpg",
            model_name="llava-next:7b",
            prompt_template="Simple fallback prompt",
            use_base64=False,
        )

    def test_analyze_fast_strategy_no_retries(self, mock_ollama_client, mock_context_result):
        """Test FAST strategy should work with retries disabled."""
        # Setup - Note: FAST strategy doesn't disable retries by itself,
        # but we can test with retries disabled in config
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.FAST,
            enable_retries=False,  # Disable retries for speed
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock the model priority getter
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")

        # Mock successful analysis
        mock_ollama_client.analyze_image_context.return_value = mock_context_result

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result == mock_context_result
        mock_ollama_client.analyze_image_context.assert_called_once()

    def test_analyze_strategy_override(self, mock_ollama_client, mock_context_result):
        """Test analyze() method can override default strategy."""
        # Setup with DEFAULT strategy and disable retries
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.DEFAULT,
            enable_retries=False,  # Disable retries for predictable behavior
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock the model priority getter
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")

        # Mock successful analysis
        mock_ollama_client.analyze_image_context.return_value = mock_context_result

        # Execute with FAST strategy override
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg", strategy=AnalysisStrategy.FAST)

        # Verify
        assert result == mock_context_result
        # Should use FAST strategy logic (simple_fallback prompt)
        mock_ollama_client.get_prompt_template.assert_called_with("simple_fallback")

    def test_analyze_fallback_strategy_application(self, mock_ollama_client, mock_context_result):
        """Test fallback strategy is applied when primary analysis fails."""
        # Setup with SIMPLE fallback and disable retries
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.DEFAULT,
            fallback_strategy=FallbackStrategy.SIMPLE,
            enable_retries=False,  # Disable retries for predictable call count
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock the model priority getters
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer._get_fallback_model_name = Mock(return_value=None)  # No internal fallback

        # Mock primary analysis returns None, fallback succeeds
        mock_ollama_client.analyze_image_context.side_effect = [
            None,  # Primary analysis fails
            mock_context_result,  # Fallback succeeds
        ]

        # Mock get_prompt_template for fallback
        # The fallback strategy with SIMPLE calls get_prompt_template with "simple_fallback"
        def get_prompt_template_side_effect(template_name):
            if template_name == "simple_fallback":
                return "Simple fallback prompt"
            return "Default prompt"

        mock_ollama_client.get_prompt_template.side_effect = get_prompt_template_side_effect

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result == mock_context_result
        assert mock_ollama_client.analyze_image_context.call_count == 2

        # Verify fallback used simple_fallback prompt
        calls = mock_ollama_client.analyze_image_context.call_args_list
        assert calls[1].kwargs["prompt_template"] == "Simple fallback prompt"

    def test_analyze_no_fallback_strategy(self, mock_ollama_client):
        """Test NONE fallback strategy returns None when analysis fails."""
        # Setup with NONE fallback and disable retries
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.DEFAULT,
            fallback_strategy=FallbackStrategy.NONE,
            enable_retries=False,  # Disable retries for predictable call count
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock the model priority getters
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer._get_fallback_model_name = Mock(return_value=None)  # No internal fallback

        # Mock analysis fails
        mock_ollama_client.analyze_image_context.return_value = None

        # Execute
        with patch("pathlib.Path.exists", return_value=True):
            result = analyzer.analyze("test.jpg")

        # Verify
        assert result is None  # No fallback, should return None
        mock_ollama_client.analyze_image_context.assert_called_once()

    def test_analyze_unknown_strategy_falls_back_to_default(self, mock_ollama_client, mock_context_result, caplog):
        """Test unknown strategy falls back to DEFAULT strategy."""
        # Setup with a mock strategy (not in enum) and disable retries
        config = ContextAnalyzerConfig(
            enable_retries=False,  # Disable retries for predictable behavior
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Mock the model priority getter
        analyzer._get_primary_model_name = Mock(return_value="llava-next:7b")

        # Mock successful analysis
        mock_ollama_client.analyze_image_context.return_value = mock_context_result

        # Execute with unknown strategy string
        with patch("pathlib.Path.exists", return_value=True):
            with caplog.at_level(logging.WARNING):
                # Pass a string that's not in AnalysisStrategy enum
                result = analyzer.analyze("test.jpg", strategy="UNKNOWN_STRATEGY")

        # Verify
        assert result == mock_context_result
        # Should log warning about unknown strategy
        assert "Unknown strategy" in caplog.text
        assert "using default" in caplog.text

    def test_analyze_image_not_found(self, mock_ollama_client, caplog):
        """Test analyze() returns None when image file doesn't exist."""
        # Setup
        config = ContextAnalyzerConfig()
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Execute with non-existent file
        with patch("pathlib.Path.exists", return_value=False):
            with caplog.at_level(logging.ERROR):
                result = analyzer.analyze("nonexistent.jpg")

        # Verify
        assert result is None
        assert "Image file does not exist" in caplog.text

    def test_calculate_overall_confidence(self, mock_ollama_client):
        """Test _calculate_overall_confidence() method."""
        # Setup
        config = ContextAnalyzerConfig()
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Test case 1: All confidences available
        result1 = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.8,
            season="summer",
            season_confidence=0.7,
            event_hint="wedding",
            event_confidence=0.6,
            photo_medium="print_scan",
            photo_medium_confidence=0.9,
        )

        # Expected: (0.8*0.5 + 0.7*0.25 + 0.6*0.25) / (0.5+0.25+0.25) = (0.4 + 0.175 + 0.15) / 1.0 = 0.725
        confidence1 = analyzer._calculate_overall_confidence(result1)
        assert abs(confidence1 - 0.725) < 0.001

        # Test case 2: Only decade confidence available
        result2 = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.9,
            season=None,  # No season
            event_hint=None,  # No event
            photo_medium="print_scan",
            photo_medium_confidence=0.9,
        )

        # Expected: (0.9*0.5) / 0.5 = 0.9
        confidence2 = analyzer._calculate_overall_confidence(result2)
        assert abs(confidence2 - 0.9) < 0.001

        # Test case 3: Decade and season confidence only
        result3 = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.7,
            season="winter",
            season_confidence=0.8,
            event_hint=None,  # No event
            photo_medium="print_scan",
            photo_medium_confidence=0.9,
        )

        # Expected: (0.7*0.5 + 0.8*0.25) / (0.5+0.25) = (0.35 + 0.2) / 0.75 = 0.55 / 0.75 ≈ 0.7333
        confidence3 = analyzer._calculate_overall_confidence(result3)
        assert abs(confidence3 - 0.73333) < 0.001

    def test_validate_and_clean_result(self, mock_ollama_client):
        """Test _validate_and_clean_result() method."""
        # Setup
        config = ContextAnalyzerConfig(
            min_decade_confidence=0.3,
            min_season_confidence=0.4,
            min_event_confidence=0.5,
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Test case: Low confidence fields get cleared
        result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.2,  # Below threshold
            season="summer",
            season_confidence=0.3,  # Below threshold
            event_hint="wedding",
            event_confidence=0.4,  # Below threshold
            photo_medium="print_scan",
            photo_medium_confidence=0.8,
            visual_evidence=["test evidence"],
            alternative_decades=["1975-1980", "1990-1995"],
        )

        cleaned = analyzer._validate_and_clean_result(result)

        # Verify low confidence fields are cleared
        assert cleaned.decade is None
        assert cleaned.decade_confidence == 0.0
        assert cleaned.season is None
        assert cleaned.season_confidence is None
        assert cleaned.event_hint is None
        assert cleaned.event_confidence is None
        assert cleaned.alternative_decades is None  # Should be cleared when decade is cleared

        # Verify other fields remain
        assert cleaned.photo_medium == "print_scan"
        assert cleaned.photo_medium_confidence == 0.8
        assert cleaned.visual_evidence == ["test evidence"]

    def test_get_available_strategies(self, mock_ollama_client):
        """Test get_available_strategies() method."""
        # Setup
        config = ContextAnalyzerConfig()
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Execute
        strategies = analyzer.get_available_strategies()

        # Verify
        expected = ["default", "aggressive", "conservative", "fast"]
        assert set(strategies) == set(expected)

    def test_get_available_fallback_strategies(self, mock_ollama_client):
        """Test get_available_fallback_strategies() method."""
        # Setup
        config = ContextAnalyzerConfig()
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Execute
        fallback_strategies = analyzer.get_available_fallback_strategies()

        # Verify
        expected = ["none", "simple", "uncertainty", "multi_hypothesis"]
        assert set(fallback_strategies) == set(expected)

    def test_health_check(self, mock_ollama_client):
        """Test health_check() method."""
        # Setup
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.AGGRESSIVE,
            fallback_strategy=FallbackStrategy.MULTI_HYPOTHESIS,
            min_decade_confidence=0.3,
            min_season_confidence=0.4,
            min_event_confidence=0.5,
        )
        analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

        # Execute
        health = analyzer.health_check()

        # Verify
        assert "status" in health
        assert "analyzer_config" in health
        assert "ollama_health" in health

        config_info = health["analyzer_config"]
        assert config_info["strategy"] == "aggressive"
        assert config_info["fallback_strategy"] == "multi_hypothesis"
        assert config_info["min_decade_confidence"] == 0.3
        assert config_info["min_season_confidence"] == 0.4
        assert config_info["min_event_confidence"] == 0.5

    def test_get_context_analyzer_singleton(self, mock_ollama_client):
        """Test get_context_analyzer() returns singleton instance."""
        # First call should create new instance
        analyzer1 = get_context_analyzer(ollama_client=mock_ollama_client)

        # Second call should return same instance
        analyzer2 = get_context_analyzer()

        # Verify they're the same object
        assert analyzer1 is analyzer2

        # Verify it was initialized with our mock client
        assert analyzer1.ollama_client is mock_ollama_client

    def test_get_context_analyzer_with_config(self):
        """Test get_context_analyzer() with custom configuration."""
        # Reset singleton before test
        import photochron.context.analyzer as analyzer_module

        analyzer_module._default_analyzer = None

        # Setup custom config
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.CONSERVATIVE,
            fallback_strategy=FallbackStrategy.UNCERTAINTY,
            min_decade_confidence=0.5,
        )

        # Create analyzer with custom config
        analyzer = get_context_analyzer(config=config)

        # Verify config was applied
        assert analyzer.config.strategy == AnalysisStrategy.CONSERVATIVE
        assert analyzer.config.fallback_strategy == FallbackStrategy.UNCERTAINTY
        assert analyzer.config.min_decade_confidence == 0.5

    def test_get_context_analyzer_reset_global(self):
        """Test get_context_analyzer() global instance can be reset."""
        # Import the module to access the global variable
        import photochron.context.analyzer as analyzer_module

        # Save original
        original_analyzer = analyzer_module._default_analyzer

        try:
            # Set to None to force recreation
            analyzer_module._default_analyzer = None

            # Create new analyzer
            analyzer1 = get_context_analyzer()

            # Set to None again
            analyzer_module._default_analyzer = None

            # Create another analyzer
            analyzer2 = get_context_analyzer()

            # They should be different objects since we reset in between
            assert analyzer1 is not analyzer2
        finally:
            # Restore original
            analyzer_module._default_analyzer = original_analyzer
