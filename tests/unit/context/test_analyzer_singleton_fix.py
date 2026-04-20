"""
Tests for the singleton bug fix in analyzer.py.

This module tests the fixed get_context_analyzer() function with:
1. Test config updates when singleton already exists
2. Test ValueError for different ollama_client
3. Test warning logging for ollama_config
4. Test backward compatibility
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import logging
from typing import Optional

from photochron.context.analyzer import (
    ContextAnalyzer,
    ContextAnalyzerConfig,
    AnalysisStrategy,
    FallbackStrategy,
    get_context_analyzer,
    _default_analyzer,
)
from photochron.models.ollama_client import (
    OllamaClient,
    OllamaConfig,
    ModelType,
)


class TestSingletonBugFix:
    """Test suite for singleton bug fix in get_context_analyzer()."""

    def setup_method(self):
        """Reset the singleton before each test."""
        # Import the module to access the global variable
        import photochron.context.analyzer as analyzer_module

        analyzer_module._default_analyzer = None

    def teardown_method(self):
        """Reset the singleton after each test."""
        # Import the module to access the global variable
        import photochron.context.analyzer as analyzer_module

        analyzer_module._default_analyzer = None

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
    def mock_ollama_client2(self):
        """Create a second mock OllamaClient."""
        mock_client = Mock(spec=OllamaClient)
        mock_client.analyze_image_context = Mock()
        mock_client.get_prompt_template = Mock(return_value="Test prompt template 2")
        mock_client.connect = Mock(return_value=True)
        mock_client.health_check = Mock(
            return_value={"status": "healthy", "server_available": True}
        )
        return mock_client

    def test_get_context_analyzer_creates_singleton_first_time(
        self, mock_ollama_client
    ):
        """Test get_context_analyzer() creates singleton on first call."""
        # First call should create new instance
        analyzer1 = get_context_analyzer(ollama_client=mock_ollama_client)

        # Verify it was created
        assert analyzer1 is not None
        assert analyzer1.ollama_client is mock_ollama_client

        # Second call should return same instance
        analyzer2 = get_context_analyzer()

        # Verify they're the same object
        assert analyzer1 is analyzer2

    def test_get_context_analyzer_updates_config_when_singleton_exists(
        self, mock_ollama_client
    ):
        """Test get_context_analyzer() updates config when singleton already exists."""
        # Create initial analyzer with default config
        analyzer1 = get_context_analyzer(ollama_client=mock_ollama_client)
        initial_config = analyzer1.config

        # Verify initial config has default values
        assert initial_config.strategy == AnalysisStrategy.DEFAULT
        assert initial_config.fallback_strategy == FallbackStrategy.SIMPLE
        assert initial_config.min_decade_confidence == 0.3

        # Create new config with different values
        new_config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.CONSERVATIVE,
            fallback_strategy=FallbackStrategy.UNCERTAINTY,
            min_decade_confidence=0.5,
            min_season_confidence=0.6,
            min_event_confidence=0.7,
        )

        # Get analyzer again with new config
        analyzer2 = get_context_analyzer(config=new_config)

        # Verify it's the same instance
        assert analyzer1 is analyzer2

        # Verify config was updated
        assert analyzer2.config.strategy == AnalysisStrategy.CONSERVATIVE
        assert analyzer2.config.fallback_strategy == FallbackStrategy.UNCERTAINTY
        assert analyzer2.config.min_decade_confidence == 0.5
        assert analyzer2.config.min_season_confidence == 0.6
        assert analyzer2.config.min_event_confidence == 0.7

    def test_get_context_analyzer_raises_value_error_for_different_ollama_client(
        self, mock_ollama_client, mock_ollama_client2
    ):
        """Test get_context_analyzer() raises ValueError when trying to use different ollama_client."""
        # Create initial analyzer with first client
        analyzer1 = get_context_analyzer(ollama_client=mock_ollama_client)

        # Try to get analyzer with different client - should raise ValueError
        with pytest.raises(ValueError) as exc_info:
            get_context_analyzer(ollama_client=mock_ollama_client2)

        # Verify error message
        assert "Cannot get context analyzer with different ollama_client" in str(
            exc_info.value
        )
        assert "singleton already exists" in str(exc_info.value)

        # Verify the original analyzer is still accessible
        analyzer2 = get_context_analyzer()
        assert analyzer1 is analyzer2
        assert analyzer2.ollama_client is mock_ollama_client  # Not mock_ollama_client2

    def test_get_context_analyzer_allows_same_ollama_client(self, mock_ollama_client):
        """Test get_context_analyzer() allows same ollama_client when singleton exists."""
        # Create initial analyzer
        analyzer1 = get_context_analyzer(ollama_client=mock_ollama_client)

        # Get analyzer again with same client - should work
        analyzer2 = get_context_analyzer(ollama_client=mock_ollama_client)

        # Verify it's the same instance
        assert analyzer1 is analyzer2
        assert analyzer2.ollama_client is mock_ollama_client

    def test_get_context_analyzer_allows_none_ollama_client_when_singleton_exists(
        self, mock_ollama_client
    ):
        """Test get_context_analyzer() allows None ollama_client when singleton exists."""
        # Create initial analyzer
        analyzer1 = get_context_analyzer(ollama_client=mock_ollama_client)

        # Get analyzer again with None client - should work (returns existing)
        analyzer2 = get_context_analyzer(ollama_client=None)

        # Verify it's the same instance
        assert analyzer1 is analyzer2
        assert analyzer2.ollama_client is mock_ollama_client

    def test_get_context_analyzer_warns_for_ollama_config_when_singleton_exists(
        self, mock_ollama_client, caplog
    ):
        """Test get_context_analyzer() warns when ollama_config is provided but singleton exists."""
        # Create initial analyzer
        analyzer1 = get_context_analyzer(ollama_client=mock_ollama_client)

        # Create an OllamaConfig
        ollama_config = OllamaConfig(
            host="http://localhost:11434",
            timeout=30,
            max_retries=3,
        )

        # Get analyzer again with ollama_config - should log warning
        with caplog.at_level(logging.WARNING):
            analyzer2 = get_context_analyzer(ollama_config=ollama_config)

        # Verify warning was logged
        assert (
            "get_context_analyzer() called with ollama_config when singleton already exists"
            in caplog.text
        )
        assert "Ollama configuration is being ignored" in caplog.text

        # Verify it's the same instance
        assert analyzer1 is analyzer2

    def test_get_context_analyzer_backward_compatibility_none_params(
        self, mock_ollama_client
    ):
        """Test get_context_analyzer() backward compatibility with None parameters."""
        # Create initial analyzer
        analyzer1 = get_context_analyzer(
            ollama_client=mock_ollama_client,
            config=None,
            ollama_config=None,
        )

        # Get analyzer again with all None params
        analyzer2 = get_context_analyzer(
            ollama_client=None,
            config=None,
            ollama_config=None,
        )

        # Verify it's the same instance
        assert analyzer1 is analyzer2

    def test_get_context_analyzer_with_ollama_config_first_time(self):
        """Test get_context_analyzer() uses ollama_config when creating singleton first time."""
        # Reset singleton
        import photochron.context.analyzer as analyzer_module

        analyzer_module._default_analyzer = None

        # Create OllamaConfig
        ollama_config = OllamaConfig(
            host="http://localhost:11434",
            timeout=30,
            max_retries=3,
        )

        # Create config
        config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.AGGRESSIVE,
            enable_retries=True,
            max_retries=3,
        )

        # Get analyzer with ollama_config (no ollama_client provided)
        # This should create a new OllamaClient with the provided config
        analyzer = get_context_analyzer(
            ollama_client=None,
            config=config,
            ollama_config=ollama_config,
        )

        # Verify analyzer was created
        assert analyzer is not None
        assert analyzer.config.strategy == AnalysisStrategy.AGGRESSIVE
        assert analyzer.config.enable_retries == True
        assert analyzer.config.max_retries == 3

        # Note: We can't easily verify the OllamaClient was created with the right config
        # without mocking the OllamaClient constructor, but the test shows the code path works

    def test_get_context_analyzer_config_update_replaces_entire_config(
        self, mock_ollama_client
    ):
        """Test config update replaces entire config object (not merging fields)."""
        # Create initial analyzer with custom config
        initial_config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.AGGRESSIVE,
            fallback_strategy=FallbackStrategy.MULTI_HYPOTHESIS,
            min_decade_confidence=0.4,
            min_season_confidence=0.5,
            min_event_confidence=0.6,
            enable_retries=True,
            max_retries=5,
            use_base64=True,
            prompt_templates=["default", "detailed"],
            model_priority=[ModelType.LLAVA_NEXT_7B, ModelType.MOONDREAM2],
        )

        analyzer1 = get_context_analyzer(
            ollama_client=mock_ollama_client,
            config=initial_config,
        )

        # Create partial config update - only specifying some fields
        # Other fields will get default values
        update_config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.CONSERVATIVE,
            min_decade_confidence=0.7,
            # Note: Other fields not specified will get default values
        )

        # Get analyzer with partial update
        analyzer2 = get_context_analyzer(config=update_config)

        # Verify it's the same instance
        assert analyzer1 is analyzer2

        # Verify updated fields
        assert analyzer2.config.strategy == AnalysisStrategy.CONSERVATIVE
        assert analyzer2.config.min_decade_confidence == 0.7

        # Verify other fields get DEFAULT values (not preserved from initial config)
        # This is because the entire config object is replaced
        assert (
            analyzer2.config.fallback_strategy == FallbackStrategy.SIMPLE
        )  # Default, not MULTI_HYPOTHESIS
        assert analyzer2.config.min_season_confidence == 0.4  # Default, not 0.5
        assert analyzer2.config.min_event_confidence == 0.5  # Default, not 0.6
        assert analyzer2.config.enable_retries == True  # Default matches
        assert analyzer2.config.max_retries == 2  # Default, not 5
        assert analyzer2.config.use_base64 == False  # Default, not True
        # prompt_templates default is ["default", "detailed_decade", "season_focused", "event_detection"]
        assert "default" in analyzer2.config.prompt_templates
        assert "detailed_decade" in analyzer2.config.prompt_templates
        # model_priority default is [ModelType.LLAVA_NEXT_7B, ModelType.MOONDREAM2]
        assert analyzer2.config.model_priority == [
            ModelType.LLAVA_NEXT_7B,
            ModelType.MOONDREAM2,
        ]

    def test_get_context_analyzer_reset_singleton_workaround(
        self, mock_ollama_client, mock_ollama_client2
    ):
        """Test the workaround for resetting singleton mentioned in error message."""
        # Create initial analyzer with first client
        analyzer1 = get_context_analyzer(ollama_client=mock_ollama_client)

        # Try to get analyzer with different client - should fail
        with pytest.raises(ValueError):
            get_context_analyzer(ollama_client=mock_ollama_client2)

        # Reset singleton as mentioned in error message
        import photochron.context.analyzer as analyzer_module

        analyzer_module._default_analyzer = None

        # Now should be able to create analyzer with different client
        analyzer2 = get_context_analyzer(ollama_client=mock_ollama_client2)

        # Verify it's a different instance
        assert analyzer1 is not analyzer2
        assert analyzer2.ollama_client is mock_ollama_client2

    def test_get_context_analyzer_logs_info_on_config_update(
        self, mock_ollama_client, caplog
    ):
        """Test get_context_analyzer() logs info when updating config."""
        # Create initial analyzer
        analyzer1 = get_context_analyzer(ollama_client=mock_ollama_client)

        # Create new config
        new_config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.CONSERVATIVE,
            min_decade_confidence=0.8,
        )

        # Get analyzer with new config
        with caplog.at_level(logging.INFO):
            analyzer2 = get_context_analyzer(config=new_config)

        # Verify info was logged
        assert "Updating existing ContextAnalyzer configuration" in caplog.text

        # Verify it's the same instance
        assert analyzer1 is analyzer2

    def test_get_context_analyzer_no_warning_when_ollama_config_is_none(
        self, mock_ollama_client, caplog
    ):
        """Test get_context_analyzer() doesn't warn when ollama_config is None."""
        # Create initial analyzer
        analyzer1 = get_context_analyzer(ollama_client=mock_ollama_client)

        # Get analyzer again with None ollama_config - should not warn
        with caplog.at_level(logging.WARNING):
            analyzer2 = get_context_analyzer(ollama_config=None)

        # Verify no warning was logged
        assert (
            "get_context_analyzer() called with ollama_config when singleton already exists"
            not in caplog.text
        )

        # Verify it's the same instance
        assert analyzer1 is analyzer2

    def test_get_context_analyzer_combined_parameters(self, mock_ollama_client):
        """Test get_context_analyzer() with combined parameters when singleton exists."""
        # Create initial analyzer
        initial_config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.DEFAULT,
            min_decade_confidence=0.3,
        )
        analyzer1 = get_context_analyzer(
            ollama_client=mock_ollama_client,
            config=initial_config,
        )

        # Create OllamaConfig (should be ignored with warning)
        ollama_config = OllamaConfig(timeout=60)

        # Create update config
        update_config = ContextAnalyzerConfig(
            strategy=AnalysisStrategy.FAST,
            min_decade_confidence=0.5,
        )

        # Get analyzer with combined parameters
        # ollama_client=None means use existing
        # config=update_config should update config
        # ollama_config should be ignored with warning
        analyzer2 = get_context_analyzer(
            ollama_client=None,  # Use existing
            config=update_config,
            ollama_config=ollama_config,
        )

        # Verify it's the same instance
        assert analyzer1 is analyzer2

        # Verify config was updated
        assert analyzer2.config.strategy == AnalysisStrategy.FAST
        assert analyzer2.config.min_decade_confidence == 0.5

        # Verify client is still the same
        assert analyzer2.ollama_client is mock_ollama_client
