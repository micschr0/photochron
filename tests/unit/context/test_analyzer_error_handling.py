"""
Tests for ContextAnalyzer error handling, specifically the _with_retry() method.

This module tests the improved error handling in ContextAnalyzer._with_retry()
method, including:
- Network errors (ConnectionError, TimeoutError, etc.)
- Model not found errors
- Other exceptions
- enable_retries=False case
- Clear error messages for different error types
"""

import logging
from unittest.mock import Mock, patch

import pytest

from photochron.context.analyzer import (
    ContextAnalyzer,
    ContextAnalyzerConfig,
)
from photochron.models.ollama_client import (
    ContextAnalysisResult,
    ModelType,
    OllamaClient,
)

# Import ollama exceptions if available (availability check only)
try:
    from ollama import RequestError, ResponseError  # noqa: F401

    HAS_OLLAMA_EXCEPTIONS = True
except ImportError:
    HAS_OLLAMA_EXCEPTIONS = False


class TestContextAnalyzerWithRetry:
    """Test suite for ContextAnalyzer._with_retry() error handling."""

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
    def analyzer_with_retries(self, mock_ollama_client):
        """Create ContextAnalyzer with retries enabled."""
        config = ContextAnalyzerConfig(
            enable_retries=True,
            max_retries=2,
            model_priority=[ModelType.LLAVA_NEXT_7B, ModelType.MOONDREAM2],
        )
        return ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

    @pytest.fixture
    def analyzer_without_retries(self, mock_ollama_client):
        """Create ContextAnalyzer with retries disabled."""
        config = ContextAnalyzerConfig(
            enable_retries=False,
            max_retries=2,  # Should be ignored when enable_retries=False
            model_priority=[ModelType.LLAVA_NEXT_7B, ModelType.MOONDREAM2],
        )
        return ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

    @pytest.fixture
    def mock_context_result(self):
        """Create a mock ContextAnalysisResult."""
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

    def test_with_retry_success_on_first_attempt(self, analyzer_with_retries, mock_ollama_client, mock_context_result):
        """Test _with_retry() when operation succeeds on first attempt."""
        # Setup
        mock_ollama_client.analyze_image_context.return_value = mock_context_result

        # Execute
        result = analyzer_with_retries._with_retry(
            operation=lambda: mock_ollama_client.analyze_image_context(
                image_input="test.jpg",
                model_name="llava-next:7b",
                prompt_template="Test prompt",
            ),
            operation_name="Test analysis",
        )

        # Verify
        assert result == mock_context_result
        mock_ollama_client.analyze_image_context.assert_called_once()

    def test_with_retry_success_on_second_attempt(self, analyzer_with_retries, mock_ollama_client, mock_context_result):
        """Test _with_retry() when operation fails first time but succeeds on retry."""
        # Setup - fail first time, succeed second time
        mock_ollama_client.analyze_image_context.side_effect = [
            ConnectionError("Connection refused"),
            mock_context_result,
        ]

        # Execute
        with patch("time.sleep") as mock_sleep:
            result = analyzer_with_retries._with_retry(
                operation=lambda: mock_ollama_client.analyze_image_context(
                    image_input="test.jpg",
                    model_name="llava-next:7b",
                    prompt_template="Test prompt",
                ),
                operation_name="Test analysis",
            )

        # Verify
        assert result == mock_context_result
        assert mock_ollama_client.analyze_image_context.call_count == 2
        mock_sleep.assert_called_once()  # Should sleep before retry

    def test_with_retry_all_attempts_fail_network_error(self, analyzer_with_retries, mock_ollama_client, caplog):
        """Test _with_retry() when all attempts fail with network error."""
        # Setup - always fail with network error
        mock_ollama_client.analyze_image_context.side_effect = ConnectionError("Connection refused")

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.ERROR):
                result = analyzer_with_retries._with_retry(
                    operation=lambda: mock_ollama_client.analyze_image_context(
                        image_input="test.jpg",
                        model_name="llava-next:7b",
                        prompt_template="Test prompt",
                    ),
                    operation_name="Test analysis",
                )

        # Verify
        assert result is None
        assert mock_ollama_client.analyze_image_context.call_count == 3  # Initial + 2 retries
        # Check that error was logged
        assert "Test analysis failed after" in caplog.text
        assert "Ollama connection failed" in caplog.text
        assert "check if Ollama server is running" in caplog.text

    def test_with_retry_model_not_found_error(self, analyzer_with_retries, mock_ollama_client, caplog):
        """Test _with_retry() with model not found error."""
        # Setup - fail with model not found error
        error_msg = "model 'llava-next:7b' not found, try pulling it first"
        mock_ollama_client.analyze_image_context.side_effect = Exception(error_msg)

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.ERROR):
                result = analyzer_with_retries._with_retry(
                    operation=lambda: mock_ollama_client.analyze_image_context(
                        image_input="test.jpg",
                        model_name="llava-next:7b",
                        prompt_template="Test prompt",
                    ),
                    operation_name="Test analysis",
                )

        # Verify
        assert result is None
        # Check that model not found error was logged with helpful message
        assert "Model 'llava-next:7b' not found" in caplog.text
        assert "verify model is pulled with 'ollama pull llava-next:7b'" in caplog.text

    def test_with_retry_ollama_request_error(self, analyzer_with_retries, mock_ollama_client, caplog):
        """Test _with_retry() with Ollama RequestError."""
        if not HAS_OLLAMA_EXCEPTIONS:
            pytest.skip("Ollama exceptions not available")

        # Setup - fail with RequestError
        from ollama import RequestError

        error_msg = "model 'llava-next:7b' not found"
        mock_ollama_client.analyze_image_context.side_effect = RequestError(error_msg)

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.ERROR):
                result = analyzer_with_retries._with_retry(
                    operation=lambda: mock_ollama_client.analyze_image_context(
                        image_input="test.jpg",
                        model_name="llava-next:7b",
                        prompt_template="Test prompt",
                    ),
                    operation_name="Test analysis",
                )

        # Verify
        assert result is None
        # Check that Ollama-specific error was logged
        assert "Model 'llava-next:7b' not found" in caplog.text
        assert "verify model is pulled with 'ollama pull llava-next:7b'" in caplog.text

    def test_with_retry_ollama_response_error(self, analyzer_with_retries, mock_ollama_client, caplog):
        """Test _with_retry() with Ollama ResponseError."""
        if not HAS_OLLAMA_EXCEPTIONS:
            pytest.skip("Ollama exceptions not available")

        # Setup - fail with ResponseError (not model not found)
        from ollama import ResponseError

        error_msg = "Server error: internal server error"
        mock_ollama_client.analyze_image_context.side_effect = ResponseError(error_msg)

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.ERROR):
                result = analyzer_with_retries._with_retry(
                    operation=lambda: mock_ollama_client.analyze_image_context(
                        image_input="test.jpg",
                        model_name="llava-next:7b",
                        prompt_template="Test prompt",
                    ),
                    operation_name="Test analysis",
                )

        # Verify
        assert result is None
        # Check that generic Ollama error was logged
        assert "Ollama error:" in caplog.text
        assert "Server error: internal server error" in caplog.text

    def test_with_retry_timeout_error(self, analyzer_with_retries, mock_ollama_client, caplog):
        """Test _with_retry() with TimeoutError."""
        # Setup - fail with TimeoutError
        mock_ollama_client.analyze_image_context.side_effect = TimeoutError("Request timed out")

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.ERROR):
                result = analyzer_with_retries._with_retry(
                    operation=lambda: mock_ollama_client.analyze_image_context(
                        image_input="test.jpg",
                        model_name="llava-next:7b",
                        prompt_template="Test prompt",
                    ),
                    operation_name="Test analysis",
                )

        # Verify
        assert result is None
        # Check that timeout error was logged with server check message
        assert "Ollama connection failed" in caplog.text
        assert "check if Ollama server is running" in caplog.text

    def test_with_retry_connection_refused_error(self, analyzer_with_retries, mock_ollama_client, caplog):
        """Test _with_retry() with ConnectionRefusedError."""
        # Setup - fail with ConnectionRefusedError
        mock_ollama_client.analyze_image_context.side_effect = ConnectionRefusedError("Connection refused")

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.ERROR):
                result = analyzer_with_retries._with_retry(
                    operation=lambda: mock_ollama_client.analyze_image_context(
                        image_input="test.jpg",
                        model_name="llava-next:7b",
                        prompt_template="Test prompt",
                    ),
                    operation_name="Test analysis",
                )

        # Verify
        assert result is None
        # Check that connection refused error was logged with server check message
        assert "Ollama connection failed" in caplog.text
        assert "check if Ollama server is running" in caplog.text

    def test_with_retry_os_error(self, analyzer_with_retries, mock_ollama_client, caplog):
        """Test _with_retry() with OSError (e.g., socket timeout)."""
        # Setup - fail with OSError
        mock_ollama_client.analyze_image_context.side_effect = OSError("Socket timeout")

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.ERROR):
                result = analyzer_with_retries._with_retry(
                    operation=lambda: mock_ollama_client.analyze_image_context(
                        image_input="test.jpg",
                        model_name="llava-next:7b",
                        prompt_template="Test prompt",
                    ),
                    operation_name="Test analysis",
                )

        # Verify
        assert result is None
        # Check that OS error was logged with server check message
        assert "Ollama connection failed" in caplog.text
        assert "check if Ollama server is running" in caplog.text

    def test_with_retry_generic_exception(self, analyzer_with_retries, mock_ollama_client, caplog):
        """Test _with_retry() with generic exception."""
        # Setup - fail with generic exception
        mock_ollama_client.analyze_image_context.side_effect = ValueError("Invalid image format")

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.ERROR):
                result = analyzer_with_retries._with_retry(
                    operation=lambda: mock_ollama_client.analyze_image_context(
                        image_input="test.jpg",
                        model_name="llava-next:7b",
                        prompt_template="Test prompt",
                    ),
                    operation_name="Test analysis",
                )

        # Verify
        assert result is None
        # Check that generic error was logged
        assert "Test analysis failed after" in caplog.text
        assert "Invalid image format" in caplog.text

    def test_with_retry_operation_returns_none(self, analyzer_with_retries, mock_ollama_client, caplog):
        """Test _with_retry() when operation returns None (not an exception)."""
        # Setup - operation returns None
        mock_ollama_client.analyze_image_context.return_value = None

        # Execute with log capture
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer_with_retries._with_retry(
                    operation=lambda: mock_ollama_client.analyze_image_context(
                        image_input="test.jpg",
                        model_name="llava-next:7b",
                        prompt_template="Test prompt",
                    ),
                    operation_name="Test analysis",
                )

        # Verify
        assert result is None
        # Should try all retries when operation returns None
        assert mock_ollama_client.analyze_image_context.call_count == 3
        # Check debug log for None return
        assert "returned None on attempt" in caplog.text

    def test_with_retry_disabled_success(self, analyzer_without_retries, mock_ollama_client, mock_context_result):
        """Test _with_retry() with retries disabled when operation succeeds."""
        # Setup
        mock_ollama_client.analyze_image_context.return_value = mock_context_result

        # Execute
        result = analyzer_without_retries._with_retry(
            operation=lambda: mock_ollama_client.analyze_image_context(
                image_input="test.jpg",
                model_name="llava-next:7b",
                prompt_template="Test prompt",
            ),
            operation_name="Test analysis",
        )

        # Verify
        assert result == mock_context_result
        mock_ollama_client.analyze_image_context.assert_called_once()

    def test_with_retry_disabled_failure(self, analyzer_without_retries, mock_ollama_client, caplog):
        """Test _with_retry() with retries disabled when operation fails."""
        # Setup - fail with exception
        mock_ollama_client.analyze_image_context.side_effect = ConnectionError("Connection refused")

        # Execute with log capture
        with caplog.at_level(logging.ERROR):
            result = analyzer_without_retries._with_retry(
                operation=lambda: mock_ollama_client.analyze_image_context(
                    image_input="test.jpg",
                    model_name="llava-next:7b",
                    prompt_template="Test prompt",
                ),
                operation_name="Test analysis",
            )

        # Verify
        assert result is None
        # Should only try once when retries are disabled
        mock_ollama_client.analyze_image_context.assert_called_once()
        # Check that error was logged (with is_final=True)
        assert "Test analysis failed" in caplog.text
        assert "Ollama connection failed" in caplog.text

    def test_with_retry_disabled_returns_none(self, analyzer_without_retries, mock_ollama_client, caplog):
        """Test _with_retry() with retries disabled when operation returns None."""
        # Setup - operation returns None
        mock_ollama_client.analyze_image_context.return_value = None

        # Execute
        result = analyzer_without_retries._with_retry(
            operation=lambda: mock_ollama_client.analyze_image_context(
                image_input="test.jpg",
                model_name="llava-next:7b",
                prompt_template="Test prompt",
            ),
            operation_name="Test analysis",
        )

        # Verify
        assert result is None
        # Should only try once when retries are disabled
        mock_ollama_client.analyze_image_context.assert_called_once()
        # Note: When operation returns None (not exception), no error is logged
        # in the disable retries case - it just returns None

    def test_with_retry_exponential_backoff(self, analyzer_with_retries, mock_ollama_client):
        """Test _with_retry() uses exponential backoff with jitter."""
        # Setup - fail first time, succeed second time
        mock_ollama_client.analyze_image_context.side_effect = [
            ConnectionError("First failure"),
            "success_result",
        ]

        # Execute with patched sleep and random
        with patch("time.sleep") as mock_sleep, patch("random.uniform") as mock_uniform:
            # Mock random.uniform to return 0 (no jitter) for predictable test
            mock_uniform.return_value = 0.0

            result = analyzer_with_retries._with_retry(
                operation=lambda: (
                    "success_result"
                    if mock_ollama_client.analyze_image_context.call_count > 1
                    else mock_ollama_client.analyze_image_context()
                ),
                operation_name="Test analysis",
            )

        # Verify
        assert result == "success_result"
        # Check sleep was called with exponential backoff: base_delay = 2.0 * (2**attempt)
        # For attempt 0 (first failure), base_delay = 2.0 * (2**0) = 2.0
        mock_sleep.assert_called_once_with(2.0)  # max(0.5, 2.0 + 0.0) = 2.0

    def test_with_retry_minimum_delay(self, analyzer_with_retries, mock_ollama_client):
        """Test _with_retry() ensures minimum delay of 0.5 seconds."""
        # Setup - fail with negative jitter that would make delay < 0.5
        mock_ollama_client.analyze_image_context.side_effect = ConnectionError("Fail")

        # Execute with patched sleep and random
        with patch("time.sleep") as mock_sleep, patch("random.uniform") as mock_uniform:
            # Mock random.uniform to return -1.0 (large negative jitter)
            # base_delay for attempt 0 = 2.0, with jitter -1.0 = 1.0, but min is 0.5
            mock_uniform.return_value = -1.0

            result = analyzer_with_retries._with_retry(
                operation=lambda: mock_ollama_client.analyze_image_context(),
                operation_name="Test analysis",
            )

        # Verify
        assert result is None
        # Should have slept twice (for attempts 0 and 1, before attempts 1 and 2)
        assert mock_sleep.call_count == 2

        # Check the delay values
        # For attempt 0: base_delay = 2.0 * (2**0) = 2.0, with jitter -1.0 = 1.0
        # For attempt 1: base_delay = 2.0 * (2**1) = 4.0, with jitter -1.0 = 3.0
        # Both are > 0.5, so no minimum clamping needed
        # The actual values should be 1.0 and 3.0
        call_args = [call[0][0] for call in mock_sleep.call_args_list]
        # Check approximate values (floating point)
        assert abs(call_args[0] - 1.0) < 0.01  # 2.0 + (-1.0) = 1.0
        assert abs(call_args[1] - 3.0) < 0.01  # 4.0 + (-1.0) = 3.0

    def test_is_model_not_found_error_detection(self, analyzer_with_retries):
        """Test _is_model_not_found_error() method detection patterns."""
        test_cases = [
            ("model not found", True),
            ("model does not exist", True),
            ("no such model", True),
            ("unable to find model", True),
            ('model "llava-next:7b" not found', True),
            ("model llava-next:7b not found", True),
            ("pull model llava-next:7b first", True),
            ("model llava-next:7b is not available", True),
            ("some other error", False),
            ("", False),
        ]

        for error_msg, expected in test_cases:
            result = analyzer_with_retries._is_model_not_found_error(error_msg)
            assert result == expected, f"Failed for: '{error_msg}'"

    def test_extract_model_name_from_error(self, analyzer_with_retries):
        """Test _extract_model_name_from_error() method."""
        test_cases = [
            ('model "llava-next:7b" not found', "llava-next:7b"),
            ("pull model llava-next:7b first", "llava-next:7b"),
            ("model does not exist: llava-next:7b", "llava-next:7b"),
            ("model: llava-next:7b", "llava-next:7b"),
            ("model llava-next:7b is not available", "llava-next:7b"),
            ("model llava-next:7b not found", "llava-next:7b"),
            ("some other error", "unknown"),
            ("", "unknown"),
        ]

        for error_msg, expected in test_cases:
            result = analyzer_with_retries._extract_model_name_from_error(error_msg)
            assert result == expected, f"Failed for: '{error_msg}'"

    def test_handle_operation_exception_model_not_found(self, analyzer_with_retries, caplog):
        """Test _handle_operation_exception() with model not found error."""
        error_msg = 'model "llava-next:7b" not found'
        exception = Exception(error_msg)

        with caplog.at_level(logging.WARNING):
            analyzer_with_retries._handle_operation_exception(
                exception=exception,
                operation_name="Test analysis",
                attempt=0,
                is_final=False,
            )

        assert "Model 'llava-next:7b' not found" in caplog.text
        assert "verify model is pulled with 'ollama pull llava-next:7b'" in caplog.text

    def test_handle_operation_exception_network_error(self, analyzer_with_retries, caplog):
        """Test _handle_operation_exception() with network error."""
        exception = ConnectionError("Connection refused")

        with caplog.at_level(logging.WARNING):
            analyzer_with_retries._handle_operation_exception(
                exception=exception,
                operation_name="Test analysis",
                attempt=0,
                is_final=False,
            )

        assert "Ollama connection failed" in caplog.text
        assert "check if Ollama server is running" in caplog.text

    def test_analyze_default_with_model_fallback(
        self, analyzer_with_retries, mock_ollama_client, mock_context_result, caplog
    ):
        """Test _analyze_default() uses fallback model when primary fails."""
        # Setup - primary model fails with model not found, fallback succeeds
        error_msg = 'model "llava-next:7b" not found'

        def analyze_side_effect(*args, **kwargs):
            # Check which model is being called
            model_name = kwargs.get("model_name", "llava-next:7b")
            if model_name == "llava-next:7b":
                raise Exception(error_msg)
            elif model_name == "moondream2":
                return mock_context_result
            return None

        mock_ollama_client.analyze_image_context.side_effect = analyze_side_effect

        # Mock the model priority getters
        analyzer_with_retries._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer_with_retries._get_fallback_model_name = Mock(return_value="moondream2")

        # Execute
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer_with_retries._analyze_default("test.jpg")

        # Verify
        assert result == mock_context_result
        # Should have tried primary, then fallback
        assert mock_ollama_client.analyze_image_context.call_count >= 2
        # Should log about trying fallback model (DEBUG level)
        assert "Trying fallback model due to low confidence" in caplog.text

    def test_analyze_default_fallback_on_connection_error(
        self, analyzer_with_retries, mock_ollama_client, mock_context_result, caplog
    ):
        """Test _analyze_default() uses fallback model when primary raises ConnectionError."""

        # Setup - primary model raises ConnectionError, fallback succeeds
        def analyze_side_effect(*args, **kwargs):
            model_name = kwargs.get("model_name", "llava-next:7b")
            if model_name == "llava-next:7b":
                raise ConnectionError("Connection refused")
            elif model_name == "moondream2":
                return mock_context_result
            return None

        mock_ollama_client.analyze_image_context.side_effect = analyze_side_effect

        # Mock the model priority getters
        analyzer_with_retries._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer_with_retries._get_fallback_model_name = Mock(return_value="moondream2")

        # Execute
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer_with_retries._analyze_default("test.jpg")

        # Verify
        assert result == mock_context_result
        # Should have tried primary (with retries), then fallback
        assert mock_ollama_client.analyze_image_context.call_count >= 4  # 3 primary attempts + 1 fallback
        # Should log about trying fallback model
        assert "Trying fallback model due to low confidence" in caplog.text
        # Verify fallback was called with correct model
        calls = mock_ollama_client.analyze_image_context.call_args_list
        # First 3 calls should be to primary model (initial + 2 retries)
        for i in range(3):
            assert calls[i].kwargs.get("model_name") == "llava-next:7b"
        # Last call should be to fallback model
        assert calls[-1].kwargs.get("model_name") == "moondream2"

    def test_analyze_default_fallback_on_timeout_error(
        self, analyzer_with_retries, mock_ollama_client, mock_context_result, caplog
    ):
        """Test _analyze_default() uses fallback model when primary raises TimeoutError."""

        # Setup - primary model raises TimeoutError, fallback succeeds
        def analyze_side_effect(*args, **kwargs):
            model_name = kwargs.get("model_name", "llava-next:7b")
            if model_name == "llava-next:7b":
                raise TimeoutError("Request timed out")
            elif model_name == "moondream2":
                return mock_context_result
            return None

        mock_ollama_client.analyze_image_context.side_effect = analyze_side_effect

        # Mock the model priority getters
        analyzer_with_retries._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer_with_retries._get_fallback_model_name = Mock(return_value="moondream2")

        # Execute
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer_with_retries._analyze_default("test.jpg")

        # Verify
        assert result == mock_context_result
        # Should have tried primary (with retries), then fallback
        assert mock_ollama_client.analyze_image_context.call_count >= 4  # 3 primary attempts + 1 fallback
        # Should log about trying fallback model
        assert "Trying fallback model due to low confidence" in caplog.text
        # Verify fallback was called with correct model
        calls = mock_ollama_client.analyze_image_context.call_args_list
        # First 3 calls should be to primary model (initial + 2 retries)
        for i in range(3):
            assert calls[i].kwargs.get("model_name") == "llava-next:7b"
        # Last call should be to fallback model
        assert calls[-1].kwargs.get("model_name") == "moondream2"

    def test_analyze_default_fallback_on_low_confidence(
        self, analyzer_with_retries, mock_ollama_client, mock_context_result, caplog
    ):
        """Test _analyze_default() uses fallback model when primary returns low confidence result."""
        # Setup - primary model returns result with low confidence, fallback succeeds
        low_confidence_result = ContextAnalysisResult(
            decade="1985-1990",
            decade_confidence=0.2,  # Below min_decade_confidence (0.3)
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

        def analyze_side_effect(*args, **kwargs):
            model_name = kwargs.get("model_name", "llava-next:7b")
            if model_name == "llava-next:7b":
                return low_confidence_result
            elif model_name == "moondream2":
                return mock_context_result  # Higher confidence result
            return None

        mock_ollama_client.analyze_image_context.side_effect = analyze_side_effect

        # Mock the model priority getters
        analyzer_with_retries._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer_with_retries._get_fallback_model_name = Mock(return_value="moondream2")

        # Execute
        with patch("time.sleep"):
            with caplog.at_level(logging.DEBUG):
                result = analyzer_with_retries._analyze_default("test.jpg")

        # Verify
        assert result == mock_context_result  # Should return fallback result, not low confidence result
        # Should have tried primary, then fallback
        assert mock_ollama_client.analyze_image_context.call_count >= 2
        # Should log about trying fallback model due to low confidence
        assert "Trying fallback model due to low confidence" in caplog.text
        # Verify fallback was called with correct model
        calls = mock_ollama_client.analyze_image_context.call_args_list
        assert calls[0].kwargs.get("model_name") == "llava-next:7b"
        assert calls[1].kwargs.get("model_name") == "moondream2"
