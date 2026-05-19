"""
Comprehensive error handling tests for retry logic and fallback strategies.

This module tests error handling across the system, including:
- ContextAnalyzer retry logic (_with_retry method)
- Fallback strategies in ContextAnalyzer
- OllamaClient error handling
- Various error scenarios: connection errors, timeout, JSON parsing errors, etc.
"""

import json
import logging
from unittest.mock import Mock, patch

import pytest

from photochron.context.analyzer import (
    AnalysisStrategy,
    ContextAnalyzer,
    ContextAnalyzerConfig,
)
from photochron.models.ollama_client import (
    ContextAnalysisResult,
    ModelType,
    OllamaClient,
    OllamaConfig,
)

# Import ollama exceptions if available (availability check only)
try:
    from ollama import RequestError, ResponseError  # noqa: F401

    HAS_OLLAMA_EXCEPTIONS = True
except ImportError:
    HAS_OLLAMA_EXCEPTIONS = False


class TestErrorHandlingComprehensive:
    """Comprehensive test suite for error handling across the system."""

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

    @pytest.fixture
    def ollama_client(self):
        """Create an OllamaClient instance for testing."""
        config = OllamaConfig(
            host="http://localhost:11434",
            timeout=300,
            max_retries=3,
            retry_delay=2.0,
            jitter_percentage=0.2,
            primary_model=ModelType.LLAVA_NEXT_7B,
            fallback_model=ModelType.MOONDREAM2,
        )
        client = OllamaClient(config)
        # Mock the connect method to avoid actual connection
        client.connect = Mock(return_value=True)
        client._available_models = ["llava-next:7b", "moondream2"]
        return client

    # Test 1: ContextAnalyzer retry logic - success cases
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

    # Test 2: ContextAnalyzer retry logic - failure cases
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
        assert "Model 'llava-next:7b' not found or cannot be loaded" in caplog.text
        assert "verify model is pulled with 'ollama pull llava-next:7b'" in caplog.text

    # Test 3: ContextAnalyzer retry logic - various error types
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

    # Test 4: ContextAnalyzer retry logic - disabled retries
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
        assert "Test analysis failed after" in caplog.text
        assert "Ollama connection failed" in caplog.text

    # Test 5: ContextAnalyzer fallback strategies
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
        # Should log about model not found error
        assert "Model 'llava-next:7b' not found or cannot be loaded" in caplog.text
        # Should also log about trying fallback model (DEBUG level)
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
        # Should log about connection error
        assert "Ollama connection failed" in caplog.text
        # Should also log about trying fallback model
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

    # Test 6: OllamaClient error handling - JSON parsing errors
    def test_ollama_client_json_parsing_fallback(self, ollama_client, caplog):
        """Test OllamaClient JSON parsing fallback with various invalid JSON."""
        test_cases = [
            {
                "name": "malformed_json",
                "response": """{
  "decade": "1985-1990"
  "decade_confidence": 0.82,  # Missing comma
  "season": "summer"
}""",
                "expected_result": None,
                "expected_log": "JSON decode error",
            },
            {
                "name": "trailing_commas",
                "response": """{
  "decade": "1985-1990",
  "decade_confidence": 0.82,
  "season": "summer",
  "season_confidence": 0.7,
  "event_hint": null,
  "photo_medium": "print_scan",
  "visual_evidence": ["bell-bottom jeans", "large collar shirt"],
},""",  # Trailing comma after closing brace
                "expected_result": ContextAnalysisResult,
                "expected_log": "JSON decode error",
            },
            {
                "name": "valid_json_but_validation_error",
                "response": json.dumps(
                    {
                        "decade": "1985-1990",
                        "decade_confidence": 1.5,  # > 1.0, triggers validation error
                        "season": "summer",
                        "season_confidence": 0.7,
                        "photo_medium": "print_scan",
                    }
                ),
                "expected_result": ContextAnalysisResult,
                "expected_log": "Validation failed for LLM response",
            },
        ]

        for test_case in test_cases:
            with patch("photochron.models.ollama_client.ollama") as mock_ollama:
                # Mock the generate method to return test response
                mock_ollama.generate.return_value = {"response": test_case["response"]}

                # Mock other methods
                ollama_client._prepare_image_input = Mock(return_value="/fake/image.jpg")
                ollama_client.is_model_available = Mock(return_value=True)

                with caplog.at_level(logging.WARNING):
                    result = ollama_client.analyze_image_context(
                        image_input="/fake/image.jpg",
                        model_name="llava-next:7b",
                        use_base64=False,
                    )

                # Verify assertions based on test case
                if test_case["expected_result"] is None:
                    assert result is None, f"Test case '{test_case['name']}': Expected None but got {result}"
                elif test_case["expected_result"] == ContextAnalysisResult:
                    assert result is not None, (
                        f"Test case '{test_case['name']}': Expected ContextAnalysisResult but got None"
                    )
                    assert isinstance(result, ContextAnalysisResult), (
                        f"Test case '{test_case['name']}': Expected ContextAnalysisResult but got {type(result)}"
                    )

                # Verify logging
                assert test_case["expected_log"] in caplog.text, (
                    f"Test case '{test_case['name']}': Expected log '{test_case['expected_log']}' not found in: {caplog.text}"
                )

                # Clear caplog for next test case
                caplog.clear()

    # Test 7: OllamaClient error handling - timeout and network errors
    def test_ollama_client_timeout_handling(self, ollama_client, caplog):
        """Test OllamaClient timeout handling and circuit breaker logic."""
        with patch("photochron.models.ollama_client.ollama") as mock_ollama:
            # Mock ollama.generate to raise TimeoutError
            mock_ollama.generate.side_effect = TimeoutError("Request timed out")

            # Mock other methods to avoid actual file operations
            ollama_client._prepare_image_input = Mock(return_value="/fake/image.jpg")
            ollama_client.is_model_available = Mock(return_value=True)

            with caplog.at_level(logging.WARNING):
                result = ollama_client.analyze_image_context(
                    image_input="/fake/image.jpg",
                    model_name="llava-next:7b",
                    use_base64=False,
                )

            # Verify 1: TimeoutError was caught and handled (no exception raised)
            # The method should return None after all retries fail
            assert result is None, "Should return None after timeout retries exhausted"

            # Verify 2: Circuit breaker incremented consecutive_timeouts
            # Check logs for timeout warnings
            assert "Network/timeout error analyzing image on attempt 1" in caplog.text
            assert "Network/timeout error analyzing image on attempt 2" in caplog.text

            # Verify 3: Circuit breaker triggered after max_consecutive_timeouts (2)
            # Should see circuit breaker log message
            assert "Circuit breaker triggered: 2 consecutive timeouts" in caplog.text

            # Verify 4: Should not attempt more retries after circuit breaker triggers
            # ollama.generate should be called exactly max_consecutive_timeouts times (2)
            assert mock_ollama.generate.call_count == 2, (
                f"Should call generate exactly 2 times (circuit breaker limit), "
                f"but called {mock_ollama.generate.call_count} times"
            )

            # Additional assertion: Verify the error message is logged
            assert "Request timed out" in caplog.text

    def test_ollama_client_os_error_handling(self, ollama_client, caplog):
        """Test OllamaClient OSError handling and circuit breaker logic."""
        with patch("photochron.models.ollama_client.ollama") as mock_ollama:
            # Mock ollama.generate to raise OSError (socket timeout)
            mock_ollama.generate.side_effect = OSError("Socket timeout")

            # Mock other methods to avoid actual file operations
            ollama_client._prepare_image_input = Mock(return_value="/fake/image.jpg")
            ollama_client.is_model_available = Mock(return_value=True)

            with caplog.at_level(logging.WARNING):
                result = ollama_client.analyze_image_context(
                    image_input="/fake/image.jpg",
                    model_name="llava-next:7b",
                    use_base64=False,
                )

            # Verify 1: OSError was caught and handled (no exception raised)
            # The method should return None after all retries fail
            assert result is None, "Should return None after OSError retries exhausted"

            # Verify 2: Circuit breaker incremented consecutive_timeouts for OSError
            # Check logs for timeout warnings
            assert "Network/timeout error analyzing image on attempt 1" in caplog.text
            assert "Network/timeout error analyzing image on attempt 2" in caplog.text
            assert "Socket timeout" in caplog.text

            # Verify 3: Circuit breaker triggered after max_consecutive_timeouts (2)
            # Should see circuit breaker log message
            assert "Circuit breaker triggered: 2 consecutive timeouts" in caplog.text

            # Verify 4: Should not attempt more retries after circuit breaker triggers
            # ollama.generate should be called exactly max_consecutive_timeouts times (2)
            assert mock_ollama.generate.call_count == 2, (
                f"Should call generate exactly 2 times (circuit breaker limit), "
                f"but called {mock_ollama.generate.call_count} times"
            )

    # Test 8: OllamaClient error handling - recovery scenarios
    def test_ollama_client_timeout_then_success(self, ollama_client, caplog):
        """Test OllamaClient with timeout on first attempt, success on second."""
        with patch("photochron.models.ollama_client.ollama") as mock_ollama:
            # Mock ollama.generate to raise TimeoutError first, then succeed
            mock_ollama.generate.side_effect = [
                TimeoutError("Request timed out"),
                {
                    "response": json.dumps(
                        {
                            "decade": "1985-1990",
                            "decade_confidence": 0.82,
                            "season": "summer",
                            "season_confidence": 0.7,
                            "event_hint": None,
                            "event_confidence": None,
                            "photo_medium": "print_scan",
                            "photo_medium_confidence": 0.8,
                            "visual_evidence": [
                                "bell-bottom jeans",
                                "large collar shirt",
                                "1970s car model",
                            ],
                        }
                    )
                },
            ]

            # Mock other methods to avoid actual file operations
            ollama_client._prepare_image_input = Mock(return_value="/fake/image.jpg")
            ollama_client.is_model_available = Mock(return_value=True)

            with caplog.at_level(logging.INFO):
                result = ollama_client.analyze_image_context(
                    image_input="/fake/image.jpg",
                    model_name="llava-next:7b",
                    use_base64=False,
                )

            # Verify: Should return successful result
            assert result is not None, "Should return result after successful retry"
            assert isinstance(result, ContextAnalysisResult)
            assert result.decade == "1985-1990"
            assert result.decade_confidence == 0.82

            # Verify: Timeout was logged
            assert "Network/timeout error analyzing image on attempt 1" in caplog.text
            assert "Request timed out" in caplog.text

            # Verify: Success was logged (at INFO level)
            assert "Successfully analyzed image /fake/image.jpg" in caplog.text

            # Verify: Circuit breaker was reset (no circuit breaker trigger log)
            assert "Circuit breaker triggered" not in caplog.text

            # Verify: ollama.generate was called twice (timeout + success)
            assert mock_ollama.generate.call_count == 2, (
                f"Should call generate exactly 2 times, but called {mock_ollama.generate.call_count} times"
            )

    # Test 9: Edge cases and error recovery
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

    # Test 10: Error detection helper methods
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

    # Test 11: Integration error scenarios
    def test_analyze_image_context_empty_response(self, ollama_client, caplog):
        """Test analyze_image_context() with empty response."""
        with patch("photochron.models.ollama_client.ollama") as mock_ollama:
            # Mock the generate method to return empty response
            mock_ollama.generate.return_value = {"response": ""}

            # Mock other methods
            ollama_client._prepare_image_input = Mock(return_value="/fake/image.jpg")
            ollama_client.is_model_available = Mock(return_value=True)

            with caplog.at_level(logging.WARNING):
                result = ollama_client.analyze_image_context(
                    image_input="/fake/image.jpg",
                    model_name="llava-next:7b",
                    use_base64=False,
                )

            # Should return None due to empty response
            assert result is None
            assert "Empty LLM response received" in caplog.text

    def test_analyze_image_context_no_json(self, ollama_client, caplog):
        """Test analyze_image_context() with response containing no JSON."""
        with patch("photochron.models.ollama_client.ollama") as mock_ollama:
            # Mock the generate method to return non-JSON response
            mock_ollama.generate.return_value = {"response": "I cannot analyze this image."}

            # Mock other methods
            ollama_client._prepare_image_input = Mock(return_value="/fake/image.jpg")
            ollama_client.is_model_available = Mock(return_value=True)

            with caplog.at_level(logging.WARNING):
                result = ollama_client.analyze_image_context(
                    image_input="/fake/image.jpg",
                    model_name="llava-next:7b",
                    use_base64=False,
                )

            # Should return None due to invalid response
            assert result is None
            assert "No JSON found in response" in caplog.text

    # Test 12: Error handling with different analysis strategies
    def test_analyze_with_different_strategies_on_error(self, mock_ollama_client, caplog, tmp_path):
        """Test error handling with different analysis strategies."""
        strategies = [
            (AnalysisStrategy.DEFAULT, True),  # Should retry
            (AnalysisStrategy.AGGRESSIVE, True),  # Should retry
            (AnalysisStrategy.CONSERVATIVE, True),  # Should retry
            (
                AnalysisStrategy.FAST,
                True,
            ),  # Should also retry (fast mode still retries)
        ]

        # The analyzer short-circuits with "Image file does not exist" when
        # passed a bogus path, so plant a real (empty) file before exercising
        # the retry logic.
        image_path = tmp_path / "test.jpg"
        image_path.write_bytes(b"")

        for strategy, should_retry in strategies:
            config = ContextAnalyzerConfig(
                strategy=strategy,
                enable_retries=True,
                max_retries=2,
                model_priority=[ModelType.LLAVA_NEXT_7B, ModelType.MOONDREAM2],
            )
            analyzer = ContextAnalyzer(ollama_client=mock_ollama_client, config=config)

            # Setup - always fail with connection error
            mock_ollama_client.analyze_image_context.side_effect = ConnectionError("Connection refused")

            # Execute - call the public analyze method with the strategy
            with patch("time.sleep"):
                with caplog.at_level(logging.ERROR):
                    result = analyzer.analyze(str(image_path), strategy=strategy)

            # Verify
            assert result is None
            # Check retry behavior based on strategy
            if should_retry:
                # Should have retried (initial + max_retries)
                # Note: All strategies with enable_retries=True should retry
                assert mock_ollama_client.analyze_image_context.call_count >= 3
                assert "analysis failed after" in caplog.text
            else:
                # If a strategy shouldn't retry, it would only try once
                assert mock_ollama_client.analyze_image_context.call_count == 1
                assert "analysis failed after" in caplog.text

            # Reset mock for next iteration
            mock_ollama_client.analyze_image_context.reset_mock()
            caplog.clear()

    # Test 13: Comprehensive error scenario - chain of failures
    def test_comprehensive_error_scenario_chain_of_failures(self, analyzer_with_retries, mock_ollama_client, caplog):
        """Test comprehensive error scenario with chain of failures."""

        # Setup - primary model fails with connection error, fallback fails with timeout
        def analyze_side_effect(*args, **kwargs):
            model_name = kwargs.get("model_name", "llava-next:7b")
            if model_name == "llava-next:7b":
                raise ConnectionError("Connection refused")
            elif model_name == "moondream2":
                raise TimeoutError("Request timed out")
            return None

        mock_ollama_client.analyze_image_context.side_effect = analyze_side_effect

        # Mock the model priority getters
        analyzer_with_retries._get_primary_model_name = Mock(return_value="llava-next:7b")
        analyzer_with_retries._get_fallback_model_name = Mock(return_value="moondream2")

        # Execute
        with patch("time.sleep"):
            with caplog.at_level(logging.ERROR):
                result = analyzer_with_retries._analyze_default("test.jpg")

        # Verify - should return None after all attempts fail
        assert result is None
        # Should have tried primary (with retries), then fallback (with retries)
        # Primary: 3 attempts (initial + 2 retries)
        # Fallback: 3 attempts (initial + 2 retries)
        assert mock_ollama_client.analyze_image_context.call_count >= 6
        # Should log both types of errors
        assert "Ollama connection failed" in caplog.text
        assert "Request timed out" in caplog.text

    # Test 14: Error handling with invalid image input
    def test_analyze_image_context_invalid_image_input(self, ollama_client, caplog):
        """Test analyze_image_context() with invalid image input."""
        with patch("photochron.models.ollama_client.ollama"):
            # Mock _prepare_image_input to raise exception for invalid image
            ollama_client._prepare_image_input = Mock(side_effect=ValueError("Invalid image format"))
            ollama_client.is_model_available = Mock(return_value=True)

            with caplog.at_level(logging.ERROR):
                result = ollama_client.analyze_image_context(
                    image_input="/invalid/image.jpg",
                    model_name="llava-next:7b",
                    use_base64=False,
                )

            # Should return None due to invalid image
            assert result is None
            assert "Invalid image format" in caplog.text

    # Test 15: Error handling with model not available
    def test_analyze_image_context_model_not_available(self, ollama_client, caplog):
        """Test analyze_image_context() when model is not available."""
        with patch("photochron.models.ollama_client.ollama"):
            # Mock is_model_available to return False
            ollama_client.is_model_available = Mock(return_value=False)
            ollama_client._prepare_image_input = Mock(return_value="/fake/image.jpg")

            with caplog.at_level(logging.WARNING):
                result = ollama_client.analyze_image_context(
                    image_input="/fake/image.jpg",
                    model_name="llava-next:7b",
                    use_base64=False,
                )

            # Should return None due to model not available
            assert result is None
            assert "Fallback model moondream2 also not available" in caplog.text
