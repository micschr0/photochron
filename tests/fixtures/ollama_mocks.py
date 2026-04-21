"""
Test utilities for mocking Ollama responses.

This module provides fixtures and utilities for testing Ollama client
integration without requiring a real Ollama server.
"""

import json
from typing import Any
from unittest.mock import Mock, patch

import pytest

from photochron.models.ollama_client import ContextAnalysisResult


class MockOllamaResponse:
    """Mock response from Ollama API."""

    def __init__(self, response_text: str, model: str = "llava-next:7b"):
        self.response = response_text
        self.model = model
        self.created_at = "2024-01-01T00:00:00.000Z"
        self.done = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format expected by ollama library."""
        return {
            "model": self.model,
            "created_at": self.created_at,
            "response": self.response,
            "done": self.done,
        }


def create_mock_context_response(
    decade: str | None = "1985-1990",
    decade_confidence: float = 0.75,
    season: str | None = "summer",
    event_hint: str | None = None,
    photo_medium: str = "print_scan",
    photo_medium_confidence: float | None = 0.8,
    visual_evidence: list[str] | None = None,
    season_confidence: float | None = None,
    event_confidence: float | None = None,
    alternative_decades: list[str] | None = None,
    uncertainty_flag: bool | None = None,
    hypothesis_notes: str | None = None,
) -> str:
    """
    Create a mock LLM response with structured JSON.

    Args:
        decade: Estimated decade range
        decade_confidence: Confidence in decade estimate
        season: Season depicted
        event_hint: Event hint if any
        photo_medium: Photo medium type
        photo_medium_confidence: Confidence in photo medium estimate
        visual_evidence: List of visual cues
        season_confidence: Confidence in season estimate
        event_confidence: Confidence in event hint
        alternative_decades: Alternative decade possibilities
        uncertainty_flag: Flag indicating high uncertainty
        hypothesis_notes: Explanation for multiple hypotheses

    Returns:
        JSON string that would be returned by LLM
    """
    result = {
        "decade": decade,
        "decade_confidence": decade_confidence,
        "season": season,
        "event_hint": event_hint,
        "photo_medium": photo_medium,
        "photo_medium_confidence": photo_medium_confidence,
        "visual_evidence": visual_evidence,
        "season_confidence": season_confidence,
        "event_confidence": event_confidence,
        "alternative_decades": alternative_decades,
        "uncertainty_flag": uncertainty_flag,
        "hypothesis_notes": hypothesis_notes,
    }
    # Remove None values to keep JSON clean
    result = {k: v for k, v in result.items() if v is not None}
    return json.dumps(result)


def create_mock_llama_response(
    decade: str | None = "1985-1990",
    decade_confidence: float = 0.75,
    season: str | None = "summer",
    event_hint: str | None = None,
    photo_medium: str = "print_scan",
    photo_medium_confidence: float | None = 0.8,
    visual_evidence: list[str] | None = None,
    season_confidence: float | None = None,
    event_confidence: float | None = None,
    alternative_decades: list[str] | None = None,
    uncertainty_flag: bool | None = None,
    hypothesis_notes: str | None = None,
) -> MockOllamaResponse:
    """
    Create a complete mock Ollama response.

    Args:
        decade: Estimated decade range
        decade_confidence: Confidence in decade estimate
        season: Season depicted
        event_hint: Event hint if any
        photo_medium: Photo medium type
        photo_medium_confidence: Confidence in photo medium estimate
        visual_evidence: List of visual cues
        season_confidence: Confidence in season estimate
        event_confidence: Confidence in event hint
        alternative_decades: Alternative decade possibilities
        uncertainty_flag: Flag indicating high uncertainty
        hypothesis_notes: Explanation for multiple hypotheses

    Returns:
        MockOllamaResponse object
    """
    json_response = create_mock_context_response(
        decade=decade,
        decade_confidence=decade_confidence,
        season=season,
        event_hint=event_hint,
        photo_medium=photo_medium,
        photo_medium_confidence=photo_medium_confidence,
        visual_evidence=visual_evidence,
        season_confidence=season_confidence,
        event_confidence=event_confidence,
        alternative_decades=alternative_decades,
        uncertainty_flag=uncertainty_flag,
        hypothesis_notes=hypothesis_notes,
    )
    return MockOllamaResponse(json_response)


def create_mock_ollama_client(available_models: list | None = None, mock_responses: list | None = None) -> Mock:
    """
    Create a mock Ollama client for testing.

    Args:
        available_models: List of model names to return from list()
        mock_responses: List of MockOllamaResponse objects for generate()

    Returns:
        Mock object that mimics ollama module
    """
    if available_models is None:
        available_models = ["llava-next:7b", "moondream2"]

    if mock_responses is None:
        mock_responses = [create_mock_llama_response()]

    mock_ollama = Mock()

    # Mock the list function
    mock_list_response = {"models": [{"name": model} for model in available_models]}
    mock_ollama.list = Mock(return_value=mock_list_response)

    # Mock the generate function to return responses in sequence
    response_iter = iter(mock_responses)
    mock_ollama.generate = Mock(side_effect=lambda *args, **kwargs: next(response_iter).to_dict())

    return mock_ollama


@pytest.fixture
def mock_ollama():
    """Fixture providing a mocked ollama module."""
    with patch("photochron.models.ollama_client.ollama") as mock:
        # Default mock setup
        mock.list.return_value = {"models": [{"name": "llava-next:7b"}, {"name": "moondream2"}]}

        # Default successful response
        mock_response = create_mock_context_response()
        mock.generate.return_value = {
            "model": "llava-next:7b",
            "created_at": "2024-01-01T00:00:00.000Z",
            "response": mock_response,
            "done": True,
        }

        yield mock


@pytest.fixture
def mock_context_result():
    """Fixture providing a mock ContextAnalysisResult."""
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
def mock_failed_ollama():
    """Fixture providing a mocked ollama module that fails."""
    with patch("photochron.models.ollama_client.ollama") as mock:
        mock.list.return_value = {"models": []}  # No models available
        mock.generate.side_effect = Exception("Ollama server not available")
        yield mock


@pytest.fixture
def mock_invalid_json_ollama():
    """Fixture providing a mocked ollama module that returns invalid JSON."""
    with patch("photochron.models.ollama_client.ollama") as mock:
        mock.list.return_value = {"models": [{"name": "llava-next:7b"}, {"name": "moondream2"}]}

        # Return invalid JSON
        mock.generate.return_value = {
            "model": "llava-next:7b",
            "created_at": "2024-01-01T00:00:00.000Z",
            "response": "This is not valid JSON",
            "done": True,
        }

        yield mock


@pytest.fixture
def mock_partial_json_ollama():
    """Fixture providing a mocked ollama module that returns partial JSON."""
    with patch("photochron.models.ollama_client.ollama") as mock:
        mock.list.return_value = {"models": [{"name": "llava-next:7b"}, {"name": "moondream2"}]}

        # Return JSON missing required fields
        partial_json = json.dumps(
            {
                "decade": "1985-1990",
                # Missing decade_confidence
                "season": "summer",
                # Missing photo_medium
            }
        )

        mock.generate.return_value = {
            "model": "llava-next:7b",
            "created_at": "2024-01-01T00:00:00.000Z",
            "response": partial_json,
            "done": True,
        }

        yield mock


# Sample test responses for different scenarios
SAMPLE_RESPONSES = {
    "modern_digital": create_mock_context_response(
        decade="2015-2020",
        decade_confidence=0.85,
        season=None,
        event_hint=None,
        photo_medium="digital",
        photo_medium_confidence=0.9,
    ),
    "vintage_print": create_mock_context_response(
        decade="1975-1980",
        decade_confidence=0.65,
        season="winter",
        event_hint="family_gathering",
        photo_medium="print_scan",
        photo_medium_confidence=0.7,
    ),
    "wedding_polaroid": create_mock_context_response(
        decade="1990-1995",
        decade_confidence=0.8,
        season="spring",
        event_hint="wedding",
        photo_medium="polaroid",
        photo_medium_confidence=0.85,
    ),
    "low_confidence": create_mock_context_response(
        decade="1950-1960",
        decade_confidence=0.25,  # Low confidence
        season=None,
        event_hint=None,
        photo_medium="unknown",
        photo_medium_confidence=0.2,
    ),
}
