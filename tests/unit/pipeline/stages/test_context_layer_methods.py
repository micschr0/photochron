"""
Unit tests for ContextLayerStage storage methods.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from photochron.pipeline.stages.context_layer import ContextLayerStage
from photochron.config import Config, ConfigContext
from photochron.models.ollama_client import ContextAnalysisResult, ModelType
from photochron.models import ContextCreate


class TestContextLayerStageStorageMethods:
    """Test suite for ContextLayerStage storage methods."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock Config with ConfigContext."""
        mock_config = Mock(spec=Config)
        mock_config.context = Mock(spec=ConfigContext)
        mock_config.context.ollama_host = "http://localhost:11434"
        mock_config.context.ollama_timeout = 300
        mock_config.context.max_retries = 3
        mock_config.context.retry_delay = 2.0
        mock_config.context.primary_model = "llava-next:7b"
        mock_config.context.fallback_model = "moondream2"
        mock_config.context.batch_size = 1
        mock_config.context.min_decade_confidence = 0.3
        mock_config.context.min_season_confidence = 0.4
        mock_config.context.use_fallback_on_failure = True
        mock_config.context.store_minimal_on_complete_failure = True
        mock_config.context.memory_warning_threshold_mb = 100
        mock_config.context.memory_critical_threshold_mb = 50
        mock_config.context.memory_retry_delay_seconds = 30
        return mock_config

    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock ContextAnalyzer."""
        mock_analyzer = Mock()
        mock_analyzer.config = Mock()
        mock_analyzer.config.model_priority = []
        return mock_analyzer

    def test_store_context_result_success(self, mock_config, mock_analyzer):
        """Test _store_context_result stores analysis result correctly."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.get_store"
            ) as mock_get_store,
        ):
            # Mock health check
            mock_analyzer.health_check.return_value = {
                "status": "healthy",
                "ollama_health": {
                    "server_available": True,
                    "model_details": {
                        "primary": {"available": True},
                        "fallback": {"available": True},
                    },
                },
            }

            # Create stage
            stage = ContextLayerStage()

            # Create mock analysis result
            mock_result = Mock(spec=ContextAnalysisResult)
            mock_result.decade = "1990-1995"
            mock_result.decade_confidence = 0.85
            mock_result.season = "summer"
            mock_result.season_confidence = 0.75
            mock_result.event_hint = "beach vacation"
            mock_result.event_confidence = 0.65
            mock_result.photo_medium = "print_scan"
            mock_result.photo_medium_confidence = 0.95
            mock_result.visual_evidence = ["palm trees", "ocean", "swimwear"]
            mock_result.alternative_decades = ["1985-1990", "1995-2000"]
            mock_result.uncertainty_flag = False
            mock_result.hypothesis_notes = "Clear summer beach scene"
            mock_result.model_dump_json.return_value = (
                '{"decade": "1990-1995", "confidence": 0.85}'
            )

            # Create mock store and helper
            mock_store = Mock()
            mock_conn = Mock()
            mock_helper = Mock()
            mock_transaction_context = MagicMock()

            # Set up mock chain
            mock_get_store.return_value = mock_store
            mock_store.transaction.return_value = mock_transaction_context
            mock_transaction_context.__enter__.return_value = mock_conn
            mock_transaction_context.__exit__.return_value = None
            mock_store.get_query_helper.return_value = mock_helper

            # Call _store_context_result
            photo_id = 123
            stage._store_context_result(photo_id, mock_result)

            # Verify the mock chain was called correctly
            mock_get_store.assert_called_once()
            mock_store.transaction.assert_called_once()
            mock_store.get_query_helper.assert_called_once_with(mock_conn)

            # Verify insert_context was called with correct ContextCreate
            mock_helper.insert_context.assert_called_once()
            call_args = mock_helper.insert_context.call_args[0]
            assert len(call_args) == 1
            context_data = call_args[0]

            # Verify ContextCreate fields
            assert isinstance(context_data, ContextCreate)
            assert context_data.photo_id == photo_id
            assert context_data.decade == "1990-1995"
            assert context_data.decade_confidence == 0.85
            assert context_data.season == "summer"
            assert context_data.season_confidence == 0.75
            assert context_data.event_hint == "beach vacation"
            assert context_data.event_confidence == 0.65
            assert context_data.photo_medium == "print_scan"
            assert context_data.photo_medium_confidence == 0.95
            assert context_data.visual_evidence == ["palm trees", "ocean", "swimwear"]
            assert context_data.alternative_decades == ["1985-1990", "1995-2000"]
            assert context_data.uncertainty_flag is False
            assert context_data.hypothesis_notes == "Clear summer beach scene"
            assert (
                context_data.raw_json == '{"decade": "1990-1995", "confidence": 0.85}'
            )

    def test_store_context_result_without_model_dump_json(
        self, mock_config, mock_analyzer
    ):
        """Test _store_context_result when result doesn't have model_dump_json method."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.get_store"
            ) as mock_get_store,
        ):
            # Mock health check
            mock_analyzer.health_check.return_value = {
                "status": "healthy",
                "ollama_health": {
                    "server_available": True,
                    "model_details": {
                        "primary": {"available": True},
                        "fallback": {"available": True},
                    },
                },
            }

            # Create stage
            stage = ContextLayerStage()

            # Create a custom class that doesn't have model_dump_json
            class ResultWithoutDumpJson:
                def __init__(self):
                    self.decade = "2000-2005"
                    self.decade_confidence = 0.9
                    self.season = "winter"
                    self.season_confidence = 0.8
                    self.event_hint = "christmas"
                    self.event_confidence = 0.7
                    self.photo_medium = "digital"
                    self.photo_medium_confidence = 0.98
                    self.visual_evidence = ["snow", "christmas tree"]
                    self.alternative_decades = ["1995-2000", "2005-2010"]
                    self.uncertainty_flag = True
                    self.hypothesis_notes = "Possible Christmas scene"
                    # No model_dump_json method

            mock_result = ResultWithoutDumpJson()

            # Create mock store and helper
            mock_store = Mock()
            mock_conn = Mock()
            mock_helper = Mock()
            mock_transaction_context = MagicMock()

            # Set up mock chain
            mock_get_store.return_value = mock_store
            mock_store.transaction.return_value = mock_transaction_context
            mock_transaction_context.__enter__.return_value = mock_conn
            mock_transaction_context.__exit__.return_value = None
            mock_store.get_query_helper.return_value = mock_helper

            # Call _store_context_result
            photo_id = 456
            stage._store_context_result(photo_id, mock_result)

            # Verify insert_context was called
            mock_helper.insert_context.assert_called_once()
            call_args = mock_helper.insert_context.call_args[0]
            context_data = call_args[0]

            # Verify raw_json contains error message
            assert "error" in context_data.raw_json
            raw_json_dict = json.loads(context_data.raw_json)
            assert raw_json_dict["error"] == "No model_dump_json method available"

    def test_store_context_result_with_none_values(self, mock_config, mock_analyzer):
        """Test _store_context_result with None values in result."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.get_store"
            ) as mock_get_store,
        ):
            # Mock health check
            mock_analyzer.health_check.return_value = {
                "status": "healthy",
                "ollama_health": {
                    "server_available": True,
                    "model_details": {
                        "primary": {"available": True},
                        "fallback": {"available": True},
                    },
                },
            }

            # Create stage
            stage = ContextLayerStage()

            # Create mock analysis result with None values
            mock_result = Mock(spec=ContextAnalysisResult)
            mock_result.decade = None
            mock_result.decade_confidence = 0.0
            mock_result.season = None
            mock_result.season_confidence = None
            mock_result.event_hint = None
            mock_result.event_confidence = None
            mock_result.photo_medium = "unknown"
            mock_result.photo_medium_confidence = 0.0
            mock_result.visual_evidence = None
            mock_result.alternative_decades = None
            mock_result.uncertainty_flag = True
            mock_result.hypothesis_notes = "Low confidence analysis"
            mock_result.model_dump_json.return_value = '{"status": "low_confidence"}'

            # Create mock store and helper
            mock_store = Mock()
            mock_conn = Mock()
            mock_helper = Mock()
            mock_transaction_context = MagicMock()

            # Set up mock chain
            mock_get_store.return_value = mock_store
            mock_store.transaction.return_value = mock_transaction_context
            mock_transaction_context.__enter__.return_value = mock_conn
            mock_transaction_context.__exit__.return_value = None
            mock_store.get_query_helper.return_value = mock_helper

            # Call _store_context_result
            photo_id = 789
            stage._store_context_result(photo_id, mock_result)

            # Verify insert_context was called with None values
            mock_helper.insert_context.assert_called_once()
            call_args = mock_helper.insert_context.call_args[0]
            context_data = call_args[0]

            # Verify ContextCreate fields with None values
            assert context_data.photo_id == photo_id
            assert context_data.decade is None
            assert context_data.decade_confidence == 0.0
            assert context_data.season is None
            assert context_data.season_confidence is None
            assert context_data.event_hint is None
            assert context_data.event_confidence is None
            assert context_data.photo_medium == "unknown"
            assert context_data.photo_medium_confidence == 0.0
            assert context_data.visual_evidence is None
            assert context_data.alternative_decades is None
            assert context_data.uncertainty_flag is True
            assert context_data.hypothesis_notes == "Low confidence analysis"
            assert context_data.raw_json == '{"status": "low_confidence"}'

    def test_store_minimal_context_success(self, mock_config, mock_analyzer):
        """Test _store_minimal_context stores minimal data correctly."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.get_store"
            ) as mock_get_store,
        ):
            # Mock health check
            mock_analyzer.health_check.return_value = {
                "status": "healthy",
                "ollama_health": {
                    "server_available": True,
                    "model_details": {
                        "primary": {"available": True},
                        "fallback": {"available": True},
                    },
                },
            }

            # Create stage
            stage = ContextLayerStage()

            # Create mock store and helper
            mock_store = Mock()
            mock_conn = Mock()
            mock_helper = Mock()
            mock_transaction_context = MagicMock()

            # Set up mock chain
            mock_get_store.return_value = mock_store
            mock_store.transaction.return_value = mock_transaction_context
            mock_transaction_context.__enter__.return_value = mock_conn
            mock_transaction_context.__exit__.return_value = None
            mock_store.get_query_helper.return_value = mock_helper

            # Call _store_minimal_context
            photo_id = 999
            stage._store_minimal_context(photo_id)

            # Verify the mock chain was called correctly
            mock_get_store.assert_called_once()
            mock_store.transaction.assert_called_once()
            mock_store.get_query_helper.assert_called_once_with(mock_conn)

            # Verify insert_context was called with correct ContextCreate
            mock_helper.insert_context.assert_called_once()
            call_args = mock_helper.insert_context.call_args[0]
            assert len(call_args) == 1
            context_data = call_args[0]

            # Verify ContextCreate fields for minimal context
            assert isinstance(context_data, ContextCreate)
            assert context_data.photo_id == photo_id
            assert context_data.decade is None
            assert context_data.decade_confidence == 0.0
            assert context_data.season is None
            assert context_data.season_confidence is None
            assert context_data.event_hint is None
            assert context_data.event_confidence is None
            assert context_data.photo_medium == "unknown"
            assert context_data.photo_medium_confidence == 0.0
            assert context_data.visual_evidence is None
            assert context_data.alternative_decades is None
            assert context_data.uncertainty_flag is True
            assert context_data.hypothesis_notes == "Analysis failed completely"

            # Verify raw_json contains minimal JSON
            raw_json_dict = json.loads(context_data.raw_json)
            assert raw_json_dict["status"] == "failed"
            assert raw_json_dict["error"] == "Analysis failed completely"
            assert raw_json_dict["photo_id"] == photo_id
            assert raw_json_dict["timestamp"] == "1970-01-01T00:00:00Z"

    def test_store_minimal_context_multiple_calls(self, mock_config, mock_analyzer):
        """Test _store_minimal_context can be called multiple times."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.get_store"
            ) as mock_get_store,
        ):
            # Mock health check
            mock_analyzer.health_check.return_value = {
                "status": "healthy",
                "ollama_health": {
                    "server_available": True,
                    "model_details": {
                        "primary": {"available": True},
                        "fallback": {"available": True},
                    },
                },
            }

            # Create stage
            stage = ContextLayerStage()

            # Create mock store and helper
            mock_store = Mock()
            mock_conn = Mock()
            mock_helper = Mock()
            mock_transaction_context = MagicMock()

            # Set up mock chain
            mock_get_store.return_value = mock_store
            mock_store.transaction.return_value = mock_transaction_context
            mock_transaction_context.__enter__.return_value = mock_conn
            mock_transaction_context.__exit__.return_value = None
            mock_store.get_query_helper.return_value = mock_helper

            # Call _store_minimal_context multiple times
            photo_ids = [101, 102, 103]
            for photo_id in photo_ids:
                stage._store_minimal_context(photo_id)

            # Verify insert_context was called 3 times
            assert mock_helper.insert_context.call_count == 3

            # Verify each call had correct photo_id
            calls = mock_helper.insert_context.call_args_list
            for i, call in enumerate(calls):
                context_data = call[0][0]
                assert context_data.photo_id == photo_ids[i]

    def test_store_context_result_exception_handling(self, mock_config, mock_analyzer):
        """Test _store_context_result handles exceptions from database."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.get_store"
            ) as mock_get_store,
        ):
            # Mock health check
            mock_analyzer.health_check.return_value = {
                "status": "healthy",
                "ollama_health": {
                    "server_available": True,
                    "model_details": {
                        "primary": {"available": True},
                        "fallback": {"available": True},
                    },
                },
            }

            # Create stage
            stage = ContextLayerStage()

            # Create mock analysis result
            mock_result = Mock(spec=ContextAnalysisResult)
            mock_result.decade = "1990-1995"
            mock_result.decade_confidence = 0.85
            mock_result.season = "summer"
            mock_result.season_confidence = 0.75
            mock_result.event_hint = "beach vacation"
            mock_result.event_confidence = 0.65
            mock_result.photo_medium = "print_scan"
            mock_result.photo_medium_confidence = 0.95
            mock_result.visual_evidence = ["palm trees", "ocean", "swimwear"]
            mock_result.alternative_decades = ["1985-1990", "1995-2000"]
            mock_result.uncertainty_flag = False
            mock_result.hypothesis_notes = "Clear summer beach scene"
            mock_result.model_dump_json.return_value = '{"decade": "1990-1995"}'

            # Create mock store that raises exception
            mock_store = Mock()
            mock_get_store.return_value = mock_store
            mock_store.transaction.side_effect = Exception("Database connection failed")

            # Call _store_context_result - should raise exception
            photo_id = 123
            with pytest.raises(Exception, match="Database connection failed"):
                stage._store_context_result(photo_id, mock_result)

    def test_store_minimal_context_exception_handling(self, mock_config, mock_analyzer):
        """Test _store_minimal_context handles exceptions from database."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.get_store"
            ) as mock_get_store,
        ):
            # Mock health check
            mock_analyzer.health_check.return_value = {
                "status": "healthy",
                "ollama_health": {
                    "server_available": True,
                    "model_details": {
                        "primary": {"available": True},
                        "fallback": {"available": True},
                    },
                },
            }

            # Create stage
            stage = ContextLayerStage()

            # Create mock store that raises exception
            mock_store = Mock()
            mock_get_store.return_value = mock_store
            mock_store.transaction.side_effect = Exception("Database connection failed")

            # Call _store_minimal_context - should raise exception
            photo_id = 999
            with pytest.raises(Exception, match="Database connection failed"):
                stage._store_minimal_context(photo_id)

    def test_store_context_result_edge_cases(self, mock_config, mock_analyzer):
        """Test _store_context_result with edge case values."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.get_store"
            ) as mock_get_store,
        ):
            # Mock health check
            mock_analyzer.health_check.return_value = {
                "status": "healthy",
                "ollama_health": {
                    "server_available": True,
                    "model_details": {
                        "primary": {"available": True},
                        "fallback": {"available": True},
                    },
                },
            }

            # Create stage
            stage = ContextLayerStage()

            # Create mock analysis result with edge case values
            mock_result = Mock(spec=ContextAnalysisResult)
            mock_result.decade = ""  # Empty string
            mock_result.decade_confidence = 1.0  # Max confidence
            mock_result.season = "  "  # Whitespace string
            mock_result.season_confidence = 0.0  # Min confidence
            mock_result.event_hint = "a" * 1000  # Very long string
            mock_result.event_confidence = 0.0001  # Very small confidence
            mock_result.photo_medium = "very_long_photo_medium_type_name"
            mock_result.photo_medium_confidence = 0.999999  # Very high precision
            mock_result.visual_evidence = []  # Empty list
            mock_result.alternative_decades = ["", " ", "1980-1985"]  # Mixed values
            mock_result.uncertainty_flag = False
            mock_result.hypothesis_notes = ""  # Empty string
            mock_result.model_dump_json.return_value = '{"edge": "case"}'

            # Create mock store and helper
            mock_store = Mock()
            mock_conn = Mock()
            mock_helper = Mock()
            mock_transaction_context = MagicMock()

            # Set up mock chain
            mock_get_store.return_value = mock_store
            mock_store.transaction.return_value = mock_transaction_context
            mock_transaction_context.__enter__.return_value = mock_conn
            mock_transaction_context.__exit__.return_value = None
            mock_store.get_query_helper.return_value = mock_helper

            # Call _store_context_result
            photo_id = 777
            stage._store_context_result(photo_id, mock_result)

            # Verify insert_context was called with edge case values
            mock_helper.insert_context.assert_called_once()
            call_args = mock_helper.insert_context.call_args[0]
            context_data = call_args[0]

            # Verify ContextCreate fields preserve edge case values
            assert context_data.photo_id == photo_id
            assert context_data.decade == ""  # Empty string preserved
            assert context_data.decade_confidence == 1.0
            assert context_data.season == "  "  # Whitespace preserved
            assert context_data.season_confidence == 0.0
            assert context_data.event_hint == "a" * 1000  # Long string preserved
            assert context_data.event_confidence == 0.0001
            assert context_data.photo_medium == "very_long_photo_medium_type_name"
            assert context_data.photo_medium_confidence == 0.999999
            assert context_data.visual_evidence == []  # Empty list preserved
            assert context_data.alternative_decades == ["", " ", "1980-1985"]
            assert context_data.uncertainty_flag is False
            assert context_data.hypothesis_notes == ""  # Empty string preserved
            assert context_data.raw_json == '{"edge": "case"}'
