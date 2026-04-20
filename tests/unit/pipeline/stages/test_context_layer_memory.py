"""
Additional unit tests for ContextLayerStage memory check implementation.
Focusing on edge cases and integration with config.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime

from photochron.pipeline.stages.context_layer import ContextLayerStage
from photochron.config import Config, ConfigContext
from photochron.models.ollama_client import ModelType
from photochron.context.analyzer import ContextAnalyzer, ContextAnalyzerConfig
from photochron.models import Photo


class TestContextLayerStageMemory:
    """Additional tests for ContextLayerStage memory check implementation."""

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
        mock_analyzer = Mock(spec=ContextAnalyzer)
        mock_analyzer.config = Mock(spec=ContextAnalyzerConfig)
        mock_analyzer.config.model_priority = []
        return mock_analyzer

    def test_run_with_invalid_batch_size_zero(self, mock_config, mock_analyzer):
        """Test run() method handles batch_size = 0 by using 1 instead."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch.object(ContextLayerStage, "mark_complete") as mock_mark_complete,
            patch.object(
                ContextLayerStage, "_get_photos_without_context"
            ) as mock_get_photos,
            patch.object(ContextLayerStage, "_process_photo") as mock_process_photo,
            patch.object(
                ContextLayerStage, "_check_memory_before_batch"
            ) as mock_check_memory,
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

            # Create photos for testing
            photo1 = Photo(
                id=1,
                content_hash="hash1",
                file_path="/path/to/photo1.jpg",
                downsample_path="/path/to/downsample1.jpg",
                exif_datetime="2020:01:01 12:00:00",
                make="Canon",
                model="EOS 5D",
                perceptual_hash="phash1",
                created_at=datetime(2020, 1, 1, 12, 0, 0),
            )
            photo2 = Photo(
                id=2,
                content_hash="hash2",
                file_path="/path/to/photo2.jpg",
                downsample_path="/path/to/downsample2.jpg",
                exif_datetime=None,
                make=None,
                model=None,
                perceptual_hash="phash2",
                created_at=datetime(2020, 1, 2, 12, 0, 0),
            )
            mock_get_photos.return_value = [photo1, photo2]

            # Set batch_size to 0 (invalid)
            mock_config.context.batch_size = 0

            # Mock memory check to return ok
            mock_check_memory.return_value = {
                "status": "ok",
                "available_mb": 200.0,
                "message": "Memory OK",
            }

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Should have processed both photos (with batch_size corrected to 1)
            assert mock_process_photo.call_count == 2
            mock_process_photo.assert_has_calls(
                [
                    call(photo1),
                    call(photo2),
                ]
            )

            # Should mark complete with 2 photos processed
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=2
            )

    def test_run_with_negative_batch_size(self, mock_config, mock_analyzer):
        """Test run() method handles negative batch_size by using 1 instead."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch.object(ContextLayerStage, "mark_complete") as mock_mark_complete,
            patch.object(
                ContextLayerStage, "_get_photos_without_context"
            ) as mock_get_photos,
            patch.object(ContextLayerStage, "_process_photo") as mock_process_photo,
            patch.object(
                ContextLayerStage, "_check_memory_before_batch"
            ) as mock_check_memory,
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

            # Create photos for testing
            photo1 = Photo(
                id=1,
                content_hash="hash1",
                file_path="/path/to/photo1.jpg",
                downsample_path="/path/to/downsample1.jpg",
                exif_datetime="2020:01:01 12:00:00",
                make="Canon",
                model="EOS 5D",
                perceptual_hash="phash1",
                created_at=datetime(2020, 1, 1, 12, 0, 0),
            )
            mock_get_photos.return_value = [photo1]

            # Set batch_size to -1 (invalid)
            mock_config.context.batch_size = -1

            # Mock memory check to return ok
            mock_check_memory.return_value = {
                "status": "ok",
                "available_mb": 200.0,
                "message": "Memory OK",
            }

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Should have processed the photo (with batch_size corrected to 1)
            assert mock_process_photo.call_count == 1
            mock_process_photo.assert_called_once_with(photo1)

            # Should mark complete with 1 photo processed
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=1
            )

    def test_check_memory_with_zero_thresholds(self, mock_config, mock_analyzer):
        """Test _check_memory_before_batch with zero memory thresholds."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch("photochron.pipeline.stages.context_layer.psutil") as mock_psutil,
            patch("photochron.pipeline.stages.context_layer.PSUTIL_AVAILABLE", True),
        ):
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

            # Set thresholds to 0
            mock_config.context.memory_warning_threshold_mb = 0
            mock_config.context.memory_critical_threshold_mb = 0

            # Mock psutil to return low memory (10MB available)
            mock_virtual_memory = Mock()
            mock_virtual_memory.available = 10 * 1024 * 1024  # 10MB in bytes
            mock_psutil.virtual_memory.return_value = mock_virtual_memory

            stage = ContextLayerStage()
            result = stage._check_memory_before_batch()

            # With thresholds at 0, 10MB should be "ok" (not warning or critical)
            assert result["status"] == "ok"
            assert result["available_mb"] == 10.0
            assert "Memory OK" in result["message"]

    def test_check_memory_with_negative_thresholds(self, mock_config, mock_analyzer):
        """Test _check_memory_before_batch with negative memory thresholds."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch("photochron.pipeline.stages.context_layer.psutil") as mock_psutil,
            patch("photochron.pipeline.stages.context_layer.PSUTIL_AVAILABLE", True),
        ):
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

            # Set thresholds to negative values
            mock_config.context.memory_warning_threshold_mb = -50
            mock_config.context.memory_critical_threshold_mb = -100

            # Mock psutil to return memory (100MB available)
            mock_virtual_memory = Mock()
            mock_virtual_memory.available = 100 * 1024 * 1024  # 100MB in bytes
            mock_psutil.virtual_memory.return_value = mock_virtual_memory

            stage = ContextLayerStage()
            result = stage._check_memory_before_batch()

            # With negative thresholds, 100MB should be "ok"
            # (100 > -50 and 100 > -100)
            assert result["status"] == "ok"
            assert result["available_mb"] == 100.0
            assert "Memory OK" in result["message"]

    def test_check_memory_exact_at_warning_threshold(self, mock_config, mock_analyzer):
        """Test _check_memory_before_batch when memory is exactly at warning threshold."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch("photochron.pipeline.stages.context_layer.psutil") as mock_psutil,
            patch("photochron.pipeline.stages.context_layer.PSUTIL_AVAILABLE", True),
        ):
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

            # Mock psutil to return memory exactly at warning threshold (100MB)
            mock_virtual_memory = Mock()
            mock_virtual_memory.available = 100 * 1024 * 1024  # 100MB in bytes
            mock_psutil.virtual_memory.return_value = mock_virtual_memory

            stage = ContextLayerStage()
            result = stage._check_memory_before_batch()

            # Exactly at threshold should be "ok" (not warning)
            # The condition is: available_memory_mb < memory_warning_threshold_mb
            # So 100 < 100 is False, so it should be "ok"
            assert result["status"] == "ok"
            assert result["available_mb"] == 100.0
            assert "Memory OK" in result["message"]

    def test_check_memory_exact_at_critical_threshold(self, mock_config, mock_analyzer):
        """Test _check_memory_before_batch when memory is exactly at critical threshold."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch("photochron.pipeline.stages.context_layer.psutil") as mock_psutil,
            patch("photochron.pipeline.stages.context_layer.PSUTIL_AVAILABLE", True),
        ):
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

            # Mock psutil to return memory exactly at critical threshold (50MB)
            mock_virtual_memory = Mock()
            mock_virtual_memory.available = 50 * 1024 * 1024  # 50MB in bytes
            mock_psutil.virtual_memory.return_value = mock_virtual_memory

            stage = ContextLayerStage()
            result = stage._check_memory_before_batch()

            # Exactly at critical threshold should be "warning" (not critical)
            # The condition is: available_memory_mb < memory_critical_threshold_mb
            # So 50 < 50 is False, so it should check warning threshold next
            # 50 < 100 is True, so it should be "warning"
            assert result["status"] == "warning"
            assert result["available_mb"] == 50.0
            assert "Low memory" in result["message"]
            assert "50.0MB < 100MB" in result["message"]

    def test_memory_check_integration_with_config_changes(
        self, mock_config, mock_analyzer
    ):
        """Test that memory check correctly uses config values."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch("photochron.pipeline.stages.context_layer.psutil") as mock_psutil,
            patch("photochron.pipeline.stages.context_layer.PSUTIL_AVAILABLE", True),
        ):
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

            # Set custom thresholds
            mock_config.context.memory_warning_threshold_mb = 200
            mock_config.context.memory_critical_threshold_mb = 100

            # Mock psutil to return memory between warning and critical (150MB)
            mock_virtual_memory = Mock()
            mock_virtual_memory.available = 150 * 1024 * 1024  # 150MB in bytes
            mock_psutil.virtual_memory.return_value = mock_virtual_memory

            stage = ContextLayerStage()
            result = stage._check_memory_before_batch()

            # 150MB is < 200MB (warning) but > 100MB (critical), so should be "warning"
            assert result["status"] == "warning"
            assert result["available_mb"] == 150.0
            assert "150.0MB < 200MB" in result["message"]

    def test_run_with_multiple_batches_and_memory_changes(
        self, mock_config, mock_analyzer
    ):
        """Test run() with multiple batches where memory status changes between batches."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch.object(ContextLayerStage, "mark_complete") as mock_mark_complete,
            patch.object(
                ContextLayerStage, "_get_photos_without_context"
            ) as mock_get_photos,
            patch.object(ContextLayerStage, "_process_photo") as mock_process_photo,
            patch.object(
                ContextLayerStage, "_check_memory_before_batch"
            ) as mock_check_memory,
            patch("time.sleep") as mock_sleep,
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

            # Create 4 photos for testing
            photos = [
                Photo(
                    id=i,
                    content_hash=f"hash{i}",
                    file_path=f"/path/to/photo{i}.jpg",
                    downsample_path=f"/path/to/downsample{i}.jpg",
                    exif_datetime="2020:01:01 12:00:00",
                    make="Canon",
                    model="EOS 5D",
                    perceptual_hash=f"phash{i}",
                    created_at=datetime(2020, 1, 1, 12, 0, 0),
                )
                for i in range(1, 5)
            ]
            mock_get_photos.return_value = photos

            # Set batch size to 2 so we have 2 batches
            mock_config.context.batch_size = 2

            # Mock memory check to return:
            # - ok for first batch
            # - critical for second batch (skip)
            mock_check_memory.side_effect = [
                {"status": "ok", "available_mb": 200.0, "message": "ok"},
                {"status": "critical", "available_mb": 30.0, "message": "critical"},
            ]

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Should have checked memory 2 times (once per batch)
            assert mock_check_memory.call_count == 2

            # Should have slept once when memory was critical
            mock_sleep.assert_called_once_with(30)  # memory_retry_delay_seconds = 30

            # Should have processed only first 2 photos (second batch skipped)
            assert mock_process_photo.call_count == 2
            mock_process_photo.assert_has_calls(
                [
                    call(photos[0]),
                    call(photos[1]),
                ]
            )

            # Should mark complete with 2 photos processed (not 4)
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=2
            )

    def test_memory_check_message_formatting(self, mock_config, mock_analyzer):
        """Test that memory check messages are properly formatted."""
        with (
            patch(
                "photochron.pipeline.stages.context_layer.get_config",
                return_value=mock_config,
            ),
            patch(
                "photochron.pipeline.stages.context_layer.ContextAnalyzer",
                return_value=mock_analyzer,
            ),
            patch("photochron.pipeline.stages.context_layer.psutil") as mock_psutil,
            patch("photochron.pipeline.stages.context_layer.PSUTIL_AVAILABLE", True),
        ):
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

            # Test with fractional MB value
            mock_virtual_memory = Mock()
            mock_virtual_memory.available = 123456789  # ~117.74MB
            mock_psutil.virtual_memory.return_value = mock_virtual_memory

            stage = ContextLayerStage()
            result = stage._check_memory_before_batch()

            # Should format with 1 decimal place
            assert result["status"] == "ok"  # 117.74 > 100
            assert abs(result["available_mb"] - 117.74) < 0.1
            assert "Memory OK" in result["message"]
            assert "117.7MB" in result["message"] or "117.8MB" in result["message"]
