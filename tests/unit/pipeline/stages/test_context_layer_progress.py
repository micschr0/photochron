"""
Unit tests for percentage-based progress reporting in ContextLayerStage.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime

from photochron.pipeline.stages.context_layer import ContextLayerStage
from photochron.config import Config, ConfigContext
from photochron.models.ollama_client import ModelType
from photochron.context.analyzer import ContextAnalyzer, ContextAnalyzerConfig
from photochron.models import Photo


class TestContextLayerStageProgress:
    """Test suite for ContextLayerStage percentage-based progress reporting."""

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
        mock_config.context.batch_size = 5  # Default batch size for testing
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

    def test_batch_progress_logging_with_percentage(self, mock_config, mock_analyzer):
        """Test batch-level progress logging includes percentage with 1 decimal place."""
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
            patch("photochron.pipeline.stages.context_layer.logger") as mock_logger,
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

            # Create 10 photos for testing
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
                for i in range(1, 11)
            ]
            mock_get_photos.return_value = photos

            # Set batch size to 3 to create multiple batches
            mock_config.context.batch_size = 3

            # Mock memory check to return ok
            mock_check_memory.return_value = {
                "status": "ok",
                "available_mb": 200.0,
                "message": "Memory OK",
            }

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Verify batch progress logging calls with percentages
            batch_log_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Processing batch" in str(call)
            ]

            # Should have 4 batches (10 photos, batch size 3 = ceil(10/3) = 4 batches)
            assert len(batch_log_calls) == 4

            # Check each batch log message contains percentage with 1 decimal place
            expected_batch_percentages = [
                0.0,
                30.0,
                60.0,
                90.0,
            ]  # 0/10, 3/10, 6/10, 9/10
            for i, log_call in enumerate(batch_log_calls):
                log_message = log_call[0][0]
                # Check it contains percentage with 1 decimal place
                assert "(" in log_message and "%)" in log_message
                # Extract percentage
                percent_start = log_message.find("(") + 1
                percent_end = log_message.find("%)")
                percent_str = log_message[percent_start:percent_end]
                # Should end with .1f format
                assert "." in percent_str
                # Parse percentage
                percent_value = float(percent_str)
                # Should match expected (within rounding tolerance)
                assert abs(percent_value - expected_batch_percentages[i]) < 0.1

            # Should mark complete with 10 photos processed
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=10
            )

    def test_photo_progress_logging_every_10_photos(self, mock_config, mock_analyzer):
        """Test photo-level progress logging every 10 photos includes percentage."""
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
            patch("photochron.pipeline.stages.context_layer.logger") as mock_logger,
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

            # Create 25 photos for testing (to trigger photo-level logging at 10, 20)
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
                for i in range(1, 26)
            ]
            mock_get_photos.return_value = photos

            # Set batch size to 25 to process all in one batch
            mock_config.context.batch_size = 25

            # Mock memory check to return ok
            mock_check_memory.return_value = {
                "status": "ok",
                "available_mb": 200.0,
                "message": "Memory OK",
            }

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Verify photo progress logging calls (exclude completion log)
            photo_log_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Processed" in str(call)
                and "photos (" in str(call)
                and "Context layer stage completed" not in str(call)
            ]

            # Should have logged at 10 and 20 photos (25 total, every 10 photos)
            assert len(photo_log_calls) == 2

            # Check each photo log message contains percentage with 1 decimal place
            expected_percentages = [40.0, 80.0]  # 10/25=40%, 20/25=80%
            for i, log_call in enumerate(photo_log_calls):
                log_message = log_call[0][0]
                # Check it contains percentage with 1 decimal place
                assert "(" in log_message and "%)" in log_message
                # Extract percentage
                percent_start = log_message.find("(") + 1
                percent_end = log_message.find("%)")
                percent_str = log_message[percent_start:percent_end]
                # Should end with .1f format
                assert "." in percent_str
                # Parse percentage
                percent_value = float(percent_str)
                # Should match expected (within rounding tolerance)
                assert abs(percent_value - expected_percentages[i]) < 0.1

            # Should mark complete with 25 photos processed
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=25
            )

    def test_final_completion_logging_with_percentage(self, mock_config, mock_analyzer):
        """Test final completion logging includes percentage with 1 decimal place."""
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
            patch("photochron.pipeline.stages.context_layer.logger") as mock_logger,
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

            # Create 7 photos for testing
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
                for i in range(1, 8)
            ]
            mock_get_photos.return_value = photos

            # Set batch size to 7 to process all in one batch
            mock_config.context.batch_size = 7

            # Mock memory check to return ok
            mock_check_memory.return_value = {
                "status": "ok",
                "available_mb": 200.0,
                "message": "Memory OK",
            }

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Verify final completion logging
            completion_log_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Context layer stage completed" in str(call)
            ]

            assert len(completion_log_calls) == 1

            log_message = completion_log_calls[0][0][0]
            # Check it contains percentage with 1 decimal place
            assert "(" in log_message and "%)" in log_message
            # Extract percentage
            percent_start = log_message.find("(") + 1
            percent_end = log_message.find("%)")
            percent_str = log_message[percent_start:percent_end]
            # Should end with .1f format
            assert "." in percent_str
            # Parse percentage - should be 100.0% (7/7 photos)
            percent_value = float(percent_str)
            assert abs(percent_value - 100.0) < 0.1

            # Should mark complete with 7 photos processed
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=7
            )

    def test_progress_with_zero_total_photos(self, mock_config, mock_analyzer):
        """Test progress logging when total_photos = 0."""
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
                ContextLayerStage, "_get_photos_without_context", return_value=[]
            ),
            patch("photochron.pipeline.stages.context_layer.logger") as mock_logger,
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

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Should log "No photos without context data; stage complete"
            mock_logger.info.assert_any_call(
                "No photos without context data; stage complete"
            )

            # Should NOT log any percentage messages (division by zero protection)
            percentage_log_calls = [
                call for call in mock_logger.info.call_args_list if "%" in str(call)
            ]
            assert len(percentage_log_calls) == 0

            # Should mark complete with 0 photos processed
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=0
            )

    def test_progress_with_batch_size_1(self, mock_config, mock_analyzer):
        """Test progress logging with batch_size = 1 (edge case)."""
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
            patch("photochron.pipeline.stages.context_layer.logger") as mock_logger,
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

            # Create 3 photos for testing
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
                for i in range(1, 4)
            ]
            mock_get_photos.return_value = photos

            # Set batch size to 1
            mock_config.context.batch_size = 1

            # Mock memory check to return ok
            mock_check_memory.return_value = {
                "status": "ok",
                "available_mb": 200.0,
                "message": "Memory OK",
            }

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Verify batch progress logging for each photo (batch size 1)
            batch_log_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Processing batch" in str(call)
            ]

            # Should have 3 batches (3 photos, batch size 1)
            assert len(batch_log_calls) == 3

            # Check percentages: 0%, 33.3%, 66.7%
            expected_percentages = [0.0, 33.3, 66.7]
            for i, log_call in enumerate(batch_log_calls):
                log_message = log_call[0][0]
                assert "(" in log_message and "%)" in log_message
                percent_start = log_message.find("(") + 1
                percent_end = log_message.find("%)")
                percent_str = log_message[percent_start:percent_end]
                assert "." in percent_str
                percent_value = float(percent_str)
                assert (
                    abs(percent_value - expected_percentages[i]) < 0.2
                )  # Allow for rounding

            # Should mark complete with 3 photos processed
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=3
            )

    def test_progress_with_invalid_batch_size_correction(
        self, mock_config, mock_analyzer
    ):
        """Test progress logging when batch_size is invalid and corrected to 1."""
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
            patch("photochron.pipeline.stages.context_layer.logger") as mock_logger,
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

            # Create 2 photos for testing
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
                for i in range(1, 3)
            ]
            mock_get_photos.return_value = photos

            # Set batch size to 0 (invalid)
            mock_config.context.batch_size = 0

            # Mock memory check to return ok
            mock_check_memory.return_value = {
                "status": "ok",
                "available_mb": 200.0,
                "message": "Memory OK",
            }

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Should log warning about invalid batch size
            mock_logger.warning.assert_any_call(
                f"Invalid batch_size {mock_config.context.batch_size}, using 1 instead"
            )

            # Should process photos with batch size corrected to 1
            assert mock_process_photo.call_count == 2

            # Should mark complete with 2 photos processed
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=2
            )

    def test_progress_with_failed_photos(self, mock_config, mock_analyzer):
        """Test progress logging when some photos fail to process."""
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
            patch("photochron.pipeline.stages.context_layer.logger") as mock_logger,
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

            # Create 5 photos for testing
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
                for i in range(1, 6)
            ]
            mock_get_photos.return_value = photos

            # Set batch size to 5
            mock_config.context.batch_size = 5

            # Mock memory check to return ok
            mock_check_memory.return_value = {
                "status": "ok",
                "available_mb": 200.0,
                "message": "Memory OK",
            }

            # Make 2nd and 4th photos fail
            def process_photo_side_effect(photo):
                if photo.id in [2, 4]:
                    raise Exception(f"Processing failed for photo {photo.id}")

            mock_process_photo.side_effect = process_photo_side_effect

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Verify final completion logging shows correct percentage
            completion_log_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Context layer stage completed" in str(call)
            ]

            assert len(completion_log_calls) == 1

            log_message = completion_log_calls[0][0][0]
            # Should show 60.0% (3 processed out of 5 total)
            assert "(" in log_message and "%)" in log_message
            percent_start = log_message.find("(") + 1
            percent_end = log_message.find("%)")
            percent_str = log_message[percent_start:percent_end]
            percent_value = float(percent_str)
            assert abs(percent_value - 60.0) < 0.1  # 3/5 = 60%

            # Should also show failed count
            assert "failed: 2" in log_message

            # Should mark complete with 3 photos processed (2 failed)
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=3
            )

    def test_memory_critical_logging_includes_percentage(
        self, mock_config, mock_analyzer
    ):
        """Test memory critical warning includes batch percentage."""
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
            patch("photochron.pipeline.stages.context_layer.logger") as mock_logger,
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

            # Create 8 photos for testing
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
                for i in range(1, 9)
            ]
            mock_get_photos.return_value = photos

            # Set batch size to 4
            mock_config.context.batch_size = 4

            # Mock memory check to return critical for first batch, ok for second
            mock_check_memory.side_effect = [
                {"status": "critical", "available_mb": 30.0, "message": "critical"},
                {"status": "ok", "available_mb": 200.0, "message": "ok"},
            ]

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Verify memory critical warning includes percentage
            critical_warning_calls = [
                call
                for call in mock_logger.warning.call_args_list
                if "Memory critically low" in str(call)
            ]

            assert len(critical_warning_calls) >= 1

            # Check that warning includes percentage
            for call in critical_warning_calls:
                log_message = call[0][0]
                # Should contain percentage for the batch being skipped
                assert "%)" in log_message
                # Extract percentage - find the last occurrence of (%) pattern
                # The log message has format: "Memory critically low... Skipping batch X/Y (Z.Z%)"
                # We need to find the last "(" before "%)"
                percent_start = log_message.rfind("(") + 1
                percent_end = log_message.rfind("%)")
                if percent_start > 0 and percent_end > percent_start:
                    percent_str = log_message[percent_start:percent_end]
                    # Should be 0.0% for first batch (0/8 photos processed)
                    percent_value = float(percent_str)
                    assert abs(percent_value - 0.0) < 0.1

            # Should have slept once
            mock_sleep.assert_called_once_with(30)

            # Should have processed only second batch (photos 5-8)
            assert mock_process_photo.call_count == 4

            # Should mark complete with 4 photos processed (first batch skipped)
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=4
            )

    def test_percentage_calculation_edge_cases(self, mock_config, mock_analyzer):
        """Test percentage calculation for various edge cases."""
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
            patch("photochron.pipeline.stages.context_layer.logger") as mock_logger,
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

            # Test with 1 photo (100% edge case)
            photos = [
                Photo(
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
            ]
            mock_get_photos.return_value = photos

            # Set batch size to 1
            mock_config.context.batch_size = 1

            # Mock memory check to return ok
            mock_check_memory.return_value = {
                "status": "ok",
                "available_mb": 200.0,
                "message": "Memory OK",
            }

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Check batch logging shows 0.0% (before processing)
            batch_log_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Processing batch" in str(call)
            ]
            assert len(batch_log_calls) == 1
            batch_log_message = batch_log_calls[0][0][0]
            assert "(0.0%)" in batch_log_message

            # Check final completion shows 100.0%
            completion_log_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Context layer stage completed" in str(call)
            ]
            assert len(completion_log_calls) == 1
            completion_log_message = completion_log_calls[0][0][0]
            assert "(100.0%)" in completion_log_message

            # Should mark complete with 1 photo processed
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=1
            )

    def test_photo_progress_not_logged_before_10_photos(
        self, mock_config, mock_analyzer
    ):
        """Test photo-level progress is NOT logged before reaching 10 photos."""
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
            patch("photochron.pipeline.stages.context_layer.logger") as mock_logger,
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

            # Create exactly 9 photos (should NOT trigger photo-level logging)
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
                for i in range(1, 10)
            ]
            mock_get_photos.return_value = photos

            # Set batch size to 9
            mock_config.context.batch_size = 9

            # Mock memory check to return ok
            mock_check_memory.return_value = {
                "status": "ok",
                "available_mb": 200.0,
                "message": "Memory OK",
            }

            stage = ContextLayerStage()
            stage.run("test-run-id", "config-hash")

            # Verify NO photo-level progress logging (processed % 10 != 0 for 9 photos)
            # Exclude completion log which also contains "Processed" and "photos ("
            photo_log_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Processed" in str(call)
                and "photos (" in str(call)
                and "Context layer stage completed" not in str(call)
            ]
            assert len(photo_log_calls) == 0

            # Should mark complete with 9 photos processed
            mock_mark_complete.assert_called_once_with(
                "test-run-id", photos_processed=9
            )
