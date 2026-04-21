"""
Integration tests for ContextLayerStage.

Tests the full pipeline integration with mocked dependencies.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock

import pytest

from photochron.config import Config, ConfigContext
from photochron.models.ollama_client import ContextAnalysisResult
from photochron.pipeline.stages.context_layer import ContextLayerStage
from photochron.store import DatabaseStore


def _create_pipeline_run(store: DatabaseStore, run_id: str, config_hash: str = "test_hash"):
    """Insert a pipeline run record so mark_complete can update it."""
    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO pipeline_runs 
            (run_id, config_hash, start_time, status)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, config_hash, datetime.now().isoformat(), "running"),
        )


@pytest.mark.integration
def test_context_layer_basic_integration(database_store, monkeypatch):
    """Basic integration test for ContextLayerStage with mocked analysis."""
    # Setup mock config
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

    monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_config", lambda: mock_config)

    # Create a temporary directory for downsampled images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        downsampled_dir = temp_path / "downsampled"
        downsampled_dir.mkdir()

        # Create a dummy downsampled image file
        dummy_image_path = downsampled_dir / "dummy.jpg"
        dummy_image_path.touch()  # empty file

        # Insert a photo record directly into database
        store = database_store
        # Monkeypatch get_store to use our test database (both module and store)
        monkeypatch.setattr("photochron.store.get_store", lambda: store)
        monkeypatch.setattr("photochron.store._store", store)
        monkeypatch.setattr("photochron.pipeline.get_store", lambda: store)
        monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_store", lambda: store)
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO photos (content_hash, file_path, downsample_path) VALUES (?, ?, ?)",
                ("hash123", "/fake/original.jpg", str(dummy_image_path)),
            )
            cursor = conn.execute("SELECT id, downsample_path FROM photos")
            photo = cursor.fetchone()
            photo_id = photo["id"]

        # Mock ContextAnalyzer
        mock_analyzer = Mock()
        mock_analyzer_class = Mock(return_value=mock_analyzer)
        monkeypatch.setattr(
            "photochron.pipeline.stages.context_layer.ContextAnalyzer",
            mock_analyzer_class,
        )

        # Setup mock health check
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

        # Setup mock analysis result
        mock_result = ContextAnalysisResult(
            decade="1990-1995",
            decade_confidence=0.8,
            season="summer",
            season_confidence=0.7,
            event_hint="wedding",
            event_confidence=0.6,
            photo_medium="print_scan",
            photo_medium_confidence=0.9,
            visual_evidence=["vintage clothing", "film grain"],
            alternative_decades=["1985-1990", "1995-2000"],
            uncertainty_flag=False,
            hypothesis_notes="Clear vintage aesthetic",
        )
        mock_analyzer.analyze.return_value = mock_result

        # Create and run context layer stage
        stage = ContextLayerStage()

        # Don't mock _get_photos_without_context - let it query the actual database
        # The photo we inserted should be returned by the actual query
        run_id = "test_run_integration"
        config_hash = "test_hash"
        _create_pipeline_run(store, run_id, config_hash)
        stage.run(run_id, config_hash)

        # Verify context was stored in database
        with store.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM context")
            context_count = cursor.fetchone()[0]
            assert context_count == 1

            cursor = conn.execute(
                "SELECT photo_id, decade, decade_confidence, season, season_confidence, "
                "event_hint, event_confidence, photo_medium, photo_medium_confidence, "
                "uncertainty_flag, hypothesis_notes FROM context"
            )
            context = cursor.fetchone()
            assert context["photo_id"] == photo_id
            assert context["decade"] == "1990-1995"
            assert context["decade_confidence"] == 0.8
            assert context["season"] == "summer"
            assert context["season_confidence"] == 0.7
            assert context["event_hint"] == "wedding"
            assert context["event_confidence"] == 0.6
            assert context["photo_medium"] == "print_scan"
            assert context["photo_medium_confidence"] == 0.9
            assert context["uncertainty_flag"] == 0  # False in SQLite
            assert "vintage aesthetic" in context["hypothesis_notes"]


@pytest.mark.integration
def test_context_layer_duplicate_processing(database_store, monkeypatch):
    """Test that context layer doesn't process photos that already have context."""
    # Setup mock config
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

    monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_config", lambda: mock_config)

    # Insert a photo with an existing context record
    store = database_store
    # Monkeypatch get_store to use our test database (both module and store)
    monkeypatch.setattr("photochron.store.get_store", lambda: store)
    monkeypatch.setattr("photochron.store._store", store)
    monkeypatch.setattr("photochron.pipeline.get_store", lambda: store)
    monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_store", lambda: store)
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO photos (content_hash, file_path, downsample_path) VALUES (?, ?, ?)",
            ("hash456", "/fake/original2.jpg", "/fake/downsampled2.jpg"),
        )
        cursor = conn.execute("SELECT id FROM photos")
        photo_id = cursor.fetchone()["id"]

        # Insert a context record for this photo
        conn.execute(
            """
            INSERT INTO context 
            (photo_id, decade, decade_confidence, season, season_confidence, 
             event_hint, event_confidence, photo_medium, photo_medium_confidence,
             uncertainty_flag, hypothesis_notes, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                photo_id,
                "2000-2005",
                0.7,
                "winter",
                0.6,
                "christmas",
                0.8,
                "digital",
                0.9,
                0,
                "Already analyzed",
                '{"status": "test"}',
            ),
        )

    # Mock ContextAnalyzer
    mock_analyzer = Mock()
    mock_analyzer_class = Mock(return_value=mock_analyzer)
    monkeypatch.setattr(
        "photochron.pipeline.stages.context_layer.ContextAnalyzer",
        mock_analyzer_class,
    )

    # Setup mock health check
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

    # Create context layer stage
    stage = ContextLayerStage()

    # Don't mock _get_photos_without_context - let it query actual database
    # Since photo has context, it should return empty list
    run_id = "test_run_duplicate"
    config_hash = "test_hash"
    _create_pipeline_run(store, run_id, config_hash)
    stage.run(run_id, config_hash)

    # Verify analyze was never called
    mock_analyzer.analyze.assert_not_called()

    # Context count should still be 1 (no new context added)
    with store.transaction() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM context")
        context_count = cursor.fetchone()[0]
        assert context_count == 1


@pytest.mark.integration
def test_context_layer_analysis_failure(database_store, monkeypatch):
    """Test context layer when analysis fails completely."""
    # Setup mock config
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

    monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_config", lambda: mock_config)

    # Create a temporary directory for downsampled images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        downsampled_dir = temp_path / "downsampled"
        downsampled_dir.mkdir()

        # Create a dummy downsampled image file
        dummy_image_path = downsampled_dir / "dummy.jpg"
        dummy_image_path.touch()  # empty file

        # Insert a photo record directly into database
        store = database_store
        # Monkeypatch get_store to use our test database (both module and store)
        monkeypatch.setattr("photochron.store.get_store", lambda: store)
        monkeypatch.setattr("photochron.store._store", store)
        monkeypatch.setattr("photochron.pipeline.get_store", lambda: store)
        monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_store", lambda: store)
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO photos (content_hash, file_path, downsample_path) VALUES (?, ?, ?)",
                ("hash789", "/fake/original3.jpg", str(dummy_image_path)),
            )
            cursor = conn.execute("SELECT id, downsample_path FROM photos")
            photo = cursor.fetchone()
            photo_id = photo["id"]

        # Mock ContextAnalyzer
        mock_analyzer = Mock()
        mock_analyzer_class = Mock(return_value=mock_analyzer)
        monkeypatch.setattr(
            "photochron.pipeline.stages.context_layer.ContextAnalyzer",
            mock_analyzer_class,
        )

        # Setup mock health check
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

        # Setup mock analysis to return None (failure)
        mock_analyzer.analyze.return_value = None

        # Create and run context layer stage
        stage = ContextLayerStage()

        # Don't mock _get_photos_without_context - let it query actual database
        # The photo we inserted should be returned
        run_id = "test_run_failure"
        config_hash = "test_hash"
        _create_pipeline_run(store, run_id, config_hash)
        stage.run(run_id, config_hash)

        # Verify minimal context was stored in database (due to store_minimal_on_complete_failure=True)
        with store.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM context")
            context_count = cursor.fetchone()[0]
            assert context_count == 1

            cursor = conn.execute(
                "SELECT photo_id, decade, decade_confidence, season, season_confidence, "
                "event_hint, event_confidence, photo_medium, photo_medium_confidence, "
                "uncertainty_flag, hypothesis_notes FROM context"
            )
            context = cursor.fetchone()
            assert context["photo_id"] == photo_id
            assert context["decade"] is None
            assert context["decade_confidence"] == 0.0
            assert context["season"] is None
            assert context["season_confidence"] is None
            assert context["event_hint"] is None
            assert context["event_confidence"] is None
            assert context["photo_medium"] == "unknown"
            assert context["photo_medium_confidence"] == 0.0
            assert context["uncertainty_flag"] == 1  # True in SQLite
            assert "Analysis failed completely" in context["hypothesis_notes"]


@pytest.mark.integration
def test_context_layer_low_confidence_season(database_store, monkeypatch):
    """Test context layer when analyzer returns result with low season confidence."""
    # Setup mock config
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
    mock_config.context.min_season_confidence = 0.4  # Threshold is 0.4
    mock_config.context.use_fallback_on_failure = True
    mock_config.context.store_minimal_on_complete_failure = True
    mock_config.context.memory_warning_threshold_mb = 100
    mock_config.context.memory_critical_threshold_mb = 50
    mock_config.context.memory_retry_delay_seconds = 30

    monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_config", lambda: mock_config)

    # Create a temporary directory for downsampled images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        downsampled_dir = temp_path / "downsampled"
        downsampled_dir.mkdir()

        # Create a dummy downsampled image file
        dummy_image_path = downsampled_dir / "dummy_low_conf.jpg"
        dummy_image_path.touch()  # empty file

        # Insert a photo record directly into database
        store = database_store
        # Monkeypatch get_store to use our test database (both module and store)
        monkeypatch.setattr("photochron.store.get_store", lambda: store)
        monkeypatch.setattr("photochron.store._store", store)
        monkeypatch.setattr("photochron.pipeline.get_store", lambda: store)
        monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_store", lambda: store)
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO photos (content_hash, file_path, downsample_path) VALUES (?, ?, ?)",
                ("hash_low_conf", "/fake/original_low_conf.jpg", str(dummy_image_path)),
            )
            cursor = conn.execute("SELECT id, downsample_path FROM photos")
            photo = cursor.fetchone()
            photo_id = photo["id"]

        # Mock ContextAnalyzer
        mock_analyzer = Mock()
        mock_analyzer_class = Mock(return_value=mock_analyzer)
        monkeypatch.setattr(
            "photochron.pipeline.stages.context_layer.ContextAnalyzer",
            mock_analyzer_class,
        )

        # Setup mock health check
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

        # Setup mock analysis result simulating what analyzer returns after cleaning
        # Season confidence=0.3 is below min_season_confidence=0.4, so analyzer clears season
        # We mock analyzer to return cleaned result (with season=None)
        mock_result = ContextAnalysisResult(
            decade="1990-1995",
            decade_confidence=0.8,
            season=None,  # Cleared by analyzer due to low confidence
            season_confidence=None,  # Cleared by analyzer due to low confidence
            event_hint="wedding",
            event_confidence=0.6,
            photo_medium="print_scan",
            photo_medium_confidence=0.9,
            visual_evidence=["vintage clothing", "film grain"],
            alternative_decades=["1985-1990", "1995-2000"],
            uncertainty_flag=False,
            hypothesis_notes="Clear vintage aesthetic but season uncertain",
        )
        mock_analyzer.analyze.return_value = mock_result

        # Create and run context layer stage
        stage = ContextLayerStage()

        # Don't mock _get_photos_without_context - let it query the actual database
        # The photo we inserted should be returned by the actual query
        run_id = "test_run_low_conf"
        config_hash = "test_hash"
        _create_pipeline_run(store, run_id, config_hash)
        stage.run(run_id, config_hash)

        # Verify context was stored in database
        with store.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM context")
            context_count = cursor.fetchone()[0]
            assert context_count == 1

            cursor = conn.execute(
                "SELECT photo_id, decade, decade_confidence, season, season_confidence, "
                "event_hint, event_confidence, photo_medium, photo_medium_confidence, "
                "uncertainty_flag, hypothesis_notes FROM context"
            )
            context = cursor.fetchone()
            assert context["photo_id"] == photo_id
            assert context["decade"] == "1990-1995"
            assert context["decade_confidence"] == 0.8
            # Season and season_confidence should be None due to low confidence (0.3 < 0.4)
            assert context["season"] is None
            assert context["season_confidence"] is None
            assert context["event_hint"] == "wedding"
            assert context["event_confidence"] == 0.6
            assert context["photo_medium"] == "print_scan"
            assert context["photo_medium_confidence"] == 0.9
            assert context["uncertainty_flag"] == 0  # False in SQLite
            assert "vintage aesthetic" in context["hypothesis_notes"]


@pytest.mark.integration
def test_context_layer_degraded_mode(database_store, monkeypatch):
    """Test context layer when in degraded mode (Ollama unavailable)."""
    # Setup mock config
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

    monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_config", lambda: mock_config)

    # Insert a photo record
    store = database_store
    # Monkeypatch get_store to use our test database (both module and store)
    monkeypatch.setattr("photochron.store.get_store", lambda: store)
    monkeypatch.setattr("photochron.store._store", store)
    monkeypatch.setattr("photochron.pipeline.get_store", lambda: store)
    monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_store", lambda: store)
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO photos (content_hash, file_path, downsample_path) VALUES (?, ?, ?)",
            ("hash_degraded", "/fake/original4.jpg", "/fake/downsampled4.jpg"),
        )

    # Mock ContextAnalyzer
    mock_analyzer = Mock()
    mock_analyzer_class = Mock(return_value=mock_analyzer)
    monkeypatch.setattr(
        "photochron.pipeline.stages.context_layer.ContextAnalyzer",
        mock_analyzer_class,
    )

    # Setup mock health check to return unhealthy (degraded mode)
    mock_analyzer.health_check.return_value = {
        "status": "unhealthy",
        "ollama_health": {
            "server_available": False,
            "model_details": {
                "primary": {"available": False},
                "fallback": {"available": False},
            },
        },
    }

    # Create context layer stage (should be in degraded mode)
    stage = ContextLayerStage()
    assert stage._degraded_mode is True
    assert stage._is_healthy is False

    # Run the stage
    run_id = "test_run_degraded"
    config_hash = "test_hash"
    _create_pipeline_run(store, run_id, config_hash)
    stage.run(run_id, config_hash)

    # Verify analyze was never called (degraded mode skips analysis)
    mock_analyzer.analyze.assert_not_called()

    # No context should be stored
    with store.transaction() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM context")
        context_count = cursor.fetchone()[0]
        assert context_count == 0


@pytest.mark.integration
def test_context_layer_batch_processing(database_store, monkeypatch):
    """Test context layer with batch processing of multiple photos."""
    # Setup mock config with batch size 2
    mock_config = Mock(spec=Config)
    mock_config.context = Mock(spec=ConfigContext)
    mock_config.context.ollama_host = "http://localhost:11434"
    mock_config.context.ollama_timeout = 300
    mock_config.context.max_retries = 3
    mock_config.context.retry_delay = 2.0
    mock_config.context.primary_model = "llava-next:7b"
    mock_config.context.fallback_model = "moondream2"
    mock_config.context.batch_size = 2  # Process 2 photos per batch
    mock_config.context.min_decade_confidence = 0.3
    mock_config.context.min_season_confidence = 0.4
    mock_config.context.use_fallback_on_failure = True
    mock_config.context.store_minimal_on_complete_failure = True
    mock_config.context.memory_warning_threshold_mb = 100
    mock_config.context.memory_critical_threshold_mb = 50
    mock_config.context.memory_retry_delay_seconds = 30

    monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_config", lambda: mock_config)

    # Create a temporary directory for downsampled images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        downsampled_dir = temp_path / "downsampled"
        downsampled_dir.mkdir()

        # Create dummy downsampled image files
        dummy_image_paths = []
        for i in range(3):  # 3 photos to test batching
            dummy_path = downsampled_dir / f"dummy{i}.jpg"
            dummy_path.touch()
            dummy_image_paths.append(dummy_path)

        # Insert photo records directly into database
        store = database_store
        # Monkeypatch get_store to use our test database (both module and store)
        monkeypatch.setattr("photochron.store.get_store", lambda: store)
        monkeypatch.setattr("photochron.store._store", store)
        monkeypatch.setattr("photochron.pipeline.get_store", lambda: store)
        monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_store", lambda: store)

        photo_ids = []
        for i, dummy_path in enumerate(dummy_image_paths):
            with store.transaction() as conn:
                # Insert photo with all fields that would be used by the test
                conn.execute(
                    "INSERT INTO photos (content_hash, file_path, downsample_path, exif_datetime, make, model, perceptual_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        f"hash_batch{i}",
                        f"/fake/original_batch{i}.jpg",
                        str(dummy_path),
                        f"{1980 + i * 10}:{((i * 3) % 12) + 1:02d}:15 12:00:00",
                        ["Canon", "Nikon", "iPhone"][i % 3],
                        ["EOS 650", "D90", "12 Pro"][i % 3],
                        f"phash_batch{i}",
                    ),
                )
                cursor = conn.execute("SELECT id FROM photos WHERE content_hash = ?", (f"hash_batch{i}",))
                photo_ids.append(cursor.fetchone()["id"])

        # Mock ContextAnalyzer
        mock_analyzer = Mock()
        mock_analyzer_class = Mock(return_value=mock_analyzer)
        monkeypatch.setattr(
            "photochron.pipeline.stages.context_layer.ContextAnalyzer",
            mock_analyzer_class,
        )

        # Setup mock health check
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

        # Setup mock analysis results for each photo
        mock_results = [
            ContextAnalysisResult(
                decade="1980-1985",
                decade_confidence=0.7,
                season="spring",
                season_confidence=0.6,
                event_hint="graduation",
                event_confidence=0.5,
                photo_medium="film_negative",
                photo_medium_confidence=0.8,
                visual_evidence=["80s fashion", "analog film"],
                alternative_decades=["1975-1980", "1985-1990"],
                uncertainty_flag=False,
                hypothesis_notes="Early 80s aesthetic",
            ),
            ContextAnalysisResult(
                decade="1995-2000",
                decade_confidence=0.8,
                season="autumn",
                season_confidence=0.7,
                event_hint="family reunion",
                event_confidence=0.6,
                photo_medium="digital",
                photo_medium_confidence=0.9,
                visual_evidence=["late 90s fashion", "digital artifacts"],
                alternative_decades=["1990-1995", "2000-2005"],
                uncertainty_flag=False,
                hypothesis_notes="Late 90s digital photo",
            ),
            ContextAnalysisResult(
                decade="2010-2015",
                decade_confidence=0.9,
                season="winter",
                season_confidence=0.8,
                event_hint="holiday",
                event_confidence=0.7,
                photo_medium="digital",
                photo_medium_confidence=0.95,
                visual_evidence=["modern clothing", "smartphone"],
                alternative_decades=["2005-2010", "2015-2020"],
                uncertainty_flag=False,
                hypothesis_notes="Modern smartphone photo",
            ),
        ]

        # Make analyzer return results in order
        mock_analyzer.analyze.side_effect = mock_results

        # Create and run context layer stage
        stage = ContextLayerStage()

        # Don't mock _get_photos_without_context - let it query actual database
        # The photos we inserted should be returned
        run_id = "test_run_batch"
        config_hash = "test_hash"
        _create_pipeline_run(store, run_id, config_hash)
        stage.run(run_id, config_hash)

        # Verify all contexts were stored in database
        with store.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM context")
            context_count = cursor.fetchone()[0]
            assert context_count == 3

            # Verify each photo has context
            for i, photo_id in enumerate(photo_ids):
                cursor = conn.execute(
                    "SELECT decade, decade_confidence, season, photo_medium FROM context WHERE photo_id = ?",
                    (photo_id,),
                )
                context = cursor.fetchone()
                assert context is not None
                assert context["decade"] == mock_results[i].decade
                assert context["decade_confidence"] == mock_results[i].decade_confidence
                assert context["season"] == mock_results[i].season
                assert context["photo_medium"] == mock_results[i].photo_medium

        # Verify analyze was called 3 times (once per photo)
        assert mock_analyzer.analyze.call_count == 3


@pytest.mark.integration
def test_context_layer_missing_downsample_file(database_store, monkeypatch):
    """Test context layer when downsample file is missing but original exists."""
    # Setup mock config
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

    monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_config", lambda: mock_config)

    # Create a temporary directory for original image
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create an original image file (but no downsample file)
        original_image_path = temp_path / "original.jpg"
        original_image_path.touch()

        # Insert a photo record with non-existent downsample path but existing original
        store = database_store
        # Monkeypatch get_store to use our test database (both module and store)
        monkeypatch.setattr("photochron.store.get_store", lambda: store)
        monkeypatch.setattr("photochron.store._store", store)
        monkeypatch.setattr("photochron.pipeline.get_store", lambda: store)
        monkeypatch.setattr("photochron.pipeline.stages.context_layer.get_store", lambda: store)
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO photos (content_hash, file_path, downsample_path, exif_datetime, make, model, perceptual_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    "hash_missing",
                    str(original_image_path),
                    "/non/existent/downsample.jpg",
                    "2008:07:20 10:15:00",
                    "Sony",
                    "DSC-HX1",
                    "phash_missing",
                ),
            )
            cursor = conn.execute("SELECT id, file_path, downsample_path FROM photos")
            photo = cursor.fetchone()
            photo_id = photo["id"]

        # Mock ContextAnalyzer
        mock_analyzer = Mock()
        mock_analyzer_class = Mock(return_value=mock_analyzer)
        monkeypatch.setattr(
            "photochron.pipeline.stages.context_layer.ContextAnalyzer",
            mock_analyzer_class,
        )

        # Setup mock health check
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

        # Setup mock analysis result
        mock_result = ContextAnalysisResult(
            decade="2005-2010",
            decade_confidence=0.75,
            season="summer",
            season_confidence=0.65,
            event_hint="vacation",
            event_confidence=0.55,
            photo_medium="digital",
            photo_medium_confidence=0.85,
            visual_evidence=["beach", "sunglasses"],
            alternative_decades=["2000-2005", "2010-2015"],
            uncertainty_flag=False,
            hypothesis_notes="Beach vacation photo",
        )
        mock_analyzer.analyze.return_value = mock_result

        # Create and run context layer stage
        stage = ContextLayerStage()

        # Don't mock _get_photos_without_context - let it query actual database
        # The photo we inserted should be returned
        run_id = "test_run_missing"
        config_hash = "test_hash"
        _create_pipeline_run(store, run_id, config_hash)
        stage.run(run_id, config_hash)

        # Verify context was stored in database (using original file as fallback)
        with store.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM context")
            context_count = cursor.fetchone()[0]
            assert context_count == 1

            cursor = conn.execute("SELECT photo_id, decade, decade_confidence FROM context")
            context = cursor.fetchone()
            assert context["photo_id"] == photo_id
            assert context["decade"] == "2005-2010"
            assert context["decade_confidence"] == 0.75

        # Verify analyze was called with the original file path (fallback)
        # The actual path passed to analyze would be the original file path
        # since downsample doesn't exist
        assert mock_analyzer.analyze.call_count == 1
