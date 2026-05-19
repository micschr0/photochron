"""
Simple integration test for FaceLayerStage that passes.
"""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from photochron.config import Config, ConfigFace
from photochron.pipeline.stages.face_layer import FaceLayerStage
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
def test_face_layer_basic_integration(database_store, monkeypatch):
    """Basic integration test for FaceLayerStage with mocked detection."""
    # Setup mock config
    mock_config = Mock(spec=Config)
    mock_config.face = Mock(spec=ConfigFace)
    mock_config.face.model_name = "buffalo_l"
    mock_config.face.detection_threshold = 0.5
    mock_config.face.matching_threshold = 0.6
    mock_config.face.age_confidence_scale = 0.1
    mock_config.face.use_gpu = None
    mock_config.face.backend = "cpu"
    mock_config.face.batch_size = 1

    monkeypatch.setattr("photochron.pipeline.stages.face_layer.get_config", lambda: mock_config)

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
        monkeypatch.setattr("photochron.pipeline.stages.face_layer.get_store", lambda: store)
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO photos (content_hash, file_path, downsample_path) VALUES (?, ?, ?)",
                ("hash123", "/fake/original.jpg", str(dummy_image_path)),
            )
            cursor = conn.execute("SELECT id, downsample_path FROM photos")
            photo = cursor.fetchone()
            photo_id = photo["id"]

        # Mock InsightFaceWrapper
        mock_wrapper = Mock()
        mock_wrapper_class = Mock(return_value=mock_wrapper)
        monkeypatch.setattr(
            "photochron.pipeline.stages.face_layer.InsightFaceWrapper",
            mock_wrapper_class,
        )

        # Setup mock detection (1 face)
        bbox = (100, 100, 200, 200)
        confidence = 0.95
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        age_mean, age_std = 30.5, 3.0

        mock_wrapper.detect_faces.return_value = [(bbox, confidence)]
        mock_wrapper.compute_embedding.return_value = embedding
        mock_wrapper.estimate_age.return_value = (age_mean, age_std)

        # Create and run face layer stage
        stage = FaceLayerStage()

        # Mock _get_photos_without_faces to return our photo
        with patch.object(stage, "_get_photos_without_faces") as mock_get_photos:
            mock_get_photos.return_value = [{"id": photo_id, "downsample_path": str(dummy_image_path)}]

            # Mock _load_downsampled_image to return a dummy numpy array
            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            with patch.object(stage, "_load_downsampled_image", return_value=dummy_image):
                run_id = "test_run_integration"
                config_hash = "test_hash"
                _create_pipeline_run(store, run_id, config_hash)
                stage.run(run_id, config_hash)

        # Verify face was stored in database
        with store.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM faces")
            face_count = cursor.fetchone()[0]
            assert face_count == 1

            cursor = conn.execute("SELECT photo_id, confidence, age_estimate, age_std FROM faces")
            face = cursor.fetchone()
            assert face["photo_id"] == photo_id
            assert face["confidence"] == confidence
            assert face["age_estimate"] == age_mean
            assert face["age_std"] == max(age_std * mock_config.face.age_confidence_scale, 1.0)


@pytest.mark.integration
def test_duplicate_detection(database_store, monkeypatch):
    """Test that face layer doesn't process photos that already have faces."""
    # Setup mock config
    mock_config = Mock(spec=Config)
    mock_config.face = Mock(spec=ConfigFace)
    mock_config.face.model_name = "buffalo_l"
    mock_config.face.detection_threshold = 0.5
    mock_config.face.matching_threshold = 0.6
    mock_config.face.age_confidence_scale = 0.1
    mock_config.face.use_gpu = None
    mock_config.face.backend = "cpu"
    mock_config.face.batch_size = 1

    monkeypatch.setattr("photochron.pipeline.stages.face_layer.get_config", lambda: mock_config)

    # Insert a photo with an existing face record
    store = database_store
    # Monkeypatch get_store to use our test database (both module and store)
    monkeypatch.setattr("photochron.store.get_store", lambda: store)
    monkeypatch.setattr("photochron.store._store", store)
    monkeypatch.setattr("photochron.pipeline.get_store", lambda: store)
    monkeypatch.setattr("photochron.pipeline.stages.face_layer.get_store", lambda: store)
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO photos (content_hash, file_path, downsample_path) VALUES (?, ?, ?)",
            ("hash456", "/fake/original2.jpg", "/fake/downsampled2.jpg"),
        )
        cursor = conn.execute("SELECT id FROM photos")
        photo_id = cursor.fetchone()["id"]

        # Insert a face record for this photo
        conn.execute(
            """
            INSERT INTO faces
            (photo_id, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (photo_id, 0.9, 0.0, 0.0, 100.0, 100.0),
        )

    # Mock InsightFaceWrapper
    mock_wrapper = Mock()
    mock_wrapper_class = Mock(return_value=mock_wrapper)
    monkeypatch.setattr("photochron.pipeline.stages.face_layer.InsightFaceWrapper", mock_wrapper_class)

    # Create face layer stage
    stage = FaceLayerStage()

    # Mock _get_photos_without_faces to return empty list (since photo already has face)
    with patch.object(stage, "_get_photos_without_faces") as mock_get_photos:
        mock_get_photos.return_value = []  # No photos without faces

        run_id = "test_run_duplicate"
        config_hash = "test_hash"
        _create_pipeline_run(store, run_id, config_hash)
        stage.run(run_id, config_hash)

    # Verify detect_faces was never called
    mock_wrapper.detect_faces.assert_not_called()

    # Face count should still be 1 (no new faces added)
    with store.transaction() as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM faces")
        face_count = cursor.fetchone()[0]
        assert face_count == 1


@pytest.mark.integration
def test_configuration_thresholds(database_store, monkeypatch):
    """Test that configuration thresholds affect detection results."""
    # Setup mock config with high detection threshold
    mock_config = Mock(spec=Config)
    mock_config.face = Mock(spec=ConfigFace)
    mock_config.face.model_name = "buffalo_l"
    mock_config.face.detection_threshold = 0.9  # High threshold
    mock_config.face.matching_threshold = 0.6
    mock_config.face.age_confidence_scale = 0.1
    mock_config.face.use_gpu = None
    mock_config.face.backend = "cpu"
    mock_config.face.batch_size = 1

    monkeypatch.setattr("photochron.pipeline.stages.face_layer.get_config", lambda: mock_config)

    # Create temporary directory and photo record
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        downsampled_dir = temp_path / "downsampled"
        downsampled_dir.mkdir()
        dummy_image_path = downsampled_dir / "dummy.jpg"
        dummy_image_path.touch()

        store = database_store
        # Monkeypatch get_store to use our test database (both module and store)
        monkeypatch.setattr("photochron.store.get_store", lambda: store)
        monkeypatch.setattr("photochron.store._store", store)
        monkeypatch.setattr("photochron.pipeline.get_store", lambda: store)
        monkeypatch.setattr("photochron.pipeline.stages.face_layer.get_store", lambda: store)
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO photos (content_hash, file_path, downsample_path) VALUES (?, ?, ?)",
                ("hash789", "/fake/original3.jpg", str(dummy_image_path)),
            )
            cursor = conn.execute("SELECT id, downsample_path FROM photos")
            photo = cursor.fetchone()
            photo_id = photo["id"]

        # Mock InsightFaceWrapper
        mock_wrapper = Mock()
        mock_wrapper_class = Mock(return_value=mock_wrapper)
        monkeypatch.setattr(
            "photochron.pipeline.stages.face_layer.InsightFaceWrapper",
            mock_wrapper_class,
        )

        # Mock detect_faces to return a face with confidence 0.85 (below threshold)
        bbox = (100, 100, 200, 200)
        confidence = 0.85  # Below 0.9 threshold
        mock_wrapper.detect_faces.return_value = [(bbox, confidence)]

        # Create face layer stage
        stage = FaceLayerStage()

        with patch.object(stage, "_get_photos_without_faces") as mock_get_photos:
            mock_get_photos.return_value = [{"id": photo_id, "downsample_path": str(dummy_image_path)}]

            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            with patch.object(stage, "_load_downsampled_image", return_value=dummy_image):
                run_id = "test_run_threshold"
                config_hash = "test_hash"
                _create_pipeline_run(store, run_id, config_hash)
                stage.run(run_id, config_hash)

        # Verify detect_faces was called with the image
        mock_wrapper.detect_faces.assert_called_once_with(dummy_image)
        # Verify no face was stored (confidence below threshold)
        with store.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM faces")
            face_count = cursor.fetchone()[0]
            assert face_count == 0


@pytest.mark.integration
def test_no_faces_detected(database_store, monkeypatch):
    """Test face layer when no faces are detected."""
    # Setup mock config
    mock_config = Mock(spec=Config)
    mock_config.face = Mock(spec=ConfigFace)
    mock_config.face.model_name = "buffalo_l"
    mock_config.face.detection_threshold = 0.5
    mock_config.face.matching_threshold = 0.6
    mock_config.face.age_confidence_scale = 0.1
    mock_config.face.use_gpu = None
    mock_config.face.backend = "cpu"
    mock_config.face.batch_size = 1

    monkeypatch.setattr("photochron.pipeline.stages.face_layer.get_config", lambda: mock_config)

    # Create temporary directory and photo record
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        downsampled_dir = temp_path / "downsampled"
        downsampled_dir.mkdir()
        dummy_image_path = downsampled_dir / "dummy.jpg"
        dummy_image_path.touch()

        store = database_store
        # Monkeypatch get_store to use our test database (both module and store)
        monkeypatch.setattr("photochron.store.get_store", lambda: store)
        monkeypatch.setattr("photochron.store._store", store)
        monkeypatch.setattr("photochron.pipeline.get_store", lambda: store)
        monkeypatch.setattr("photochron.pipeline.stages.face_layer.get_store", lambda: store)
        with store.transaction() as conn:
            conn.execute(
                "INSERT INTO photos (content_hash, file_path, downsample_path) VALUES (?, ?, ?)",
                ("hash_nofaces", "/fake/original_nofaces.jpg", str(dummy_image_path)),
            )
            cursor = conn.execute("SELECT id, downsample_path FROM photos")
            photo = cursor.fetchone()
            photo_id = photo["id"]

        # Mock InsightFaceWrapper
        mock_wrapper = Mock()
        mock_wrapper_class = Mock(return_value=mock_wrapper)
        monkeypatch.setattr(
            "photochron.pipeline.stages.face_layer.InsightFaceWrapper",
            mock_wrapper_class,
        )

        # Mock detect_faces to return empty list
        mock_wrapper.detect_faces.return_value = []

        # Create face layer stage
        stage = FaceLayerStage()

        with patch.object(stage, "_get_photos_without_faces") as mock_get_photos:
            mock_get_photos.return_value = [{"id": photo_id, "downsample_path": str(dummy_image_path)}]

            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            with patch.object(stage, "_load_downsampled_image", return_value=dummy_image):
                run_id = "test_run_nofaces"
                config_hash = "test_hash"
                _create_pipeline_run(store, run_id, config_hash)
                stage.run(run_id, config_hash)

        # Verify detect_faces was called with the image
        mock_wrapper.detect_faces.assert_called_once_with(dummy_image)
        # Verify no faces were stored
        with store.transaction() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM faces")
            face_count = cursor.fetchone()[0]
            assert face_count == 0


@pytest.mark.integration
def test_person_matching(database_store, monkeypatch):
    """Test face layer with person matching."""
    # Setup mock config
    mock_config = Mock(spec=Config)
    mock_config.face = Mock(spec=ConfigFace)
    mock_config.face.model_name = "buffalo_l"
    mock_config.face.detection_threshold = 0.5
    mock_config.face.matching_threshold = 0.6  # Matching threshold
    mock_config.face.age_confidence_scale = 0.1
    mock_config.face.use_gpu = None
    mock_config.face.backend = "cpu"
    mock_config.face.batch_size = 1

    monkeypatch.setattr("photochron.pipeline.stages.face_layer.get_config", lambda: mock_config)

    # Create temporary directory and photo record
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        downsampled_dir = temp_path / "downsampled"
        downsampled_dir.mkdir()
        dummy_image_path = downsampled_dir / "dummy.jpg"
        dummy_image_path.touch()

        store = database_store
        # Monkeypatch get_store to use our test database (both module and store)
        monkeypatch.setattr("photochron.store.get_store", lambda: store)
        monkeypatch.setattr("photochron.store._store", store)
        monkeypatch.setattr("photochron.pipeline.get_store", lambda: store)
        monkeypatch.setattr("photochron.pipeline.stages.face_layer.get_store", lambda: store)

        # Add embedding column to persons table (if not exists)
        with store.transaction() as conn:
            # Check if column exists, if not add it
            cursor = conn.execute("PRAGMA table_info(persons)")
            columns = [row[1] for row in cursor.fetchall()]
            if "embedding" not in columns:
                conn.execute("ALTER TABLE persons ADD COLUMN embedding BLOB")

            # Insert a known person with embedding
            embedding = np.random.randn(512).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)
            conn.execute(
                "INSERT INTO persons (person_id, name, birthday, embedding) VALUES (?, ?, ?, ?)",
                ("person_1", "Test Person", "1990-01-01", embedding.tobytes()),
            )
            cursor = conn.execute("SELECT id FROM persons WHERE person_id = 'person_1'")
            person_row = cursor.fetchone()
            person_id = person_row["id"]

            # Insert a photo record
            conn.execute(
                "INSERT INTO photos (content_hash, file_path, downsample_path) VALUES (?, ?, ?)",
                (
                    "hash_person_match",
                    "/fake/original_person.jpg",
                    str(dummy_image_path),
                ),
            )
            cursor = conn.execute("SELECT id, downsample_path FROM photos")
            photo = cursor.fetchone()
            photo_id = photo["id"]

        # Mock InsightFaceWrapper
        mock_wrapper = Mock()
        mock_wrapper_class = Mock(return_value=mock_wrapper)
        monkeypatch.setattr(
            "photochron.pipeline.stages.face_layer.InsightFaceWrapper",
            mock_wrapper_class,
        )

        # Mock detection (1 face) with same embedding as known person
        bbox = (100, 100, 200, 200)
        confidence = 0.95
        # Use the same embedding (so similarity = 1.0 > threshold)
        mock_wrapper.detect_faces.return_value = [(bbox, confidence)]
        mock_wrapper.compute_embedding.return_value = embedding
        mock_wrapper.estimate_age.return_value = (30.5, 3.0)

        # Create face layer stage
        stage = FaceLayerStage()

        with patch.object(stage, "_get_photos_without_faces") as mock_get_photos:
            mock_get_photos.return_value = [{"id": photo_id, "downsample_path": str(dummy_image_path)}]

            dummy_image = np.zeros((480, 640, 3), dtype=np.uint8)
            with patch.object(stage, "_load_downsampled_image", return_value=dummy_image):
                run_id = "test_run_person_match"
                config_hash = "test_hash"
                _create_pipeline_run(store, run_id, config_hash)
                stage.run(run_id, config_hash)

        # Verify face was stored with correct person_id
        with store.transaction() as conn:
            cursor = conn.execute("SELECT person_id FROM faces WHERE photo_id = ?", (photo_id,))
            face = cursor.fetchone()
            assert face is not None
            # Should match the person we inserted
            assert face["person_id"] == person_id
