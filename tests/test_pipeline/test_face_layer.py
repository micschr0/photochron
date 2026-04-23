"""
Unit tests for the FaceLayerStage.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

from photochron.config import Config, ConfigFace
from photochron.pipeline.stages.face_layer import FaceLayerStage


class TestFaceLayerStage:
    """Test suite for FaceLayerStage."""

    @pytest.fixture
    def stage(self):
        """Create a FaceLayerStage instance with mocked config and wrapper."""
        with (
            patch("photochron.pipeline.stages.face_layer.get_config") as mock_get_config,
            patch("photochron.pipeline.stages.face_layer.InsightFaceWrapper") as mock_wrapper_class,
        ):
            # Setup config mock
            mock_config = Mock(spec=Config)
            mock_config.face = Mock(spec=ConfigFace)
            mock_config.face.model_name = "buffalo_l"
            mock_config.face.detection_threshold = 0.5
            mock_config.face.matching_threshold = 0.6
            mock_config.face.age_confidence_scale = 0.1
            mock_config.face.use_gpu = None
            mock_config.face.backend = "cpu"
            mock_config.face.batch_size = 1
            mock_get_config.return_value = mock_config

            # Setup wrapper mock
            mock_wrapper = Mock()
            mock_wrapper_class.return_value = mock_wrapper

            # Create stage instance (will use mocked config and wrapper)
            stage = FaceLayerStage()
            # Ensure wrapper attribute is our mock
            assert stage.wrapper is mock_wrapper
            return stage

    @pytest.fixture
    def temp_image_dir(self):
        """Create a temporary directory with a test image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (640, 480), color="green")
            img.save(img_path, format="JPEG")
            yield tmpdir, img_path

    def test_name_property(self, stage):
        """Test that stage has correct name."""
        assert stage.name == "face_layer"

    def test_dependencies_property(self, stage):
        """Test that stage depends on ingestion."""
        assert stage.dependencies == ["ingestion"]

    def test_process_photo_no_faces(self, stage):
        """Test processing a photo with no faces detected."""
        # Mock image loading
        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch.object(stage, "_load_downsampled_image", return_value=fake_image):
            # Mock detection returning empty list
            stage.wrapper.detect_faces.return_value = []
            # Call _process_photo
            stage._process_photo(123, Path("/fake/path.jpg"))
            # Ensure store_faces not called because no faces
            # Note: store_faces is not mocked yet, but we can check wrapper calls
            stage.wrapper.detect_faces.assert_called_once_with(fake_image)
            stage.wrapper.compute_embedding.assert_not_called()
            stage.wrapper.estimate_age.assert_not_called()

    def test_process_photo_with_face(self, stage):
        """Test processing a photo with one face."""
        fake_image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)
        confidence = 0.9
        embedding = np.random.randn(512).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        age_mean = 30.5
        age_std = 3.0

        with patch.object(stage, "_load_downsampled_image", return_value=fake_image):
            stage.wrapper.detect_faces.return_value = [(bbox, confidence)]
            stage.wrapper.compute_embedding.return_value = embedding
            stage.wrapper.estimate_age.return_value = (age_mean, age_std)
            # Mock internal methods
            with (
                patch.object(stage, "_match_person", return_value=None) as mock_match,
                patch.object(stage, "_store_faces") as mock_store,
            ):
                stage._process_photo(123, Path("/fake/path.jpg"))

                # Verify wrapper methods called
                stage.wrapper.detect_faces.assert_called_once_with(fake_image)
                stage.wrapper.compute_embedding.assert_called_once()
                stage.wrapper.estimate_age.assert_called_once()
                mock_match.assert_called_once_with(embedding)
                mock_store.assert_called_once()
                # Check store_faces arguments
                call_args = mock_store.call_args
                assert call_args[0][0] == 123
                faces = call_args[0][1]
                assert len(faces) == 1
                face = faces[0]
                assert face["photo_id"] == 123
                assert face["person_id"] is None
                assert np.allclose(face["embedding"], embedding)
                assert face["age_estimate"] == age_mean
                expected_age_std = max(age_std * stage.config.face.age_confidence_scale, 1.0)
                assert face["age_std"] == expected_age_std
                assert face["confidence"] == confidence
                assert face["bbox_x1"] == bbox[0]
                assert face["bbox_y1"] == bbox[1]
                assert face["bbox_x2"] == bbox[2]
                assert face["bbox_y2"] == bbox[3]

    def test_match_person_no_embeddings(self, stage):
        """Test person matching when persons table has no embeddings."""
        with patch.object(stage, "_get_known_persons_with_embeddings", return_value=[]):
            result = stage._match_person(np.random.randn(512))
            assert result is None

    def test_cosine_similarity(self, stage):
        """Test cosine similarity calculation."""
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = stage._cosine_similarity(a, b)
        assert np.allclose(sim, 0.0)

        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        sim = stage._cosine_similarity(a, b)
        assert np.allclose(sim, 1.0)

        a = np.array([1.0, 1.0])
        b = np.array([1.0, 1.0])
        sim = stage._cosine_similarity(a, b)
        assert np.allclose(sim, 1.0)

    def test_crop_face_with_margin(self, stage):
        """Test cropping face with margin."""
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        bbox = (100, 100, 200, 200)
        cropped = stage._crop_face_with_margin(image, bbox, margin_ratio=0.1)
        # Expected margins: width=100, height=100, margin_x=10, margin_y=10
        # Expanded bbox: (90, 90, 210, 210) clamped to image bounds (0,0,640,480)
        # Actually y2 becomes 210 within height 480, x2 210 within width 640
        # So crop shape should be (120, 120, 3)
        assert cropped.shape == (120, 120, 3)
        # Ensure the crop corresponds to the correct region
        assert np.array_equal(cropped, image[90:210, 90:210])
