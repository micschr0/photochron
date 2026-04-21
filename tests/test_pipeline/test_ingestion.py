"""
Unit tests for the IngestionStage.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image

from photochron.config import Config
from photochron.pipeline.stages.ingestion import IngestionStage


class TestIngestionStage:
    """Test suite for IngestionStage."""

    @pytest.fixture
    def stage(self):
        """Create an IngestionStage instance with mocked config."""
        stage = IngestionStage()
        stage.config = Mock(spec=Config)
        stage.config.input_dir = "/fake/input"
        stage.config.cache_dir = "/fake/cache"
        return stage

    @pytest.fixture
    def temp_image_dir(self):
        """Create a temporary directory with test images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple JPEG image
            img_path = Path(tmpdir) / "test.jpg"
            img = Image.new("RGB", (100, 100), color="red")
            img.save(img_path, format="JPEG")

            # Create a simple PNG image
            img_path2 = Path(tmpdir) / "test.png"
            img2 = Image.new("RGB", (200, 150), color="blue")
            img2.save(img_path2, format="PNG")

            yield tmpdir

    def test_name_property(self, stage):
        """Test that stage has correct name."""
        assert stage.name == "ingestion"

    def test_dependencies_property(self, stage):
        """Test that stage has no dependencies."""
        assert stage.dependencies == []

    def test_scan_image_files(self, stage, temp_image_dir):
        """Test scanning for image files."""
        stage.config.input_dir = temp_image_dir
        image_files = stage._scan_image_files(Path(temp_image_dir))

        # Should find both jpg and png
        assert len(image_files) == 2
        extensions = {f.suffix.lower() for f in image_files}
        assert {".jpg", ".png"} == extensions

    def test_scan_image_files_empty(self, stage, tmp_path):
        """Test scanning empty directory."""
        stage.config.input_dir = str(tmp_path)
        image_files = stage._scan_image_files(tmp_path)
        assert image_files == []

    def test_scan_image_files_nonexistent(self, stage):
        """Test scanning non-existent directory raises error."""
        with pytest.raises(FileNotFoundError):
            stage._scan_image_files(Path("/nonexistent/path"))

    @patch("photochron.pipeline.stages.ingestion.Image.open")
    def test_process_image_basic(self, mock_image_open, stage, tmp_path):
        """Test basic image processing."""
        # Mock image
        mock_img = Mock()
        mock_img.mode = "RGB"
        mock_img.size = (800, 600)
        mock_img.format = "JPEG"
        mock_img.convert.return_value = mock_img
        mock_image_open.return_value.__enter__.return_value = mock_img

        # Mock imagehash
        with patch("photochron.pipeline.stages.ingestion.imagehash") as mock_imagehash:
            mock_phash = Mock()
            mock_phash.__str__.return_value = "abcdef1234567890"
            mock_imagehash.phash.return_value = mock_phash

            # Mock other dependencies
            with patch.object(stage, "_compute_content_hash", return_value="md5hash"):
                with patch.object(
                    stage,
                    "_extract_exif_metadata",
                    return_value={"datetime": "2023-01-01T12:00:00"},
                ):
                    with patch.object(stage, "_create_downsampled_image", return_value=None):
                        with patch.object(stage, "_store_photo_metadata") as mock_store:
                            stage._process_image(Path("/fake/image.jpg"), Path(tmp_path), "run123")

                            # Verify image was opened
                            mock_image_open.assert_called_once()
                            # Verify perceptual hash was computed
                            mock_imagehash.phash.assert_called_once_with(mock_img)
                            # Verify metadata was stored
                            mock_store.assert_called_once()

    def test_create_downsampled_image_large(self, stage, tmp_path):
        """Test downsampling of large image."""
        # Create a large mock image
        mock_img = Mock()
        mock_img.size = (4000, 3000)  # Large image
        mock_img.format = "JPEG"

        downsampled_path = stage._create_downsampled_image(mock_img, "hash123", Path(tmp_path), "JPEG")

        # Should return a path
        assert downsampled_path is not None
        assert downsampled_path.exists()

        # Check file is JPEG
        assert downsampled_path.suffix == ".jpg"

        # Clean up
        downsampled_path.unlink()

    def test_create_downsampled_image_small(self, stage, tmp_path):
        """Test that small images are not downsampled."""
        # Create a small mock image
        mock_img = Mock()
        mock_img.size = (800, 600)  # Smaller than max_size
        mock_img.format = "JPEG"

        with patch.object(mock_img, "resize") as mock_resize:
            downsampled_path = stage._create_downsampled_image(mock_img, "hash123", Path(tmp_path), "JPEG")

            # Should return None (no downsampling needed)
            assert downsampled_path is None
            # resize should not be called
            mock_resize.assert_not_called()

    @patch("photochron.pipeline.stages.ingestion.piexif")
    def test_extract_exif_metadata_with_piexif(self, mock_piexif, stage):
        """Test EXIF extraction using piexif."""
        # Mock piexif data
        mock_exif_dict = {
            "Exif": {36867: b"2023:01:01 12:00:00"},  # DateTimeOriginal
            "0th": {271: b"Canon", 272: b"EOS R5"},
            "GPS": {},
        }
        mock_piexif.load.return_value = mock_exif_dict

        exif_data = stage._extract_exif_metadata(Path("/fake/image.jpg"))

        assert exif_data["datetime"] == "2023-01-01T12:00:00"
        assert exif_data["make"] == "Canon"
        assert exif_data["model"] == "EOS R5"

    @patch("photochron.pipeline.stages.ingestion.piexif")
    def test_extract_exif_metadata_fallback(self, mock_piexif, stage, tmp_path):
        """Test EXIF extraction fallback to file mtime when piexif fails."""
        # Make piexif raise an exception
        mock_piexif.load.side_effect = Exception("No EXIF")

        # Create a temporary file to get mtime
        test_file = tmp_path / "test.jpg"
        test_file.touch()

        with patch("photochron.pipeline.stages.ingestion.Image.open") as mock_image_open:
            mock_img = Mock()
            mock_img.getexif.return_value = {}
            mock_image_open.return_value.__enter__.return_value = mock_img

            exif_data = stage._extract_exif_metadata(test_file)

            # Should have datetime from file mtime
            assert "datetime" in exif_data
            assert exif_data["datetime_source"] == "file_mtime"

    def test_compute_content_hash(self, stage, tmp_path):
        """Test MD5 content hash computation."""
        # Create a test file with known content
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello, World!")

        content_hash = stage._compute_content_hash(test_file)

        # MD5 of "Hello, World!" (without newline)
        expected = "65a8e27d8879283831b664bd8b7f0ad4"
        assert content_hash == expected

    @patch("photochron.pipeline.stages.ingestion.get_store")
    def test_store_photo_metadata(self, mock_get_store, stage):
        """Test storing photo metadata in database."""
        mock_store = Mock()
        mock_conn = Mock()
        mock_store.transaction.return_value.__enter__.return_value = mock_conn
        mock_get_store.return_value = mock_store

        stage._store_photo_metadata(
            content_hash="md5hash",
            file_path="/fake/image.jpg",
            downsampled_path="/fake/downsampled.jpg",
            perceptual_hash="phash123",
            width=800,
            height=600,
            format_name="JPEG",
            exif_datetime="2023-01-01T12:00:00",
            make="Canon",
            model="EOS R5",
            run_id="run123",
        )

        # Verify database insert was called
        mock_conn.execute.assert_called_once()
        call_args = mock_conn.execute.call_args[0]
        assert "INSERT OR REPLACE INTO photos" in call_args[0]
        assert call_args[1][0] == "md5hash"  # content_hash

    @pytest.mark.integration
    def test_integration_full_ingestion(self, database_store, temp_image_dir):
        """Integration test: run full ingestion on mock image directory."""
        # Create a real IngestionStage with proper config
        from photochron.config import Config, ConfigModels, ConfigPaths, ConfigPipeline

        config = Config(
            version="1.0",
            paths=ConfigPaths(
                cache_dir=temp_image_dir + "/cache",
                thumbs_dir=temp_image_dir + "/cache/thumbs",
                output_dir=temp_image_dir + "/output",
                input_dir=temp_image_dir,
            ),
            models=ConfigModels(
                insightface_version="test",
                ollama_model="test",
                fallback_model="test",
                max_image_size=1024,
            ),
            pipeline=ConfigPipeline(
                enable_face_detection=True,
                enable_context_analysis=True,
                enable_anchor_matching=True,
                enable_ranking=True,
                enable_output=True,
            ),
        )

        # Patch get_config to return our config
        with patch("photochron.pipeline.stages.ingestion.get_config", return_value=config):
            # Patch get_store to return our test database store
            with patch(
                "photochron.pipeline.stages.ingestion.get_store",
                return_value=database_store,
            ):
                stage = IngestionStage()

                # Run the stage
                run_id = "test_integration_run"
                config_hash = "test_hash"

                # This should process the images in temp_image_dir
                stage.run(run_id, config_hash)

                # Verify that photos were inserted into database
                with database_store.transaction() as conn:
                    cursor = conn.execute("SELECT COUNT(*) as count FROM photos")
                    row = cursor.fetchone()
                    assert row["count"] == 2  # Should have 2 images (jpg and png)

                    # Verify at least one photo has perceptual hash
                    cursor = conn.execute("SELECT perceptual_hash FROM photos WHERE perceptual_hash IS NOT NULL")
                    hashes = cursor.fetchall()
                    assert len(hashes) > 0
