"""Unit tests for ``IngestionStage`` helper methods.

Exercises ``_scan_image_files``, ``_extract_exif_metadata`` fallback paths,
``_compute_content_hash``, ``_create_downsampled_image``, ``_parse_gps_coordinates``
and ``_convert_gps_coordinate`` without bringing up the full pipeline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from photochron.pipeline.stages.ingestion import IngestionStage


@pytest.fixture
def stage() -> IngestionStage:
    s = IngestionStage()
    # Replace config with a controllable mock so we don't depend on the real
    # config file's exact contents.
    s.config = MagicMock()
    s.config.ingestion = MagicMock()
    s.config.ingestion.supported_formats = [".jpg", ".png"]
    s.config.ingestion.max_downsample_size = 256
    s.config.ingestion.extract_gps = False
    s.config.ingestion.workers = 1
    s.supported_extensions = {".jpg", ".png"}
    return s


@pytest.fixture
def sample_jpeg(tmp_path: Path) -> Path:
    img = Image.new("RGB", (1024, 768), color="white")
    p = tmp_path / "sample.jpg"
    img.save(p, format="JPEG")
    return p


# ---------------------------------------------------------------------------
# _scan_image_files
# ---------------------------------------------------------------------------


def test_scan_image_files_returns_sorted_unique(stage: IngestionStage, tmp_path: Path) -> None:
    (tmp_path / "b.jpg").touch()
    (tmp_path / "a.jpg").touch()
    (tmp_path / "c.png").touch()
    (tmp_path / "ignored.txt").touch()
    result = stage._scan_image_files(tmp_path)
    names = [p.name for p in result]
    assert names == ["a.jpg", "b.jpg", "c.png"]


def test_scan_image_files_missing_dir_raises(stage: IngestionStage, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        stage._scan_image_files(tmp_path / "missing")


def test_scan_image_files_is_case_insensitive(stage: IngestionStage, tmp_path: Path) -> None:
    (tmp_path / "PHOTO.JPG").touch()
    result = stage._scan_image_files(tmp_path)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# _compute_content_hash
# ---------------------------------------------------------------------------


def test_compute_content_hash_deterministic(stage: IngestionStage, tmp_path: Path) -> None:
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello world")
    h1 = stage._compute_content_hash(p)
    h2 = stage._compute_content_hash(p)
    assert h1 == h2
    assert len(h1) == 32  # MD5 hex


def test_compute_content_hash_differs_for_different_files(stage: IngestionStage, tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    assert stage._compute_content_hash(a) != stage._compute_content_hash(b)


def test_compute_content_hash_streams_large_file(stage: IngestionStage, tmp_path: Path) -> None:
    """Streams in 8 KiB chunks; verify a >chunk-sized file hashes correctly."""
    p = tmp_path / "big.bin"
    p.write_bytes(b"x" * (8192 * 3 + 17))
    h = stage._compute_content_hash(p)
    assert len(h) == 32


# ---------------------------------------------------------------------------
# _create_downsampled_image
# ---------------------------------------------------------------------------


def test_create_downsampled_returns_none_for_small_images(stage: IngestionStage, tmp_path: Path) -> None:
    img = Image.new("RGB", (100, 80), "red")
    out = stage._create_downsampled_image(img, "abc123", tmp_path, "JPEG")
    assert out is None  # Already smaller than max_downsample_size=256


def test_create_downsampled_writes_jpeg_for_landscape(stage: IngestionStage, tmp_path: Path) -> None:
    img = Image.new("RGB", (1024, 512), "blue")
    out = stage._create_downsampled_image(img, "ph123", tmp_path, "JPEG")
    assert out is not None and out.exists()
    assert out.suffix == ".jpg"
    with Image.open(out) as reopened:
        assert max(reopened.size) == 256


def test_create_downsampled_writes_png_for_non_jpeg_source(stage: IngestionStage, tmp_path: Path) -> None:
    img = Image.new("RGB", (512, 1024), "green")
    out = stage._create_downsampled_image(img, "ph456", tmp_path, "PNG")
    assert out is not None and out.exists()
    assert out.suffix == ".png"
    with Image.open(out) as reopened:
        assert max(reopened.size) == 256


# ---------------------------------------------------------------------------
# _extract_exif_metadata
# ---------------------------------------------------------------------------


def test_extract_exif_falls_back_to_mtime_when_no_exif(stage: IngestionStage, sample_jpeg: Path) -> None:
    """Pillow-created JPEG has no DateTimeOriginal → fallback to file mtime."""
    result = stage._extract_exif_metadata(sample_jpeg)
    assert "datetime" in result
    # Source flag is set when we fell back.
    assert result.get("datetime_source") == "file_mtime"


def test_extract_exif_with_piexif_failure_uses_pillow_path(stage: IngestionStage, sample_jpeg: Path) -> None:
    """piexif raising should not surface; the Pillow path still runs."""
    from photochron.pipeline.stages import ingestion as ing

    with patch.object(ing.piexif, "load", side_effect=ing.InvalidImageDataError("bad")):
        result = stage._extract_exif_metadata(sample_jpeg)

    # Falls all the way through to the mtime fallback because Pillow EXIF is
    # empty for a freshly-generated JPEG.
    assert "datetime" in result


# ---------------------------------------------------------------------------
# _parse_gps_coordinates / _convert_gps_coordinate
# ---------------------------------------------------------------------------


def test_convert_gps_coordinate_handles_rational_tuples(stage: IngestionStage) -> None:
    # 48° 30' 30" N
    coord = ((48, 1), (30, 1), (30, 1))
    decimal = stage._convert_gps_coordinate(coord, b"N")
    assert decimal == pytest.approx(48 + 30 / 60 + 30 / 3600)


def test_convert_gps_coordinate_applies_south_negation(stage: IngestionStage) -> None:
    coord = ((10, 1), (0, 1), (0, 1))
    assert stage._convert_gps_coordinate(coord, b"S") == pytest.approx(-10.0)


def test_convert_gps_coordinate_west_negates(stage: IngestionStage) -> None:
    coord = ((5, 1), (0, 1), (0, 1))
    assert stage._convert_gps_coordinate(coord, b"W") == pytest.approx(-5.0)


def test_parse_gps_coordinates_returns_none_on_missing_keys(stage: IngestionStage) -> None:
    lat, lon = stage._parse_gps_coordinates({})
    assert lat is None and lon is None


def test_parse_gps_coordinates_returns_none_on_malformed_data(stage: IngestionStage) -> None:
    """Garbage values should be tolerated — return ``(None, None)``."""
    import piexif

    bad = {
        piexif.GPSIFD.GPSLatitude: "not a tuple",
        piexif.GPSIFD.GPSLatitudeRef: b"N",
        piexif.GPSIFD.GPSLongitude: "also bad",
        piexif.GPSIFD.GPSLongitudeRef: b"E",
    }
    lat, lon = stage._parse_gps_coordinates(bad)
    assert lat is None and lon is None
