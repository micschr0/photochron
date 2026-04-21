"""Tests for output filename generator."""

from photochron.output import build_renamed_filename


def test_build_renamed_filename_basic():
    name = build_renamed_filename(5, 1987, "IMG_042.jpg")
    assert name == "0005_1987-est_IMG_042.jpg"


def test_build_renamed_filename_unknown_year():
    name = build_renamed_filename(0, None, "weihnachten.jpeg")
    assert name == "0000_unknown-est_weihnachten.jpeg"


def test_build_renamed_filename_strips_directory_components():
    name = build_renamed_filename(12, 2001, "/some/dir/hello world.jpg")
    assert name == "0012_2001-est_hello world.jpg"


def test_build_renamed_filename_sanitizes_invalid_chars():
    name = build_renamed_filename(1, 2001, "ba:d?name.jpg")
    assert ":" not in name
    assert "?" not in name
    assert name.endswith(".jpg")
