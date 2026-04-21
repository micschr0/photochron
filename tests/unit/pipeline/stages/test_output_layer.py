"""Tests for the output layer pipeline stage."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image

from photochron.models import PhotoCreate, RankingCreate
from photochron.pipeline.stages.output_layer import OutputLayerStage


def _make_jpeg(path: Path, color: tuple = (255, 0, 0)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 16), color).save(path, "JPEG")


def _seed_run(helper, run_id: str) -> None:
    helper.conn.execute(
        "INSERT INTO pipeline_runs (run_id, schema_version, config_hash, start_time, status) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'running')",
        (run_id, 1, "hash"),
    )


def _seed_photo_and_ranking(helper, file_path: Path, sort_rank: int, year: int | None, confidence: float) -> int:
    photo_id = helper.insert_photo(
        PhotoCreate(
            content_hash=f"hash_{sort_rank}",
            file_path=str(file_path),
            downsample_path=None,
        )
    )
    helper.insert_ranking(
        RankingCreate(
            photo_id=photo_id,
            sort_rank=sort_rank,
            estimated_year=year,
            estimated_month=None,
            confidence=confidence,
            review_needed=confidence < 0.5,
            ranking_json=json.dumps({"file_path": str(file_path), "year": year, "signals": {}}),
        )
    )
    return photo_id


@pytest.fixture
def patched_store(database_store):
    with (
        patch(
            "photochron.pipeline.stages.output_layer.get_store",
            return_value=database_store,
        ),
        patch("photochron.pipeline.get_store", return_value=database_store),
    ):
        yield database_store


def test_output_layer_writes_renamed_and_enriched_copies(patched_store, mock_config, tmp_path):
    src = tmp_path / "input" / "IMG_042.jpg"
    _make_jpeg(src)

    mock_config.paths.output_dir = str(tmp_path / "out")

    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        _seed_photo_and_ranking(helper, src, sort_rank=0, year=1985, confidence=0.8)
        _seed_run(helper, "run_out")

    with patch("photochron.pipeline.stages.output_layer.get_config", return_value=mock_config):
        OutputLayerStage().run("run_out", "cfg")

    renamed_dir = tmp_path / "out" / "renamed"
    enriched_dir = tmp_path / "out" / "exif_enriched"

    assert renamed_dir.exists()
    renamed_files = list(renamed_dir.iterdir())
    assert any("0000_1985-est_IMG_042.jpg" in f.name for f in renamed_files)

    assert (enriched_dir / "IMG_042.jpg").exists()

    report = json.loads((tmp_path / "out" / "photochron_report.json").read_text())
    assert report["summary"]["total_photos"] == 1
    assert report["summary"]["year_range"] == [1985, 1985]

    timeline = (tmp_path / "out" / "photochron_timeline.csv").read_text()
    assert "IMG_042.jpg" in timeline


def test_output_layer_never_writes_to_original_file(patched_store, mock_config, tmp_path):
    src = tmp_path / "input" / "IMG_042.jpg"
    _make_jpeg(src)
    original_bytes = src.read_bytes()

    mock_config.paths.output_dir = str(tmp_path / "out")

    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        _seed_photo_and_ranking(helper, src, sort_rank=0, year=1985, confidence=0.8)
        _seed_run(helper, "run_nondestruct")

    with patch("photochron.pipeline.stages.output_layer.get_config", return_value=mock_config):
        OutputLayerStage().run("run_nondestruct", "cfg")

    assert src.read_bytes() == original_bytes


def test_output_layer_handles_missing_source_files(patched_store, mock_config, tmp_path):
    missing = tmp_path / "input" / "gone.jpg"
    mock_config.paths.output_dir = str(tmp_path / "out")

    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        _seed_photo_and_ranking(helper, missing, sort_rank=0, year=1985, confidence=0.8)
        _seed_run(helper, "run_missing")

    with patch("photochron.pipeline.stages.output_layer.get_config", return_value=mock_config):
        OutputLayerStage().run("run_missing", "cfg")

    report = json.loads((tmp_path / "out" / "photochron_report.json").read_text())
    assert report["summary"]["total_photos"] == 0


def test_output_layer_with_no_rankings_is_noop(patched_store, mock_config, tmp_path):
    mock_config.paths.output_dir = str(tmp_path / "out")
    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        _seed_run(helper, "run_empty")

    with patch("photochron.pipeline.stages.output_layer.get_config", return_value=mock_config):
        OutputLayerStage().run("run_empty", "cfg")

    # no output dir should be created since early-return happens before mkdir
    assert not (tmp_path / "out").exists()
