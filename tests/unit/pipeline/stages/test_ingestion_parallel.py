"""
Unit tests for the parallel ingestion code path.

Validates that ``IngestionStage.run`` uses a ThreadPoolExecutor when
``config.ingestion.workers > 1`` and still processes every file (plus records
a progress count). DB / PIL / piexif are stubbed – this test only exercises
the control flow in ``run()``.
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from photochron.pipeline.stages.ingestion import IngestionStage


@pytest.fixture
def image_paths(tmp_path: Path) -> list[Path]:
    """Create 8 zero-byte JPG placeholders; _process_image is stubbed anyway."""
    files = []
    for i in range(8):
        f = tmp_path / f"photo_{i:03d}.jpg"
        f.touch()
        files.append(f)
    return files


def _make_stage(workers: int, input_dir: Path) -> IngestionStage:
    stage = IngestionStage()
    stage.config = Mock()
    stage.config.input_dir = str(input_dir)
    stage.config.cache_dir = str(input_dir / ".photochron")
    stage.config.ingestion = Mock()
    stage.config.ingestion.supported_formats = [".jpg"]
    stage.config.ingestion.workers = workers
    stage.supported_extensions = {".jpg"}
    return stage


class TestIngestionWorkerFanOut:
    def test_workers_one_processes_sequentially(self, image_paths, tmp_path):
        stage = _make_stage(workers=1, input_dir=tmp_path)

        processed: list[Path] = []

        def fake_process(file_path, downsampled_dir, run_id):  # noqa: ARG001
            processed.append(file_path)

        with (
            patch.object(IngestionStage, "_process_image", side_effect=fake_process),
            patch.object(IngestionStage, "mark_complete") as mock_complete,
        ):
            stage.run(run_id="run_abc", config_hash="deadbeef")

        # Every file processed exactly once, in scan order (single thread).
        assert sorted(p.name for p in processed) == [f.name for f in image_paths]
        mock_complete.assert_called_once()
        _, kwargs = mock_complete.call_args
        assert kwargs["photos_processed"] == len(image_paths)

    def test_workers_many_fan_out_all_files(self, image_paths, tmp_path):
        stage = _make_stage(workers=4, input_dir=tmp_path)

        seen_threads: set[str] = set()
        processed: list[Path] = []
        lock = threading.Lock()

        def fake_process(file_path, downsampled_dir, run_id):  # noqa: ARG001
            with lock:
                seen_threads.add(threading.current_thread().name)
                processed.append(file_path)

        with (
            patch.object(IngestionStage, "_process_image", side_effect=fake_process),
            patch.object(IngestionStage, "mark_complete") as mock_complete,
        ):
            stage.run(run_id="run_xyz", config_hash="cafef00d")

        assert len(processed) == len(image_paths)
        assert {p.name for p in processed} == {f.name for f in image_paths}
        # Execution went through the ThreadPoolExecutor path (worker thread
        # names have the `ingestion_N` prefix). We can't reliably assert
        # multiple threads ran without synchronization primitives – the stub
        # is fast enough that one worker may drain the queue.
        assert seen_threads, "no threads observed"
        assert all(name.startswith("ingestion_") for name in seen_threads)
        assert not any(name == "MainThread" for name in seen_threads)
        mock_complete.assert_called_once()

    def test_per_file_failure_does_not_abort_batch(self, image_paths, tmp_path):
        stage = _make_stage(workers=4, input_dir=tmp_path)

        def fake_process(file_path, downsampled_dir, run_id):  # noqa: ARG001
            # Fail on the first file only.
            if file_path.name == "photo_000.jpg":
                raise OSError("simulated decode failure")

        with (
            patch.object(IngestionStage, "_process_image", side_effect=fake_process),
            patch.object(IngestionStage, "mark_complete") as mock_complete,
        ):
            stage.run(run_id="run_err", config_hash="feedface")

        # 7 out of 8 succeed; the stage still completes.
        _, kwargs = mock_complete.call_args
        assert kwargs["photos_processed"] == 7
