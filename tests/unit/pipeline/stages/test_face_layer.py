"""Unit tests for the face layer pipeline stage.

The InsightFaceWrapper is mocked end-to-end — these tests cover the stage's
glue logic (DB query, image load, person matching, batch loop, error
handling), not the ML inference itself.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from photochron.pipeline.stages.face_layer import FaceLayerStage
from photochron.store import DatabaseStore


@pytest.fixture
def jpeg_path(tmp_path: Path) -> Path:
    """Tiny real JPEG so ``_load_downsampled_image`` succeeds."""
    img = Image.new("RGB", (64, 64), color=(128, 64, 32))
    p = tmp_path / "thumb.jpg"
    img.save(p, format="JPEG")
    return p


@pytest.fixture
def store(tmp_path: Path):
    """Isolated DatabaseStore with the minimal face-layer schema."""
    s = DatabaseStore(db_path=tmp_path / "face.db")
    with s.transaction() as conn:
        conn.executescript(
            """
            CREATE TABLE photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content_hash TEXT,
                file_path TEXT,
                downsample_path TEXT
            );
            CREATE TABLE persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT,
                name TEXT,
                birthday TEXT
            );
            CREATE TABLE faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER,
                person_id INTEGER,
                embedding BLOB,
                age_estimate REAL,
                age_std REAL,
                confidence REAL,
                bbox_x1 REAL, bbox_y1 REAL, bbox_x2 REAL, bbox_y2 REAL
            );
            """
        )
    yield s
    s.close()


def _make_stage(matching_threshold: float = 0.6) -> FaceLayerStage:
    """Build a FaceLayerStage with InsightFaceWrapper construction patched out."""
    with patch("photochron.pipeline.stages.face_layer.InsightFaceWrapper") as MockWrapper:
        MockWrapper.return_value = MagicMock()
        stage = FaceLayerStage()
    # Replace the config with a controllable mock.
    stage.face_config = MagicMock()
    stage.face_config.batch_size = 1
    stage.face_config.matching_threshold = matching_threshold
    stage.face_config.age_confidence_scale = 1.0
    return stage


def test_name_and_dependencies() -> None:
    stage = _make_stage()
    assert stage.name == "face_layer"
    assert stage.dependencies == ["ingestion"]


def test_load_downsampled_image_returns_rgb_uint8(jpeg_path: Path) -> None:
    stage = _make_stage()
    arr = stage._load_downsampled_image(jpeg_path)
    assert arr.shape == (64, 64, 3)
    assert arr.dtype == np.uint8


def test_load_downsampled_image_missing_raises(tmp_path: Path) -> None:
    stage = _make_stage()
    with pytest.raises(FileNotFoundError):
        stage._load_downsampled_image(tmp_path / "does_not_exist.jpg")


def test_crop_face_with_margin_clamps_to_bounds() -> None:
    stage = _make_stage()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # bbox at the corner; margin would exceed bounds and must clamp.
    out = stage._crop_face_with_margin(image, (90, 90, 100, 100), margin_ratio=1.0)
    assert out.shape[0] > 0 and out.shape[1] > 0


def test_crop_face_with_invalid_bbox_raises() -> None:
    stage = _make_stage()
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        stage._crop_face_with_margin(image, (50, 50, 40, 40))  # x2 < x1


def test_cosine_similarity_zero_norm_returns_zero() -> None:
    assert FaceLayerStage._cosine_similarity(np.zeros(8), np.ones(8)) == 0.0


def test_cosine_similarity_identical_vectors_is_one() -> None:
    v = np.array([1.0, 2.0, 3.0])
    assert FaceLayerStage._cosine_similarity(v, v) == pytest.approx(1.0)


def test_get_photos_without_faces_returns_only_unprocessed(store: DatabaseStore) -> None:
    with store.transaction() as conn:
        conn.execute("INSERT INTO photos (id, file_path, downsample_path) VALUES (1, 'a', 'a.jpg')")
        conn.execute("INSERT INTO photos (id, file_path, downsample_path) VALUES (2, 'b', 'b.jpg')")
        conn.execute(
            "INSERT INTO faces (photo_id, confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2) VALUES (1, 0.9, 0, 0, 10, 10)"
        )
    stage = _make_stage()
    with patch("photochron.pipeline.stages.face_layer.get_store", return_value=store):
        rows = stage._get_photos_without_faces()
    assert [r["id"] for r in rows] == [2]


def test_match_person_returns_best_above_threshold(store: DatabaseStore) -> None:
    # Seed a person with embedding; persons table needs the column.
    with store.transaction() as conn:
        conn.execute("ALTER TABLE persons ADD COLUMN embedding BLOB")
        embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        conn.execute(
            "INSERT INTO persons (id, person_id, name, embedding) VALUES (1, 'p_a', 'Alice', ?)",
            (embedding.tobytes(),),
        )
    stage = _make_stage(matching_threshold=0.5)
    with patch("photochron.pipeline.stages.face_layer.get_store", return_value=store):
        # Aligned to stored vector → similarity 1.0, above threshold.
        match = stage._match_person(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert match == 1


def test_match_person_returns_none_below_threshold(store: DatabaseStore) -> None:
    with store.transaction() as conn:
        conn.execute("ALTER TABLE persons ADD COLUMN embedding BLOB")
        embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        conn.execute(
            "INSERT INTO persons (id, person_id, name, embedding) VALUES (1, 'p_a', 'Alice', ?)",
            (embedding.tobytes(),),
        )
    stage = _make_stage(matching_threshold=0.99)
    with patch("photochron.pipeline.stages.face_layer.get_store", return_value=store):
        # Orthogonal → similarity 0, below 0.99.
        match = stage._match_person(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert match is None


def test_match_person_returns_none_with_no_persons(store: DatabaseStore) -> None:
    stage = _make_stage()
    with patch("photochron.pipeline.stages.face_layer.get_store", return_value=store):
        assert stage._match_person(np.zeros(3, dtype=np.float32)) is None


def test_process_photo_no_detections_does_not_write(store: DatabaseStore, jpeg_path: Path) -> None:
    stage = _make_stage()
    stage.wrapper.detect_faces.return_value = []
    with patch("photochron.pipeline.stages.face_layer.get_store", return_value=store):
        stage._process_photo(photo_id=1, downsample_path=jpeg_path)
    # Nothing inserted.
    with store.transaction() as conn:
        cnt = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
    assert cnt == 0


def test_process_photo_stores_detected_faces(store: DatabaseStore, jpeg_path: Path) -> None:
    """End-to-end stage logic with mocked detector + embedder + age estimator."""
    with store.transaction() as conn:
        conn.execute(
            "INSERT INTO photos (id, file_path, downsample_path) VALUES (1, 'a', ?)",
            (str(jpeg_path),),
        )

    stage = _make_stage()
    stage.wrapper.detect_faces.return_value = [((10, 10, 40, 40), 0.95)]
    stage.wrapper.compute_embedding.return_value = np.ones(8, dtype=np.float32)
    stage.wrapper.estimate_age.return_value = (32.0, 3.0)

    with patch("photochron.pipeline.stages.face_layer.get_store", return_value=store):
        stage._process_photo(photo_id=1, downsample_path=jpeg_path)

    with store.transaction() as conn:
        row = conn.execute(
            "SELECT photo_id, age_estimate, age_std, confidence FROM faces WHERE photo_id = 1"
        ).fetchone()
    assert row is not None
    assert row["photo_id"] == 1
    assert row["age_estimate"] == 32.0
    # age_confidence_scale=1.0 and floor of 1.0 → 3.0.
    assert row["age_std"] == 3.0
    assert row["confidence"] == pytest.approx(0.95)


def test_process_photo_missing_downsample_logs_and_returns(store: DatabaseStore, tmp_path: Path) -> None:
    stage = _make_stage()
    # Should swallow FileNotFoundError silently.
    stage._process_photo(photo_id=99, downsample_path=tmp_path / "missing.jpg")


def test_process_photo_per_face_value_error_skipped(store: DatabaseStore, jpeg_path: Path) -> None:
    """If embedding computation raises ValueError, the loop continues."""
    stage = _make_stage()
    stage.wrapper.detect_faces.return_value = [
        ((10, 10, 30, 30), 0.9),
        ((20, 20, 40, 40), 0.8),
    ]
    # First raises, second succeeds.
    stage.wrapper.compute_embedding.side_effect = [
        ValueError("no face"),
        np.ones(8, dtype=np.float32),
    ]
    stage.wrapper.estimate_age.return_value = (25.0, 2.5)

    with patch("photochron.pipeline.stages.face_layer.get_store", return_value=store):
        stage._process_photo(photo_id=1, downsample_path=jpeg_path)

    with store.transaction() as conn:
        cnt = conn.execute("SELECT COUNT(*) FROM faces WHERE photo_id = 1").fetchone()[0]
    assert cnt == 1
