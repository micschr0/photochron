"""Unit tests for ``InsightFaceWrapper`` covering load/detect/embed/age paths.

The actual ``insightface.app.FaceAnalysis`` and ``onnxruntime`` modules are
mocked — these tests only verify the wrapper's glue (threshold filtering,
lazy loading, normalisation, error propagation).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from photochron.face.insightface_wrapper import InsightFaceWrapper


def _make_detection(bbox: tuple[int, int, int, int], score: float, age: float = 30.0):
    """Build a mock InsightFace detection record."""
    return SimpleNamespace(
        bbox=np.array(bbox, dtype=np.float32),
        det_score=score,
        embedding=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        age=age,
    )


def test_use_gpu_true_maps_to_cuda_backend() -> None:
    w = InsightFaceWrapper(use_gpu=True)
    assert w.backend == "cuda"


def test_explicit_backend_overrides_use_gpu_flag() -> None:
    w = InsightFaceWrapper(backend="coreml", use_gpu=True)
    assert w.backend == "coreml"


def test_default_backend_is_auto_and_use_gpu_default_false() -> None:
    w = InsightFaceWrapper()
    assert w.backend == "auto"
    # `auto` doesn't claim GPU until resolution maps it.
    assert isinstance(w.use_gpu, bool)


def test_load_model_raises_when_insightface_missing() -> None:
    w = InsightFaceWrapper()
    # Simulate `import insightface.app` failing.
    with patch.dict(sys.modules, {"insightface.app": None}):
        with pytest.raises(ImportError, match="InsightFace not installed"):
            w.load_model()


def test_load_model_is_cached(_=None) -> None:
    w = InsightFaceWrapper()
    fake_model = MagicMock()
    fake_face_analysis = MagicMock(return_value=fake_model)
    fake_module = MagicMock(FaceAnalysis=fake_face_analysis)
    with patch.dict(sys.modules, {"insightface.app": fake_module}):
        w.load_model()
        w.load_model()  # second call must early-return
    fake_face_analysis.assert_called_once()
    fake_model.prepare.assert_called_once()


def test_detect_faces_filters_low_confidence() -> None:
    w = InsightFaceWrapper(detection_threshold=0.5)
    fake_model = MagicMock()
    fake_model.get.return_value = [
        _make_detection((0, 0, 10, 10), 0.9),
        _make_detection((20, 20, 30, 30), 0.4),  # below threshold
        _make_detection((40, 40, 50, 50), 0.7),
    ]
    w._model = fake_model

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    results = w.detect_faces(image)
    assert len(results) == 2
    # Returned in detection order, with bbox as ints + confidence as float.
    assert results[0][0] == (0, 0, 10, 10)
    assert results[0][1] == pytest.approx(0.9)


def test_detect_faces_lazy_loads_when_model_is_none() -> None:
    w = InsightFaceWrapper()
    image = np.zeros((10, 10, 3), dtype=np.uint8)

    def fake_load() -> None:
        fake_model = MagicMock()
        fake_model.get.return_value = []
        w._model = fake_model

    with patch.object(w, "load_model", side_effect=fake_load) as mock_load:
        w.detect_faces(image)
    mock_load.assert_called_once()


def test_compute_embedding_returns_normalised_vector() -> None:
    w = InsightFaceWrapper()
    fake_model = MagicMock()
    fake_model.get.return_value = [_make_detection((0, 0, 10, 10), 0.9)]
    w._model = fake_model

    emb = w.compute_embedding(np.zeros((20, 20, 3), dtype=np.uint8))
    # Unit length.
    assert np.linalg.norm(emb) == pytest.approx(1.0)


def test_compute_embedding_raises_when_no_face_detected() -> None:
    w = InsightFaceWrapper()
    fake_model = MagicMock()
    fake_model.get.return_value = []
    w._model = fake_model
    with pytest.raises(ValueError, match="No face found"):
        w.compute_embedding(np.zeros((20, 20, 3), dtype=np.uint8))


def test_estimate_age_returns_mean_and_floored_std() -> None:
    w = InsightFaceWrapper()
    fake_model = MagicMock()
    fake_model.get.return_value = [_make_detection((0, 0, 10, 10), 0.9, age=25.0)]
    w._model = fake_model
    mean, std = w.estimate_age(np.zeros((20, 20, 3), dtype=np.uint8))
    assert mean == 25.0
    # 25 * 0.1 = 2.5, above the 1.0 floor.
    assert std == pytest.approx(2.5)


def test_estimate_age_floors_std_at_one() -> None:
    w = InsightFaceWrapper()
    fake_model = MagicMock()
    fake_model.get.return_value = [_make_detection((0, 0, 10, 10), 0.9, age=5.0)]
    w._model = fake_model
    _, std = w.estimate_age(np.zeros((20, 20, 3), dtype=np.uint8))
    # 5 * 0.1 = 0.5 → floored to 1.0.
    assert std == 1.0


def test_estimate_age_raises_when_no_face_detected() -> None:
    w = InsightFaceWrapper()
    fake_model = MagicMock()
    fake_model.get.return_value = []
    w._model = fake_model
    with pytest.raises(ValueError):
        w.estimate_age(np.zeros((20, 20, 3), dtype=np.uint8))


def test_batch_detect_iterates_per_image() -> None:
    w = InsightFaceWrapper(detection_threshold=0.5)
    fake_model = MagicMock()
    # 2 images, each gets one detection above threshold.
    fake_model.get.side_effect = [
        [_make_detection((0, 0, 10, 10), 0.9)],
        [_make_detection((5, 5, 15, 15), 0.8)],
    ]
    w._model = fake_model

    out = w.batch_detect(
        [
            np.zeros((10, 10, 3), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.uint8),
        ]
    )
    assert len(out) == 2
    assert len(out[0]) == 1
    assert len(out[1]) == 1


def test_unload_clears_model_reference() -> None:
    w = InsightFaceWrapper()
    w._model = MagicMock()
    w.unload()
    assert w._model is None
