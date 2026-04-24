"""
Unit tests for backend → ONNX-Runtime-provider resolution.

These tests lock in the routing logic in
:func:`photochron.face.insightface_wrapper._resolve_providers` so that Apple
Silicon users get the CoreML EP without config edits, and everyone else (or
misconfigured hosts) fall back to CPU with a warning instead of crashing.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from photochron.face import insightface_wrapper as wrapper_mod
from photochron.face.insightface_wrapper import InsightFaceWrapper, _resolve_providers


@pytest.fixture
def apple_silicon():
    """Pretend the host is an Apple-Silicon Mac."""
    with (
        patch.object(wrapper_mod.platform, "system", return_value="Darwin"),
        patch.object(wrapper_mod.platform, "machine", return_value="arm64"),
    ):
        yield


@pytest.fixture
def intel_linux():
    """Pretend the host is x86_64 Linux (representative non-Apple host)."""
    with (
        patch.object(wrapper_mod.platform, "system", return_value="Linux"),
        patch.object(wrapper_mod.platform, "machine", return_value="x86_64"),
    ):
        yield


def _available(providers):
    """Stub `onnxruntime.get_available_providers()` to return *providers*."""
    import onnxruntime as ort

    return patch.object(ort, "get_available_providers", return_value=list(providers))


class TestAutoBackend:
    def test_auto_on_apple_silicon_with_coreml_available(self, apple_silicon):
        with _available(["CoreMLExecutionProvider", "CPUExecutionProvider"]):
            providers, options = _resolve_providers("auto")
        assert providers[0] == "CoreMLExecutionProvider"
        assert providers[-1] == "CPUExecutionProvider"
        # Options aligned 1:1 with providers.
        assert len(options) == len(providers)
        # "ALL" (the onnxruntime default) lets the runtime partition across
        # ANE / GPU / CPU per op instead of pinning to ANE only.
        assert options[0]["MLComputeUnits"] == "ALL"
        assert options[0]["ModelFormat"] == "MLProgram"
        assert options[0]["RequireStaticInputShapes"] == "1"

    def test_auto_on_apple_silicon_without_coreml_falls_back_to_cpu(self, apple_silicon):
        # Pretend the onnxruntime build does not expose CoreML EP.
        with _available(["CPUExecutionProvider"]):
            providers, _ = _resolve_providers("auto")
        assert providers == ["CPUExecutionProvider"]

    def test_auto_on_non_apple_picks_cpu(self, intel_linux):
        with _available(["CUDAExecutionProvider", "CPUExecutionProvider"]):
            providers, _ = _resolve_providers("auto")
        assert providers == ["CPUExecutionProvider"]


class TestExplicitBackends:
    def test_cpu_explicit_always_cpu(self, apple_silicon):
        with _available(["CoreMLExecutionProvider", "CPUExecutionProvider"]):
            providers, _ = _resolve_providers("cpu")
        assert providers == ["CPUExecutionProvider"]

    def test_coreml_on_non_apple_falls_back_with_warning(self, intel_linux, caplog):
        with _available(["CPUExecutionProvider"]):
            providers, _ = _resolve_providers("coreml")
        assert providers == ["CPUExecutionProvider"]

    def test_coreml_missing_provider_falls_back(self, apple_silicon):
        # Apple Silicon host, but onnxruntime build lacks CoreML EP.
        with _available(["CPUExecutionProvider"]):
            providers, _ = _resolve_providers("coreml")
        assert providers == ["CPUExecutionProvider"]

    def test_cuda_missing_falls_back(self, intel_linux):
        with _available(["CPUExecutionProvider"]):
            providers, _ = _resolve_providers("cuda")
        assert providers == ["CPUExecutionProvider"]

    def test_cuda_available(self, intel_linux):
        with _available(["CUDAExecutionProvider", "CPUExecutionProvider"]):
            providers, opts = _resolve_providers("cuda")
        assert providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
        assert len(opts) == 2


class TestWrapperConstructor:
    def test_backend_cpu_stores_providers(self, intel_linux):
        with _available(["CPUExecutionProvider"]):
            w = InsightFaceWrapper(model_name="buffalo_l", backend="cpu")
        assert w.backend == "cpu"
        assert w._providers == ["CPUExecutionProvider"]
        assert w.use_gpu is False

    def test_legacy_use_gpu_true_migrates_to_cuda(self, intel_linux):
        with _available(["CUDAExecutionProvider", "CPUExecutionProvider"]):
            w = InsightFaceWrapper(model_name="buffalo_l", use_gpu=True)
        assert w.backend == "cuda"
        assert w._providers[0] == "CUDAExecutionProvider"

    def test_explicit_backend_wins_over_legacy_use_gpu(self, apple_silicon):
        # backend='coreml' takes precedence even if use_gpu=True is passed.
        with _available(["CoreMLExecutionProvider", "CPUExecutionProvider"]):
            w = InsightFaceWrapper(model_name="buffalo_l", backend="coreml", use_gpu=True)
        assert w.backend == "coreml"
        assert w._providers[0] == "CoreMLExecutionProvider"
