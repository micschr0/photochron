"""Unit tests for environment-variable overrides in ``photochron.config``.

The PHOTOCHRON_<SECTION>_<KEY> mechanism lets ops override individual config
fields without editing config.yaml. These tests pin the conversion rules
(int / float / bool / str) and the unknown-section guard.
"""

from __future__ import annotations

from typing import Any

import pytest

from photochron.config import _apply_env_overrides


def _apply(env: dict[str, str], base: dict[str, Any] | None = None) -> dict[str, Any]:
    """Helper: apply env vars to a fresh config dict via monkeypatch-style."""
    import os

    base = base or {}
    # Stash and clear PHOTOCHRON_ vars so the test fully controls the env.
    saved = {k: v for k, v in os.environ.items() if k.startswith("PHOTOCHRON_")}
    for k in saved:
        del os.environ[k]
    try:
        for k, v in env.items():
            os.environ[k] = v
        return _apply_env_overrides(base)
    finally:
        for k in env:
            os.environ.pop(k, None)
        for k, v in saved.items():
            os.environ[k] = v


def test_env_override_parses_int() -> None:
    result = _apply({"PHOTOCHRON_MODELS_MAX_IMAGE_SIZE": "2048"})
    assert result["models"]["max_image_size"] == 2048
    assert isinstance(result["models"]["max_image_size"], int)


def test_env_override_parses_float() -> None:
    result = _apply({"PHOTOCHRON_PIPELINE_FACE_AGE_WEIGHT": "0.55"})
    assert result["pipeline"]["face_age_weight"] == pytest.approx(0.55)
    assert isinstance(result["pipeline"]["face_age_weight"], float)


def test_env_override_parses_true_bool() -> None:
    result = _apply({"PHOTOCHRON_INGESTION_EXTRACT_GPS": "true"})
    assert result["ingestion"]["extract_gps"] is True


def test_env_override_parses_false_bool() -> None:
    result = _apply({"PHOTOCHRON_INGESTION_SKIP_DUPLICATES": "no"})
    assert result["ingestion"]["skip_duplicates"] is False


def test_env_override_string_fallback() -> None:
    result = _apply({"PHOTOCHRON_CONTEXT_OLLAMA_HOST": "http://remote:11434"})
    assert result["context"]["ollama_host"] == "http://remote:11434"


def test_env_override_multi_word_key_kept_intact() -> None:
    """``PHOTOCHRON_CONTEXT_OLLAMA_HOST`` → section=context, key=ollama_host."""
    result = _apply({"PHOTOCHRON_CONTEXT_OLLAMA_HOST": "http://x"})
    assert "ollama_host" in result["context"]
    # Should NOT have split as section=context_ollama, key=host.
    assert "host" not in result["context"]


def test_env_override_unknown_section_is_silently_ignored() -> None:
    result = _apply({"PHOTOCHRON_BOGUS_KEY": "value"})
    assert "bogus" not in result
    assert result == {}


def test_env_override_section_only_no_key_is_skipped() -> None:
    """``PHOTOCHRON_PATHS_`` (no key after the section) must not crash."""
    # The implementation matches sections by ``s + "_"`` so a bare section
    # name wouldn't even be recognised — but a trailing underscore-only
    # variant should still be tolerated.
    result = _apply({"PHOTOCHRON_PATHS_": "x"})
    assert "paths" not in result or result.get("paths") == {}


def test_env_override_preserves_existing_config_data() -> None:
    base = {"paths": {"cache_dir": "/keep"}, "models": {"max_image_size": 512}}
    result = _apply({"PHOTOCHRON_MODELS_MAX_IMAGE_SIZE": "4096"}, base=base)
    assert result["paths"]["cache_dir"] == "/keep"
    assert result["models"]["max_image_size"] == 4096


def test_env_override_ignores_unrelated_env_vars() -> None:
    result = _apply({"PATH": "/usr/bin", "HOME": "/tmp"}, base={"paths": {}})
    assert result == {"paths": {}}
