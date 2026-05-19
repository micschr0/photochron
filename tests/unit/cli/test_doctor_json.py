"""Verify the JSON output of ``photochron doctor --json`` and ``status --json``.

These are the scripting hooks users wire into NAS dashboards / cron.
"""

from __future__ import annotations

import json

from typer.testing import CliRunner

from photochron.cli import app


def test_doctor_json_is_valid_json_with_expected_keys() -> None:
    runner = CliRunner(mix_stderr=False) if False else CliRunner()  # keep stdout clean
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output

    # Skip log lines emitted before the JSON object.
    last_brace = result.output.rfind("\n{")
    payload = result.output[last_brace + 1 :] if last_brace != -1 else result.output
    data = json.loads(payload[payload.find("{") :])

    expected = {
        "python",
        "platform",
        "onnxruntime",
        "apple_silicon",
        "available_providers",
        "face_backend",
        "configured_models",
        "ollama",
        "next_steps",
    }
    assert expected.issubset(data.keys())
    assert isinstance(data["next_steps"], list)


def test_status_json_is_valid_json() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0, result.output
    payload = result.output[result.output.find("{") :]
    data = json.loads(payload)
    # Always-present keys; values may be partial if no run has happened.
    for key in ("database", "cache", "latest_run", "face_backend"):
        assert key in data
