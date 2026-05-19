"""Unit tests for the ``photochron init`` wizard.

The wizard is mostly UI, but the two pure functions (``collect_answers``
in no-input mode and ``render_config_yaml``) and the file-writer are
worth pinning down so future Rich/Typer changes don't silently break them.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from photochron.cli import app
from photochron.cli.wizard import (
    InitAnswers,
    collect_answers,
    render_config_yaml,
    write_files,
)
from photochron.config.models import Config


def test_no_input_returns_safe_defaults() -> None:
    a = collect_answers(no_input=True)
    assert isinstance(a, InitAnswers)
    assert a.face_model == ""
    assert a.primary_llm == ""
    assert a.extract_gps is False
    assert a.write_anchors_template is False


def test_render_config_yaml_validates_against_pydantic() -> None:
    """Round-trip: wizard output must load through Config.model_validate."""
    answers = InitAnswers(
        photos_dir="./photos",
        output_dir="./out",
        face_model="buffalo_l",
        primary_llm="llava-next:7b",
        fallback_llm="moondream2",
        extract_gps=False,
        write_anchors_template=False,
    )
    rendered = render_config_yaml(answers)
    data = yaml.safe_load(rendered)
    cfg = Config.model_validate(data)
    assert cfg.face.model_name == "buffalo_l"
    assert cfg.context.primary_model == "llava-next:7b"
    assert cfg.context.fallback_model == "moondream2"
    assert cfg.paths.output_dir == "./out"
    assert cfg.ingestion.extract_gps is False


def test_write_files_no_input_creates_config(tmp_path: Path) -> None:
    answers = InitAnswers(
        photos_dir=str(tmp_path / "in"),
        output_dir=str(tmp_path / "out"),
        face_model="",
        primary_llm="",
        fallback_llm="",
        extract_gps=False,
        write_anchors_template=False,
    )
    config_path, anchors_path = write_files(answers, tmp_path, no_input=True)
    assert config_path.exists()
    assert anchors_path is None
    # File parses and validates.
    cfg = Config.model_validate(yaml.safe_load(config_path.read_text()))
    assert cfg.paths.output_dir == str(tmp_path / "out")


def test_write_files_no_input_does_not_overwrite_existing(tmp_path: Path) -> None:
    """In non-interactive mode without --force, existing files are preserved."""
    target = tmp_path / "config.yaml"
    target.write_text("preserved: true\n")
    answers = InitAnswers(
        photos_dir="./photos",
        output_dir="./out",
        face_model="",
        primary_llm="",
        fallback_llm="",
        extract_gps=False,
        write_anchors_template=False,
    )
    config_path, _ = write_files(answers, tmp_path, no_input=True, force=False)
    assert config_path.read_text() == "preserved: true\n"


def test_write_files_force_overwrites(tmp_path: Path) -> None:
    target = tmp_path / "config.yaml"
    target.write_text("preserved: true\n")
    answers = InitAnswers(
        photos_dir="./photos",
        output_dir="./out",
        face_model="",
        primary_llm="",
        fallback_llm="",
        extract_gps=False,
        write_anchors_template=False,
    )
    config_path, _ = write_files(answers, tmp_path, no_input=True, force=True)
    assert "preserved" not in config_path.read_text()


def test_init_command_no_input_end_to_end(tmp_path: Path) -> None:
    """Full Typer integration: `photochron init --no-input --dir tmp -f`."""
    runner = CliRunner()
    result = runner.invoke(app, ["init", "--dir", str(tmp_path), "--no-input", "--force"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "config.yaml").exists()
    # Next-steps banner appeared.
    assert "Next steps" in result.output


def test_cluster_and_rerun_are_hidden_from_help() -> None:
    """P0 fix: stub commands must not be advertised in the top-level help."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "cluster" not in result.output
    assert "rerun" not in result.output
    # init, run, status, doctor, review remain visible.
    for cmd in ("init", "run", "status", "doctor", "review"):
        assert cmd in result.output, f"expected `{cmd}` in --help output"
