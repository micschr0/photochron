"""Unit tests for ``photochron.cli.commands`` command implementations.

Drives the Typer commands through CliRunner with the pipeline runner, store,
and ollama dependencies stubbed so we cover the user-facing branches
(success, configuration error, generic failure, dry-run, invalid stage, etc.)
without loading any AI model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from photochron.cli import app

# ---------------------------------------------------------------------------
# photochron --version / global flags
# ---------------------------------------------------------------------------


def test_version_flag_prints_and_exits() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "photochron v" in result.output


def test_verbose_and_quiet_are_mutually_exclusive() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["--verbose", "--quiet", "status", "--json"])
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


# ---------------------------------------------------------------------------
# photochron run
# ---------------------------------------------------------------------------


def test_run_dry_run_skips_pipeline(tmp_path: Path) -> None:
    runner = CliRunner()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    result = runner.invoke(app, ["run", "--input", str(input_dir), "--dry-run"])
    assert result.exit_code == 0
    assert "Dry run" in result.output


def test_run_success_calls_pipeline_runner(tmp_path: Path) -> None:
    runner = CliRunner()
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "out"

    with patch("photochron.pipeline.PipelineRunner") as MockRunner:
        instance = MockRunner.return_value
        instance.run_pipeline.return_value = "run_abc123"
        result = runner.invoke(
            app,
            ["run", "--input", str(input_dir), "--output", str(output_dir)],
        )

    assert result.exit_code == 0, result.output
    assert "Pipeline run completed: run_abc123" in result.output
    instance.run_pipeline.assert_called_once()
    assert output_dir.exists()


def test_run_configuration_error_exits_with_code_2(tmp_path: Path) -> None:
    runner = CliRunner()
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    from photochron.pipeline import PipelineConfigurationError

    with patch("photochron.pipeline.PipelineRunner") as MockRunner:
        MockRunner.return_value.run_pipeline.side_effect = PipelineConfigurationError("face.model_name missing")
        result = runner.invoke(app, ["run", "--input", str(input_dir)])

    assert result.exit_code == 2
    assert "Configuration error" in result.output
    assert "photochron init" in result.output


def test_run_generic_exception_exits_with_code_1(tmp_path: Path) -> None:
    runner = CliRunner()
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    with patch("photochron.pipeline.PipelineRunner") as MockRunner:
        MockRunner.return_value.run_pipeline.side_effect = RuntimeError("kaboom")
        result = runner.invoke(app, ["run", "--input", str(input_dir)])

    assert result.exit_code == 1
    assert "Pipeline failed" in result.output
    assert "kaboom" in result.output


# ---------------------------------------------------------------------------
# photochron cluster / rerun (stubs)
# ---------------------------------------------------------------------------


def test_cluster_prints_not_implemented(tmp_path: Path) -> None:
    runner = CliRunner()
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    result = runner.invoke(app, ["cluster", "--input", str(input_dir)])
    assert result.exit_code == 0
    assert "not yet implemented" in result.output


def test_rerun_invalid_stage_exits_1() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["rerun", "not_a_stage"])
    assert result.exit_code == 1
    assert "Invalid stage" in result.output


def test_rerun_valid_stage_prints_placeholder() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["rerun", "ingestion"])
    assert result.exit_code == 0
    assert "Re-running stage: ingestion" in result.output
    assert "not yet implemented" in result.output


# ---------------------------------------------------------------------------
# photochron status
# ---------------------------------------------------------------------------


def test_status_rich_table_renders_with_existing_db(tmp_path: Path) -> None:
    runner = CliRunner()
    # Pre-populate the global store fixture path via patching get_store
    from photochron.store import DatabaseStore

    db_path = tmp_path / "cache.db"
    store = DatabaseStore(db_path=db_path)
    with store.transaction() as conn:
        conn.executescript(
            """
            CREATE TABLE photos (id INTEGER PRIMARY KEY);
            CREATE TABLE faces (id INTEGER PRIMARY KEY);
            CREATE TABLE context (id INTEGER PRIMARY KEY);
            CREATE TABLE rankings (id INTEGER PRIMARY KEY);
            CREATE TABLE pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT, status TEXT, start_time TEXT, photos_processed INTEGER
            );
            INSERT INTO pipeline_runs (run_id, status, start_time, photos_processed)
            VALUES ('r1', 'completed', '2026-01-01T00:00:00', 5);
            """
        )

    with patch("photochron.store.get_store", return_value=store):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Cache Statistics" in result.output
    assert "Latest Pipeline Run" in result.output
    assert "r1" in result.output


def test_status_rich_handles_missing_database() -> None:
    """When the DB layer raises, the Rich path reports the error."""
    runner = CliRunner()
    with patch("photochron.store.get_store", side_effect=RuntimeError("db gone")):
        result = runner.invoke(app, ["status"])

    assert result.exit_code == 0
    assert "Error reading database" in result.output
    assert "db gone" in result.output


# ---------------------------------------------------------------------------
# photochron review (CLI wiring)
# ---------------------------------------------------------------------------


def test_review_command_invokes_run_review_tui() -> None:
    runner = CliRunner()
    with patch("photochron.review.run_review_tui", return_value=3) as mock_tui:
        result = runner.invoke(app, ["review", "--threshold", "0.6", "--limit", "5"])
    assert result.exit_code == 0
    assert "Reviewed 3 photo(s)" in result.output
    _, kwargs = mock_tui.call_args
    assert kwargs["threshold"] == 0.6
    assert kwargs["limit"] == 5


# ---------------------------------------------------------------------------
# photochron init (no-input mode hits the wizard's default branch cheaply)
# ---------------------------------------------------------------------------


def test_init_no_input_writes_config(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        ["init", "--dir", str(tmp_path), "--no-input", "--force"],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "config.yaml").exists()
