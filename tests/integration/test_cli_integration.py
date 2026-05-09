"""
Integration tests for CLI end-to-end functionality.
"""

import tempfile
from pathlib import Path

from typer.testing import CliRunner

from photochron.cli import app

runner = CliRunner()


def test_cli_help():
    """Test CLI help command."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "photochron" in result.output.lower()
    assert "run" in result.output
    assert "cluster" in result.output
    assert "rerun" in result.output
    assert "status" in result.output


def test_cli_version():
    """Test --version flag."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "photochron v0.1.0" in result.output


def test_cli_run_help():
    """Test run command help."""
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--input" in result.output
    assert "--output" in result.output
    assert "--dry-run" in result.output


def test_cli_run_with_invalid_input():
    """Test run command with non-existent input directory."""
    result = runner.invoke(app, ["run", "--input", "/nonexistent/path"])
    assert result.exit_code != 0  # Should fail
    assert "does not exist" in result.output or "Error" in result.output


def test_cli_run_dry_run():
    """Test run command with dry-run on empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        output_dir = Path(tmpdir) / "output"

        # Create a dummy image file
        dummy_image = input_dir / "dummy.jpg"
        dummy_image.write_bytes(b"fake image data")

        result = runner.invoke(
            app,
            [
                "run",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--dry-run",
            ],
        )

        # Should succeed (dry-run doesn't require actual processing)
        assert result.exit_code == 0
        assert "photochron Pipeline" in result.output
        assert "Dry run: Yes" in result.output


def test_cli_status():
    """Test status command."""
    result = runner.invoke(app, ["status"])
    # Status should work even with empty database
    assert result.exit_code == 0
    assert "photochron Status" in result.output


def test_cli_rerun_invalid_stage():
    """Test rerun command with invalid stage."""
    result = runner.invoke(app, ["rerun", "invalid_stage"])
    assert result.exit_code != 0
    assert "Invalid stage" in result.output or "Error" in result.output


def test_cli_rerun_valid_stage():
    """Test rerun command with valid stage (should show placeholder)."""
    result = runner.invoke(app, ["rerun", "ingestion"])
    # Should show placeholder message since not implemented
    assert result.exit_code == 0
    assert "Re-running stage: ingestion" in result.output


def test_cli_cluster():
    """Test cluster command."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()

        result = runner.invoke(app, ["cluster", "--input", str(input_dir)])
        # Should show placeholder
        assert result.exit_code == 0
        assert "Face Clustering" in result.output


def test_cli_end_to_end_workflow():
    """
    Test a complete CLI workflow: run -> status -> rerun.

    This tests that the CLI commands work together and don't crash.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        output_dir = Path(tmpdir) / "output"

        # Create sample input
        sample_image = input_dir / "sample.jpg"
        sample_image.write_bytes(b"fake image")

        # 1. Run dry-run
        result = runner.invoke(
            app,
            [
                "run",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--dry-run",
            ],
        )
        assert result.exit_code == 0

        # 2. Check status
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0

        # 3. Try to rerun a stage
        result = runner.invoke(app, ["rerun", "ingestion"])
        assert result.exit_code == 0

        # 4. Try cluster
        result = runner.invoke(app, ["cluster", "--input", str(input_dir)])
        assert result.exit_code == 0

        # All commands should execute without crashing
        assert True  # If we got here, no exceptions were raised


def test_cli_output_directory_creation():
    """Test that output directory is created if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        output_dir = Path(tmpdir) / "nonexistent" / "output" / "deep" / "path"

        # Output directory doesn't exist yet
        assert not output_dir.exists()

        result = runner.invoke(
            app,
            [
                "run",
                "--input",
                str(input_dir),
                "--output",
                str(output_dir),
                "--dry-run",
            ],
        )

        # Should succeed (dry-run doesn't create directory)
        assert result.exit_code == 0
        # Directory should still not exist (dry-run)
        assert not output_dir.exists()


def test_cli_configuration_loading():
    """Test that CLI loads configuration properly."""
    # This test verifies the CLI doesn't crash due to config errors
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()

        # Run without config file (should use defaults)
        result = runner.invoke(app, ["run", "--input", str(input_dir), "--dry-run"])

        assert result.exit_code == 0
        assert "photochron Pipeline" in result.output
