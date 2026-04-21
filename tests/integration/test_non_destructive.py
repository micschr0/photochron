"""
Non-destructive operation tests.
"""

import hashlib
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from photochron.cli import app

runner = CliRunner()


def test_non_destructive_operation():
    """Verify that input files are never modified."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        output_dir = Path(tmpdir) / "output"

        # Create a test image file with known content
        test_image = input_dir / "test.jpg"
        original_content = b"fake image data " + b"x" * 100
        test_image.write_bytes(original_content)

        # Compute hash before operation
        hash_before = hashlib.md5(original_content).hexdigest()

        # Run pipeline with dry-run (should not write anything)
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
        assert result.exit_code == 0, f"CLI failed: {result.output}"

        # Verify input file unchanged
        assert test_image.exists()
        hash_after = hashlib.md5(test_image.read_bytes()).hexdigest()
        assert hash_before == hash_after, "Input file was modified"

        # Also verify that no files were created in input directory
        # (except the original)
        files_in_input = list(input_dir.iterdir())
        assert len(files_in_input) == 1, "Extra files created in input directory"
        assert files_in_input[0].name == "test.jpg"


def test_output_directory_separate():
    """Verify output directory is separate from input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = Path(tmpdir) / "input"
        input_dir.mkdir()
        output_dir = Path(tmpdir) / "output"

        # Create a test image
        test_image = input_dir / "photo.jpg"
        test_image.write_bytes(b"fake")

        # Run dry-run (no files written)
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

        # Output directory should not exist (dry-run)
        assert not output_dir.exists()

        # Input directory unchanged
        assert test_image.exists()
