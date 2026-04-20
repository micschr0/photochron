"""
PhotoChron CLI interface using Typer.
"""

from pathlib import Path
from typing import Optional
import typer

from .commands import run, cluster, rerun, status

# Create Typer app
app = typer.Typer(
    name="photochron",
    help="Local-first CLI tool that sorts digitized family photos without timestamps into chronological order",
    add_completion=False,
)

# Add commands
app.command(name="run", help="Run full pipeline on input directory")(run)
app.command(name="cluster", help="Face clustering and person assignment")(cluster)
app.command(name="rerun", help="Re-run specific pipeline stage")(rerun)
app.command(name="status", help="Show pipeline progress and cache stats")(status)


@app.callback(invoke_without_command=True)
def main(
    version: Optional[bool] = typer.Option(
        None,
        "--version",
        help="Show version and exit",
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG-level logging",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Only show WARNING and above",
    ),
) -> None:
    """
    PhotoChron - Sort family photos chronologically using AI.

    All inference runs fully on-device. No data leaves the machine.
    """
    if version:
        from photochron import __version__

        typer.echo(f"PhotoChron v{__version__}")
        raise typer.Exit()

    from photochron.config import get_config
    from photochron.logging_config import setup_logging

    level_override = None
    if verbose and quiet:
        typer.echo("Error: --verbose and --quiet are mutually exclusive", err=True)
        raise typer.Exit(2)
    if verbose:
        level_override = "DEBUG"
    elif quiet:
        level_override = "WARNING"

    setup_logging(get_config().logging, level_override=level_override)


# Export for __main__.py
__all__ = ["app"]
