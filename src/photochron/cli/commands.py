"""
CLI command implementations.
"""

from pathlib import Path

import typer
from rich.console import Console

console = Console()


def _print_privacy_banner() -> None:
    """Remind users that PhotoChron is intended for private family photos."""
    console.print(
        "[yellow]PhotoChron is intended for private family photos. "
        "Treat EXIF data (and any GPS coordinates you enable) as sensitive.[/yellow]"
    )


def run(
    input_dir: Path = typer.Option(
        ...,
        "--input",
        help="Input directory containing photos",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
    output_dir: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for sorted photos (default: ./photochron_output)",
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Run pipeline without writing output files",
    ),
) -> None:
    """
    Run full pipeline on input directory.

    Processes all photos in the input directory through the 6-stage pipeline
    and outputs sorted copies to the output directory.
    """
    from photochron.config import get_config

    config = get_config()

    if output_dir is None:
        output_dir = Path(config.paths.output_dir)

    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    _print_privacy_banner()
    console.print("[bold]PhotoChron Pipeline[/bold]")
    console.print(f"  Input directory: {input_dir}")
    console.print(f"  Output directory: {output_dir}")
    console.print(f"  Dry run: {'Yes' if dry_run else 'No'}")
    console.print()

    if dry_run:
        # Dry-run shows the plan without touching disk or loading heavy models.
        console.print("[dim]Dry run – skipping pipeline execution. Remove --dry-run to run the real pipeline.[/dim]")
        return

    # Import late so CLI help/dry-run paths stay cheap (pipeline pulls heavy deps).
    # Importing the stages package triggers @register_stage for all 6 stages.
    import photochron.pipeline.stages  # noqa: F401
    from photochron.pipeline import PipelineConfigurationError, PipelineRunner

    try:
        runner = PipelineRunner()
        run_id = runner.run_pipeline(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            dry_run=False,
        )
    except PipelineConfigurationError as e:
        console.print(f"[red]Configuration error:[/red] {e}")
        raise typer.Exit(2) from e

    console.print(f"\n[green]✓[/green] Pipeline run completed: {run_id}")
    console.print(f"  [dim]Output in: {output_dir}[/dim]")


def cluster(
    input_dir: Path = typer.Option(
        ...,
        "--input",
        help="Input directory containing photos",
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        resolve_path=True,
    ),
) -> None:
    """
    Face clustering and person assignment.

    Detects faces in all photos, clusters similar faces, and allows
    interactive assignment of clusters to known persons.
    """
    console.print("[bold]Face Clustering[/bold]")
    console.print(f"  Input directory: {input_dir}")
    console.print()

    # TODO: Implement face clustering
    console.print("[yellow]Note: Face clustering not yet implemented[/yellow]")
    console.print("  This command will:")
    console.print("  1. Detect faces in all photos")
    console.print("  2. Cluster similar faces using embedding similarity")
    console.print("  3. Show interactive interface for person assignment")


def rerun(
    stage: str = typer.Argument(
        ...,
        help="Pipeline stage to re-run",
        case_sensitive=False,
    ),
) -> None:
    """
    Re-run specific pipeline stage.

    Re-runs a single pipeline stage without re-running earlier stages.
    Useful when only part of the pipeline needs updating.
    """
    valid_stages = [
        "ingestion",
        "face_layer",
        "context_layer",
        "anchor_layer",
        "ranking_engine",
        "output_layer",
    ]

    if stage.lower() not in valid_stages:
        console.print(f"[red]Error: Invalid stage '{stage}'[/red]")
        console.print(f"Valid stages: {', '.join(valid_stages)}")
        raise typer.Exit(1)

    console.print(f"[bold]Re-running stage: {stage}[/bold]")

    # TODO: Implement stage re-run
    console.print("[yellow]Note: Stage re-run not yet implemented[/yellow]")
    console.print("  This command will:")
    console.print(f"  1. Re-run {stage} only")
    console.print("  2. Use cached results from previous stages")
    console.print("  3. Update dependent stages if needed")


def status() -> None:
    """
    Show pipeline progress and cache stats.

    Displays statistics about cached results, pipeline runs,
    and overall system status.
    """
    from photochron.store import get_store

    console.print("[bold]PhotoChron Status[/bold]")

    try:
        store = get_store()
        with store.transaction() as conn:
            # Get counts from database
            photo_count = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
            face_count = conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0]
            context_count = conn.execute("SELECT COUNT(*) FROM context").fetchone()[0]
            ranking_count = conn.execute("SELECT COUNT(*) FROM rankings").fetchone()[0]

            # Get latest pipeline run
            run_info = conn.execute(
                "SELECT run_id, status, start_time, photos_processed "
                "FROM pipeline_runs ORDER BY start_time DESC LIMIT 1"
            ).fetchone()

        console.print(f"[dim]Database: {store.db_path}[/dim]")
        console.print()
        console.print("[bold]Cache Statistics[/bold]")
        console.print(f"  Photos: {photo_count}")
        console.print(f"  Faces: {face_count}")
        console.print(f"  Context analyses: {context_count}")
        console.print(f"  Rankings: {ranking_count}")

        if run_info:
            console.print()
            console.print("[bold]Latest Pipeline Run[/bold]")
            console.print(f"  Run ID: {run_info[0]}")
            console.print(f"  Status: {run_info[1]}")
            console.print(f"  Started: {run_info[2]}")
            console.print(f"  Photos processed: {run_info[3]}")
        else:
            console.print()
            console.print("[dim]No pipeline runs recorded[/dim]")

    except Exception as e:
        console.print(f"[red]Error reading database: {e}[/red]")
        console.print("[dim]Make sure the pipeline has been run at least once.[/dim]")
