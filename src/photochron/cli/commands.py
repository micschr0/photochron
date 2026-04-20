"""
CLI command implementations.
"""

from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


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
    output_dir: Optional[Path] = typer.Option(
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

    # Set default output directory if not provided
    if output_dir is None:
        output_dir = Path(config.paths.output_dir)

    # Create output directory if it doesn't exist (unless dry run)
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Show pipeline configuration
    console.print(f"[bold]PhotoChron Pipeline[/bold]")
    console.print(f"  Input directory: {input_dir}")
    console.print(f"  Output directory: {output_dir}")
    console.print(f"  Dry run: {'Yes' if dry_run else 'No'}")
    console.print()

    # TODO: Implement actual pipeline execution
    # For now, just show a mock progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing pipeline...", total=None)

        # Simulate pipeline stages
        stages = [
            "Ingestion",
            "Face Layer",
            "Context Layer",
            "Anchor Layer",
            "Ranking Engine",
            "Output Layer",
        ]

        for i, stage in enumerate(stages):
            progress.update(task, description=f"Running {stage}...")
            # Simulate work
            import time

            time.sleep(0.5)

        progress.update(task, description="Pipeline complete!")

    console.print("\n[green]✓[/green] Pipeline execution completed")
    console.print(f"  [dim]Output ready in: {output_dir}[/dim]")


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
                "SELECT run_id, status, start_time, photos_processed FROM pipeline_runs ORDER BY start_time DESC LIMIT 1"
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
