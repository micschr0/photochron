"""
CLI command implementations.
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import typer
from loguru import logger
from rich.console import Console

console = Console()


def _print_privacy_banner() -> None:
    """Remind users that photochron is intended for private family photos."""
    console.print(
        "[yellow]photochron is intended for private family photos. "
        "Treat EXIF data (and any GPS coordinates you enable) as sensitive.[/yellow]"
    )


# ---------------------------------------------------------------------------
# photochron init
# ---------------------------------------------------------------------------


def init(
    target_dir: Path = typer.Option(
        Path("."),
        "--dir",
        "-d",
        help="Directory where config.yaml (and optionally anchors.yaml) will be written.",
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    no_input: bool = typer.Option(
        False,
        "--no-input",
        help="Skip every prompt and write safe defaults (CI / scripting friendly).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing files without asking.",
    ),
) -> None:
    """Interactive first-time setup.

    Asks for photo + output dirs, opt-in face / vision-LLM model names, GPS
    preference, and whether to drop an anchors.yaml template. Writes a clean
    config.yaml and prints a numbered "what to do next" list at the end.
    """
    from photochron.cli.wizard import (
        collect_answers,
        print_next_steps,
        write_files,
    )

    answers = collect_answers(no_input=no_input)
    config_path, anchors_path = write_files(answers, target_dir, no_input=no_input, force=force)
    print_next_steps(config_path, anchors_path)


# ---------------------------------------------------------------------------
# photochron run
# ---------------------------------------------------------------------------


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
    console.print("[bold]photochron Pipeline[/bold]")
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
        # The first message most users will see when something's off. Make
        # it actionable: name the next command to run.
        console.print(f"\n[red]Configuration error:[/red] {e}")
        console.print(
            "[dim]Hint: run `photochron init` to set up the missing models, then `photochron doctor` to verify.[/dim]"
        )
        raise typer.Exit(2) from e
    except Exception as e:  # noqa: BLE001 — user-facing safety net
        logger.exception("Pipeline run failed")
        console.print(f"\n[red]Pipeline failed:[/red] {e}")
        console.print("[dim]Hint: re-run with `-v` for DEBUG logs, or check `.photochron/logs/photochron.log`.[/dim]")
        raise typer.Exit(1) from e

    console.print(f"\n[green]✓[/green] Pipeline run completed: {run_id}")
    console.print(f"  [dim]Output in: {output_dir}[/dim]")


# ---------------------------------------------------------------------------
# photochron cluster / rerun (still stubs, kept hidden in CLI wiring)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# photochron status
# ---------------------------------------------------------------------------


def status(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON instead of a Rich table.",
    ),
) -> None:
    """Show pipeline progress and cache stats.

    Displays statistics about cached results, pipeline runs, and overall
    system status.
    """
    from photochron.store import get_store

    payload: dict[str, object] = {"database": None, "cache": {}, "latest_run": None, "face_backend": {}}

    try:
        store = get_store()
        payload["database"] = str(store.db_path)
        with store.transaction() as conn:
            payload["cache"] = {
                "photos": conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0],
                "faces": conn.execute("SELECT COUNT(*) FROM faces").fetchone()[0],
                "context": conn.execute("SELECT COUNT(*) FROM context").fetchone()[0],
                "rankings": conn.execute("SELECT COUNT(*) FROM rankings").fetchone()[0],
            }
            run_info = conn.execute(
                "SELECT run_id, status, start_time, photos_processed "
                "FROM pipeline_runs ORDER BY start_time DESC LIMIT 1"
            ).fetchone()
            if run_info:
                payload["latest_run"] = {
                    "run_id": run_info[0],
                    "status": run_info[1],
                    "started": run_info[2],
                    "photos_processed": run_info[3],
                }
    except Exception as e:  # noqa: BLE001
        payload["database_error"] = str(e)
        logger.exception("status: database read failed")

    try:
        from photochron.config import get_config
        from photochron.face.insightface_wrapper import resolve_providers

        cfg = get_config()
        providers, _ = resolve_providers(cfg.face.backend)
        payload["face_backend"] = {"configured": cfg.face.backend, "resolved": list(providers)}
    except Exception as e:  # noqa: BLE001
        payload["face_backend_error"] = str(e)
        logger.exception("status: face backend probe failed")

    if as_json:
        typer.echo(_json.dumps(payload, indent=2, default=str))
        return

    # Rich rendering
    console.print("[bold]photochron Status[/bold]")
    if payload.get("database"):
        console.print(f"[dim]Database: {payload['database']}[/dim]")
        console.print()
        cache = payload["cache"] or {}
        console.print("[bold]Cache Statistics[/bold]")
        console.print(f"  Photos: {cache.get('photos', 0)}")
        console.print(f"  Faces: {cache.get('faces', 0)}")
        console.print(f"  Context analyses: {cache.get('context', 0)}")
        console.print(f"  Rankings: {cache.get('rankings', 0)}")

        run = payload.get("latest_run")
        if run:
            console.print()
            console.print("[bold]Latest Pipeline Run[/bold]")
            console.print(f"  Run ID: {run['run_id']}")
            console.print(f"  Status: {run['status']}")
            console.print(f"  Started: {run['started']}")
            console.print(f"  Photos processed: {run['photos_processed']}")
        else:
            console.print()
            console.print("[dim]No pipeline runs recorded[/dim]")
    elif "database_error" in payload:
        console.print(f"[red]Error reading database: {payload['database_error']}[/red]")
        console.print("[dim]Make sure the pipeline has been run at least once.[/dim]")

    fb = payload.get("face_backend") or {}
    if fb:
        console.print()
        console.print("[bold]Face Backend[/bold]")
        console.print(f"  Configured: {fb.get('configured')}")
        console.print(f"  Resolved providers: {', '.join(fb.get('resolved') or [])}")
    elif "face_backend_error" in payload:
        console.print(f"[yellow]Could not resolve face backend: {payload['face_backend_error']}[/yellow]")


# ---------------------------------------------------------------------------
# photochron doctor
# ---------------------------------------------------------------------------


def doctor(
    as_json: bool = typer.Option(
        False,
        "--json",
        help="Emit machine-readable JSON (one object with checks + next_steps).",
    ),
) -> None:
    """Diagnose the photochron environment.

    Read-only health check that reports Python/platform info, the ONNX
    Runtime providers actually available on this host, the resolved face
    backend, the configured models (opt-in), and the Ollama reachability
    status. Does not download or load any models.

    The output ends with a numbered "Next steps:" section listing the
    concrete commands to fix anything that's broken — UX optimised for
    first-day use.
    """
    import platform as _platform
    import sys

    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": f"{_platform.system()} {_platform.machine()}",
        "onnxruntime": None,
        "apple_silicon": None,
        "available_providers": [],
        "face_backend": {},
        "configured_models": {},
        "ollama": {},
        "next_steps": [],
    }
    next_steps: list[str] = []

    # onnxruntime + providers ----------------------------------------------
    resolve_providers = None
    is_apple_silicon = None
    try:
        import onnxruntime as ort

        from photochron.face.insightface_wrapper import _is_apple_silicon, resolve_providers

        is_apple_silicon = _is_apple_silicon()
        available = ort.get_available_providers()
        report["onnxruntime"] = ort.__version__
        report["apple_silicon"] = is_apple_silicon
        report["available_providers"] = list(available)

        if is_apple_silicon and "CoreMLExecutionProvider" not in available:
            next_steps.append(
                "Install an onnxruntime wheel with the CoreML EP for Apple Neural Engine: "
                'pip uninstall onnxruntime && pip install "onnxruntime-silicon" '
                "(community project — verify the source first)."
            )
    except ImportError:
        report["onnxruntime"] = None
        next_steps.append("Install onnxruntime: `uv pip install onnxruntime` (or `pip install onnxruntime`).")

    # Configuration ---------------------------------------------------------
    try:
        from photochron.config import get_config

        config = get_config()
        report["face_backend"] = {
            "configured": config.face.backend,
            "resolved": (list(resolve_providers(config.face.backend)[0]) if resolve_providers else []),
        }
        report["configured_models"] = {
            "face.model_name": config.face.model_name,
            "context.primary_model": config.context.primary_model,
            "context.fallback_model": config.context.fallback_model,
            "context.keep_alive": config.context.keep_alive,
            "context.num_ctx": config.context.num_ctx,
            "context.num_gpu": config.context.num_gpu,
            "ingestion.extract_gps": config.ingestion.extract_gps,
        }
        missing: list[str] = []
        if not config.face.model_name:
            missing.append("face.model_name")
        if not config.context.primary_model:
            missing.append("context.primary_model")
        if not config.context.fallback_model:
            missing.append("context.fallback_model")
        if missing:
            report["missing_models"] = missing
            next_steps.append(
                f"Configure the missing opt-in models in config.yaml ({', '.join(missing)}) "
                f"or run `photochron init` to walk through it interactively."
            )
    except Exception as e:  # noqa: BLE001
        report["config_error"] = str(e)
        logger.exception("doctor: config load failed")
        next_steps.append("Fix config.yaml — see the error above. `photochron init` writes a known-good template.")

    # Ollama reachability ---------------------------------------------------
    try:
        import ollama  # type: ignore[import-not-found]

        response = ollama.list()
        models = [m.get("name", "?") for m in response.get("models", [])]
        report["ollama"] = {"reachable": True, "installed_models": models}
    except Exception as e:  # noqa: BLE001
        report["ollama"] = {"reachable": False, "error": str(e)}
        next_steps.append("Install and start Ollama (https://ollama.com), then `ollama pull llava-next:7b moondream2`.")

    report["next_steps"] = next_steps

    # ----------------------------------------------------------------------
    if as_json:
        typer.echo(_json.dumps(report, indent=2, default=str))
        return

    _print_privacy_banner()
    console.print("[bold]photochron doctor[/bold]")
    console.print(f"  Python: {report['python']}")
    console.print(f"  Platform: {report['platform']}")

    if report["onnxruntime"]:
        console.print(f"  onnxruntime: {report['onnxruntime']}")
        console.print(f"  Apple Silicon: {'yes' if is_apple_silicon else 'no'}")
        console.print("  Available ONNX Runtime providers:")
        for p in report["available_providers"]:
            console.print(f"    - {p}")
    else:
        console.print("[yellow]  onnxruntime: not installed[/yellow]")

    if "config_error" in report:
        console.print(f"[red]Config error: {report['config_error']}[/red]")
    else:
        fb = report["face_backend"] or {}
        console.print()
        console.print("[bold]Configuration[/bold]")
        console.print(f"  face.backend: {fb.get('configured')}")
        if fb.get("resolved"):
            console.print(f"  face.backend resolved → {', '.join(fb['resolved'])}")
        for k, v in (report["configured_models"] or {}).items():
            console.print(f"  {k}: {v!r}")
        if report.get("missing_models"):
            console.print(f"  [yellow]Opt-in models not configured: {', '.join(report['missing_models'])}[/yellow]")

    ollama_state = report["ollama"]
    console.print()
    console.print("[bold]Ollama[/bold]")
    if ollama_state.get("reachable"):
        models = ollama_state.get("installed_models") or []
        console.print("  Reachable: yes")
        console.print(f"  Installed models: {', '.join(models) if models else '(none)'}")
    else:
        console.print(f"[yellow]  Reachable: no ({ollama_state.get('error')})[/yellow]")

    # The killer UX feature: numbered, actionable next steps.
    console.print()
    if next_steps:
        console.print("[bold]Next steps[/bold]")
        for i, step in enumerate(next_steps, 1):
            console.print(f"  {i}. {step}")
    else:
        console.print("[green]All checks passed.[/green]")


# ---------------------------------------------------------------------------
# photochron review (TUI for low-confidence photos)
# ---------------------------------------------------------------------------


def review(
    threshold: float = typer.Option(
        0.5,
        "--threshold",
        "-t",
        min=0.0,
        max=1.0,
        help="Walk every photo with confidence < threshold.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        "-n",
        min=1,
        help="Stop after this many photos (handy for big libraries).",
    ),
) -> None:
    """Walk low-confidence photos and let the user accept, edit, or skip each.

    Reads from the existing rankings table; writes accepted user edits back
    into a new ``review_overrides`` table so a subsequent ``photochron run``
    can honour them. (Override application lives in ``ranking/estimator``;
    this command is currently the data-collection half.)
    """
    from photochron.review import run_review_tui

    n = run_review_tui(threshold=threshold, limit=limit, console=console)
    console.print(f"\n[green]✓[/green] Reviewed {n} photo(s).")
