"""
Pipeline foundation for photochron 6-stage architecture.

Defines the PipelineStage abstract base class and pipeline orchestration.
"""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from photochron.config import get_config
from photochron.store import get_store


class PipelineConfigurationError(RuntimeError):
    """Raised when the pipeline cannot start due to missing configuration."""


@dataclass(frozen=True)
class RunContext:
    """Per-run inputs bound to each stage before ``run()`` is called.

    Replaces the older "mutate the global Config singleton" pattern. Stages
    that need any of these values read ``self.context.<field>`` instead of
    poking the shared Config, which keeps test isolation and concurrent runs
    honest.
    """

    run_id: str
    config_hash: str
    input_dir: Path | None = None
    output_dir: Path | None = None
    dry_run: bool = False


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage reads from and writes to the SQLite Feature Store only.
    Stages are isolated and communicate solely through the database.

    Before ``run()`` is invoked the runner calls ``bind_context()`` to attach
    the per-run :class:`RunContext`. Stages that need ``input_dir`` /
    ``output_dir`` / ``dry_run`` should read them off ``self.context``.
    """

    # Bound by the runner; concrete stages may treat it as non-optional inside
    # their `run()` implementation since the runner is the only legitimate
    # entrypoint.
    context: RunContext | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this pipeline stage."""

    @property
    @abstractmethod
    def dependencies(self) -> list[str]:
        """List of stage names this stage depends on."""

    @abstractmethod
    def run(self, run_id: str, config_hash: str) -> None:
        """Execute this pipeline stage."""

    def bind_context(self, ctx: RunContext) -> None:
        """Attach the per-run context. Called by the runner before ``run()``."""
        self.context = ctx

    def should_run(self, run_id: str) -> bool:
        """Return True iff this specific stage has not yet completed for *run_id*.

        Uses the ``pipeline_stage_runs`` ledger introduced alongside this
        refactor. Falls back to True (run the stage) when the ledger does
        not exist yet, so the first run on an old database still works.
        """
        store = get_store()
        with store.transaction() as conn:
            try:
                cursor = conn.execute(
                    """
                    SELECT 1 FROM pipeline_stage_runs
                    WHERE run_id = ? AND stage_name = ? AND status = 'completed'
                    """,
                    (run_id, self.name),
                )
            except Exception:  # noqa: BLE001 — table may not exist on legacy DBs
                return True
            return cursor.fetchone() is None

    def mark_complete(self, run_id: str, photos_processed: int = 0) -> None:
        """Record successful completion of this stage for *run_id*."""
        now = datetime.now().isoformat()
        store = get_store()
        with store.transaction() as conn:
            # Per-stage ledger row.
            conn.execute(
                """
                INSERT INTO pipeline_stage_runs
                    (run_id, stage_name, status, started_at, ended_at, photos_processed)
                VALUES (?, ?, 'completed', ?, ?, ?)
                ON CONFLICT(run_id, stage_name) DO UPDATE SET
                    status='completed', ended_at=excluded.ended_at,
                    photos_processed=excluded.photos_processed
                """,
                (run_id, self.name, now, now, photos_processed),
            )
            # Bubble up to the per-run total so `status` keeps reporting
            # something useful. We add, not overwrite, so each stage contributes.
            conn.execute(
                "UPDATE pipeline_runs SET end_time = ? WHERE run_id = ?",
                (now, run_id),
            )

    def mark_failed(self, run_id: str, error: str) -> None:
        """Record failure of this stage for *run_id* and persist *error*."""
        now = datetime.now().isoformat()
        truncated = (error or "").strip()[:1024]  # keep DB rows bounded
        store = get_store()
        with store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_stage_runs
                    (run_id, stage_name, status, started_at, ended_at, error_message)
                VALUES (?, ?, 'failed', ?, ?, ?)
                ON CONFLICT(run_id, stage_name) DO UPDATE SET
                    status='failed', ended_at=excluded.ended_at,
                    error_message=excluded.error_message
                """,
                (run_id, self.name, now, now, truncated),
            )
            conn.execute(
                """
                UPDATE pipeline_runs
                SET status='failed', end_time=?, error_message=?
                WHERE run_id=?
                """,
                (now, truncated, run_id),
            )


class PipelineRegistry:
    """Registry of available pipeline stages."""

    def __init__(self) -> None:
        self._stages: dict[str, type[PipelineStage]] = {}

    def register(self, stage_class: type[PipelineStage]) -> None:
        """Register a pipeline stage class."""
        stage = stage_class()
        self._stages[stage.name] = stage_class

    def get_stage(self, name: str) -> PipelineStage | None:
        """Get an instance of a pipeline stage by name."""
        if name not in self._stages:
            return None
        return self._stages[name]()

    def get_dependency_order(self) -> list[str]:
        """Return stage names in topological order.

        Uses Kahn's algorithm. Ties are broken by registration order so the
        sequence remains stable when there are multiple valid orders. Raises
        :class:`RuntimeError` on cycles.
        """
        # Build adjacency
        deps: dict[str, set[str]] = {n: set() for n in self._stages}
        downstream: dict[str, set[str]] = defaultdict(set)

        for name, stage_class in self._stages.items():
            stage = stage_class()
            for d in stage.dependencies:
                if d not in self._stages:
                    # `validate_dependencies` reports this — skip silently here.
                    continue
                deps[name].add(d)
                downstream[d].add(name)

        # Kahn's algorithm, registration order as tiebreaker.
        registration_order = list(self._stages.keys())
        ready: deque[str] = deque(n for n in registration_order if not deps[n])
        ordered: list[str] = []

        while ready:
            n = ready.popleft()
            ordered.append(n)
            # Stable extension: collect newly-ready nodes, then sort by
            # original registration order before extending the queue.
            newly_ready: list[str] = []
            for child in downstream[n]:
                deps[child].discard(n)
                if not deps[child]:
                    newly_ready.append(child)
            newly_ready.sort(key=registration_order.index)
            ready.extend(newly_ready)

        if len(ordered) != len(self._stages):
            remaining = sorted(set(self._stages) - set(ordered))
            raise RuntimeError(f"Cycle detected in pipeline stages: {remaining}")

        return ordered

    def validate_dependencies(self) -> list[str]:
        """Validate that all stage dependencies exist."""
        errors = []
        for name, stage_class in self._stages.items():
            stage = stage_class()
            for dep in stage.dependencies:
                if dep not in self._stages:
                    errors.append(f"Stage '{name}' depends on unknown stage '{dep}'")
        return errors


# Global registry instance
_registry: PipelineRegistry | None = None


def get_registry() -> PipelineRegistry:
    """Get global pipeline registry."""
    global _registry
    if _registry is None:
        _registry = PipelineRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the global registry. Test helper; not used in normal operation."""
    global _registry
    _registry = None


def register_stage(stage_class: type[PipelineStage]) -> type[PipelineStage]:
    """Decorator to register a pipeline stage."""
    get_registry().register(stage_class)
    return stage_class


class PipelineRunner:
    """Orchestrates execution of pipeline stages.

    Keeps the surface small and stage-agnostic so that later frontends
    (Tauri sidecar, HTTP layer) can drive the same entrypoint without
    touching the CLI.
    """

    def __init__(self) -> None:
        self.registry = get_registry()
        self.config = get_config()

    def _compute_config_hash(self) -> str:
        """SHA256 over the serialized config – stable identifier for cache keys."""
        payload = self.config.model_dump_json()
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _require_configured_models(self) -> None:
        """Fail fast when opt-in model fields are still empty."""
        missing: list[str] = []
        if not self.config.face.model_name:
            missing.append("face.model_name")
        if not self.config.context.primary_model:
            missing.append("context.primary_model")
        if not self.config.context.fallback_model:
            missing.append("context.fallback_model")
        if missing:
            raise PipelineConfigurationError(
                "No AI model is configured. Uncomment the desired entries in "
                "config.yaml after verifying their licenses. Missing: " + ", ".join(missing)
            )

    def create_run(self, config_hash: str) -> str:
        """Create a new pipeline run row and return its ID.

        Ensures the SQLite schema exists first – ``migrate_schema`` is
        idempotent, so the happy path just no-ops after the initial install.
        Without this, the very first ``photochron run`` on a fresh machine
        would fail with ``no such table: pipeline_runs``.
        """
        from photochron.store.schema import migrate_schema

        run_id = f"run_{uuid.uuid4().hex[:8]}"

        store = get_store()
        with store.transaction() as conn:
            migrate_schema(conn)
            conn.execute(
                """
                INSERT INTO pipeline_runs (run_id, schema_version, config_hash, start_time, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, 1, config_hash, datetime.now().isoformat(), "running"),
            )

        return run_id

    def run_pipeline(self, input_dir: str, output_dir: str, dry_run: bool = False) -> str:
        """
        Run the full pipeline.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            dry_run: If True, stages that write to disk should skip output

        Returns:
            Run ID for this execution
        """
        self._require_configured_models()

        config_hash = self._compute_config_hash()
        run_id = self.create_run(config_hash)

        ctx = RunContext(
            run_id=run_id,
            config_hash=config_hash,
            input_dir=Path(input_dir) if input_dir else None,
            output_dir=Path(output_dir) if output_dir else None,
            dry_run=dry_run,
        )

        errors = self.registry.validate_dependencies()
        if errors:
            raise RuntimeError(f"Pipeline dependency errors: {', '.join(errors)}")

        stage_order = self.registry.get_dependency_order()

        # Rich progress bar — purely visual; falls back to dim no-op when
        # the user passes `--quiet` (the CLI's root callback already
        # raises the log level above INFO in that case).
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[stage]}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=Console(stderr=True),
            transient=False,
        )
        overall = None

        with progress:
            overall = progress.add_task("pipeline", total=len(stage_order), stage="pipeline")

            for stage_name in stage_order:
                stage = self.registry.get_stage(stage_name)
                if stage is None:
                    raise RuntimeError(f"Stage '{stage_name}' not found in registry")

                stage.bind_context(ctx)

                if not stage.should_run(run_id):
                    logger.info("Skipping stage {} (already completed for run {})", stage_name, run_id)
                    progress.update(overall, advance=1, stage=f"{stage_name} (skipped)")
                    continue

                progress.update(overall, stage=stage_name)
                with logger.contextualize(run_id=run_id, stage=stage_name):
                    logger.info("Stage starting")
                    try:
                        stage.run(run_id, config_hash)
                        logger.info("Stage completed")
                    except Exception as e:
                        logger.exception("Stage failed: {}", e)
                        stage.mark_failed(run_id, str(e))
                        raise
                progress.update(overall, advance=1)

        # Mark the whole run completed.
        store = get_store()
        with store.transaction() as conn:
            conn.execute(
                "UPDATE pipeline_runs SET status='completed', end_time=? WHERE run_id=?",
                (datetime.now().isoformat(), run_id),
            )

        return run_id


__all__ = [
    "PipelineStage",
    "PipelineRegistry",
    "PipelineRunner",
    "PipelineConfigurationError",
    "RunContext",
    "register_stage",
    "get_registry",
    "reset_registry",
]
