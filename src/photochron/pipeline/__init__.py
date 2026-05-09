"""
Pipeline foundation for photochron 6-stage architecture.

Defines the PipelineStage abstract base class and pipeline orchestration.
"""

import hashlib
import uuid
from abc import ABC, abstractmethod
from datetime import datetime

from loguru import logger

from photochron.config import get_config
from photochron.store import get_store


class PipelineConfigurationError(RuntimeError):
    """Raised when the pipeline cannot start due to missing configuration."""


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.

    Each stage reads from and writes to the SQLite Feature Store only.
    Stages are isolated and communicate solely through the database.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name of this pipeline stage."""
        pass

    @property
    @abstractmethod
    def dependencies(self) -> list[str]:
        """List of stage names this stage depends on."""
        pass

    @abstractmethod
    def run(self, run_id: str, config_hash: str) -> None:
        """
        Execute this pipeline stage.

        Args:
            run_id: Unique identifier for this pipeline run
            config_hash: Hash of configuration used for cache invalidation
        """
        pass

    def should_run(self, run_id: str) -> bool:
        """
        Determine if this stage needs to run.

        Default implementation checks if stage has already completed
        for this run_id in the database.

        Override for more sophisticated cache invalidation logic.
        """
        store = get_store()
        with store.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT 1 FROM pipeline_runs 
                WHERE run_id = ? AND status = 'completed' 
                AND photos_processed > 0
                """,
                (run_id,),
            )
            return cursor.fetchone() is None

    def mark_complete(self, run_id: str, photos_processed: int = 0) -> None:
        """Mark this stage as complete in the database."""
        store = get_store()
        with store.transaction() as conn:
            conn.execute(
                """
                UPDATE pipeline_runs 
                SET status = 'completed', end_time = ?, photos_processed = ?
                WHERE run_id = ?
                """,
                (datetime.now().isoformat(), photos_processed, run_id),
            )

    def mark_failed(self, run_id: str, error: str) -> None:
        """Mark this stage as failed in the database."""
        store = get_store()
        with store.transaction() as conn:
            conn.execute(
                """
                UPDATE pipeline_runs 
                SET status = 'failed', end_time = ?
                WHERE run_id = ?
                """,
                (datetime.now().isoformat(), run_id),
            )
            # Log error (could add error logging table)


class PipelineRegistry:
    """Registry of available pipeline stages."""

    def __init__(self):
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
        """
        Get stage names in dependency order (topological sort).

        Returns stages in execution order where dependencies are satisfied.
        """
        # Simple implementation: use order of registration
        # In a real implementation, we'd do topological sort
        return list(self._stages.keys())

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


def register_stage(stage_class: type[PipelineStage]) -> None:
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
        payload = self.config.model_dump_json(exclude={"input_dir", "dry_run"})
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

        # Runtime overrides – stages read these from config to stay stateless.
        self.config.input_dir = input_dir
        self.config.paths.output_dir = output_dir
        self.config.dry_run = dry_run

        config_hash = self._compute_config_hash()
        run_id = self.create_run(config_hash)

        errors = self.registry.validate_dependencies()
        if errors:
            raise RuntimeError(f"Pipeline dependency errors: {', '.join(errors)}")

        stage_order = self.registry.get_dependency_order()

        for stage_name in stage_order:
            stage = self.registry.get_stage(stage_name)
            if stage is None:
                raise RuntimeError(f"Stage '{stage_name}' not found in registry")

            if not stage.should_run(run_id):
                continue

            with logger.contextualize(run_id=run_id, stage=stage_name):
                logger.info("Stage starting")
                try:
                    stage.run(run_id, config_hash)
                    stage.mark_complete(run_id)
                    logger.info("Stage completed")
                except Exception as e:
                    logger.exception("Stage failed: {}", e)
                    stage.mark_failed(run_id, str(e))
                    raise

        return run_id


__all__ = [
    "PipelineStage",
    "PipelineRegistry",
    "PipelineRunner",
    "PipelineConfigurationError",
    "register_stage",
    "get_registry",
]
