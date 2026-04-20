"""
Pipeline foundation for PhotoChron 6-stage architecture.

Defines the PipelineStage abstract base class and pipeline orchestration.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type
from datetime import datetime
import uuid

from photochron.store import get_store
from photochron.config import get_config


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
    def dependencies(self) -> List[str]:
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
        self._stages: Dict[str, Type[PipelineStage]] = {}

    def register(self, stage_class: Type[PipelineStage]) -> None:
        """Register a pipeline stage class."""
        stage = stage_class()
        self._stages[stage.name] = stage_class

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get an instance of a pipeline stage by name."""
        if name not in self._stages:
            return None
        return self._stages[name]()

    def get_dependency_order(self) -> List[str]:
        """
        Get stage names in dependency order (topological sort).

        Returns stages in execution order where dependencies are satisfied.
        """
        # Simple implementation: use order of registration
        # In a real implementation, we'd do topological sort
        return list(self._stages.keys())

    def validate_dependencies(self) -> List[str]:
        """Validate that all stage dependencies exist."""
        errors = []
        for name, stage_class in self._stages.items():
            stage = stage_class()
            for dep in stage.dependencies:
                if dep not in self._stages:
                    errors.append(f"Stage '{name}' depends on unknown stage '{dep}'")
        return errors


# Global registry instance
_registry: Optional[PipelineRegistry] = None


def get_registry() -> PipelineRegistry:
    """Get global pipeline registry."""
    global _registry
    if _registry is None:
        _registry = PipelineRegistry()
    return _registry


def register_stage(stage_class: Type[PipelineStage]) -> None:
    """Decorator to register a pipeline stage."""
    get_registry().register(stage_class)
    return stage_class


class PipelineRunner:
    """Orchestrates execution of pipeline stages."""

    def __init__(self):
        self.registry = get_registry()
        self.config = get_config()

    def create_run(self) -> str:
        """Create a new pipeline run and return its ID."""
        run_id = f"run_{uuid.uuid4().hex[:8]}"
        config_hash = "placeholder"  # TODO: Compute actual config hash

        store = get_store()
        with store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO pipeline_runs (run_id, schema_version, config_hash, start_time, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, 1, config_hash, datetime.now().isoformat(), "running"),
            )

        return run_id

    def run_pipeline(
        self, input_dir: str, output_dir: str, dry_run: bool = False
    ) -> str:
        """
        Run the full pipeline.

        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            dry_run: If True, don't write output files

        Returns:
            Run ID for this execution
        """
        run_id = self.create_run()

        # Validate dependencies
        errors = self.registry.validate_dependencies()
        if errors:
            raise RuntimeError(f"Pipeline dependency errors: {', '.join(errors)}")

        # Execute stages in dependency order
        stage_order = self.registry.get_dependency_order()

        for stage_name in stage_order:
            stage = self.registry.get_stage(stage_name)
            if stage is None:
                raise RuntimeError(f"Stage '{stage_name}' not found in registry")

            # Check if stage should run
            if stage.should_run(run_id):
                try:
                    stage.run(run_id, "placeholder")
                    stage.mark_complete(run_id)
                except Exception as e:
                    stage.mark_failed(run_id, str(e))
                    raise

        return run_id


__all__ = [
    "PipelineStage",
    "PipelineRegistry",
    "PipelineRunner",
    "register_stage",
    "get_registry",
]
