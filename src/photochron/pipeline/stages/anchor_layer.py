"""
Anchor layer stage: Process user-provided anchor data and constraints.
"""

from pathlib import Path

from loguru import logger

from photochron.anchor import ConstraintSet, load_anchors
from photochron.config import get_config
from photochron.models import PersonCreate
from photochron.pipeline import PipelineStage, register_stage
from photochron.store import get_store


@register_stage
class AnchorLayerStage(PipelineStage):
    """Stage 4: Anchor data processing and constraint creation."""

    @property
    def name(self) -> str:
        return "anchor_layer"

    @property
    def dependencies(self) -> list[str]:
        return ["face_layer"]

    def run(self, run_id: str, config_hash: str) -> None:
        """Load anchors.yaml, sync persons, and persist the constraint set."""
        logger.info("Starting anchor layer stage")

        anchors_path = self._resolve_anchors_path()
        constraint_set = load_anchors(anchors_path)

        self._upsert_persons(constraint_set)
        self._persist_constraints(run_id, constraint_set, anchors_path)

        logger.info(
            "Anchor layer complete: {} persons, {} constraints ({} hard, {} soft)",
            len(constraint_set.persons),
            len(constraint_set.constraints),
            sum(1 for c in constraint_set.constraints if c.type.value == "hard"),
            sum(1 for c in constraint_set.constraints if c.type.value == "soft"),
        )
        self.mark_complete(run_id, photos_processed=0)

    def _resolve_anchors_path(self) -> Path:
        """Resolve anchors.yaml path using config + common defaults."""
        config = get_config()
        candidate = Path(config.paths.cache_dir).parent / "anchors.yaml"
        fallbacks = [
            Path("anchors.yaml"),
            Path(config.paths.cache_dir) / "anchors.yaml",
            candidate,
        ]
        for path in fallbacks:
            if path.exists():
                return path
        return fallbacks[0]

    def _upsert_persons(self, cs: ConstraintSet) -> None:
        if not cs.persons:
            return
        store = get_store()
        with store.transaction() as conn:
            helper = store.get_query_helper(conn)
            for person in cs.persons:
                helper.upsert_person(
                    PersonCreate(
                        person_id=person.id,
                        name=person.name,
                        birthday=person.birthday,
                    )
                )

    def _persist_constraints(self, run_id: str, cs: ConstraintSet, source_path: Path | None) -> None:
        store = get_store()
        with store.transaction() as conn:
            helper = store.get_query_helper(conn)
            helper.upsert_anchor_constraints(
                run_id=run_id,
                constraints_json=cs.model_dump_json(),
                source_path=str(source_path) if source_path else None,
            )
