"""
Anchor layer stage: Process user-provided anchor data and constraints.
"""

from typing import List
from photochron.pipeline import PipelineStage, register_stage


@register_stage
class AnchorLayerStage(PipelineStage):
    """Stage 4: Anchor data processing and constraint creation."""

    @property
    def name(self) -> str:
        return "anchor_layer"

    @property
    def dependencies(self) -> List[str]:
        return ["face_layer"]  # Needs person data from faces

    def run(self, run_id: str, config_hash: str) -> None:
        """
        Process anchor data and create constraints.

        1. Load anchors.yaml file
        2. Parse persons, events, known dates
        3. Create constraint set for ranking engine
        4. Validate constraint consistency
        5. Store constraint metadata (not in database - passed to ranking engine)
        """
        # TODO: Implement anchor processing
        print(f"[Anchor Layer] Running stage (placeholder)")

        # In real implementation:
        # 1. Load anchors.yaml
        # 2. Parse with validation
        # 3. Create ConstraintSet object
        # 4. Pass to ranking engine via shared state
        pass
