"""
Output layer stage: Generate final output files.
"""

from typing import List
from photochron.pipeline import PipelineStage, register_stage


@register_stage
class OutputLayerStage(PipelineStage):
    """Stage 6: Output generation."""

    @property
    def name(self) -> str:
        return "output_layer"

    @property
    def dependencies(self) -> List[str]:
        return ["ranking_engine"]  # Needs final rankings

    def run(self, run_id: str, config_hash: str) -> None:
        """
        Generate output files.

        Two output modes (both active):

        1. Renamed copies:
           {sort_rank:04d}_{estimated_year}-est_{original_name}.jpg

        2. EXIF-enriched copies:
           Original name preserved, EXIF fields added:
           - DateTimeOriginal: Estimated date
           - ImageDescription: Human-readable summary
           - UserComment: Full JSON result blob

        Additional outputs:
        - photochron_report.json
        - photochron_timeline.csv
        """
        # TODO: Implement output generation
        print(f"[Output Layer] Running stage (placeholder)")

        # In real implementation:
        # 1. Load rankings and associated data
        # 2. Create output directory structure
        # 3. Generate renamed copies
        # 4. Generate EXIF-enriched copies
        # 5. Create report and timeline files
        # 6. Ensure non-destructive operation (copies only)
        pass
