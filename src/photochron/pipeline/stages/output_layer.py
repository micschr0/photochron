"""
Output layer stage: Generate final output files.
"""

import json
import shutil
from pathlib import Path
from typing import Any

from loguru import logger

from photochron.config import get_config
from photochron.output import (
    build_renamed_filename,
    build_report,
    build_timeline_rows,
    write_exif_fields,
)
from photochron.pipeline import PipelineStage, register_stage
from photochron.store import get_store


@register_stage
class OutputLayerStage(PipelineStage):
    """Stage 6: Output generation."""

    @property
    def name(self) -> str:
        return "output_layer"

    @property
    def dependencies(self) -> list[str]:
        return ["ranking_engine"]

    def run(self, run_id: str, config_hash: str) -> None:
        """Write renamed copies, EXIF-enriched copies, JSON report, and CSV timeline."""
        logger.info("Starting output layer stage")

        output_dir = Path(getattr(self, "_output_override", None) or get_config().paths.output_dir)
        dry_run = bool(getattr(self, "_dry_run", False))

        rows = self._load_rankings()
        if not rows:
            logger.info("No rankings present; nothing to write")
            self.mark_complete(run_id, photos_processed=0)
            return

        renamed_dir = output_dir / "renamed"
        enriched_dir = output_dir / "exif_enriched"

        if not dry_run:
            renamed_dir.mkdir(parents=True, exist_ok=True)
            enriched_dir.mkdir(parents=True, exist_ok=True)

        report_rows: list[dict[str, Any]] = []
        written = 0
        skipped = 0
        for row in rows:
            original_path = Path(row["file_path"])
            if not original_path.exists():
                logger.warning("Skipping missing source file {}", original_path)
                skipped += 1
                continue

            renamed_name = build_renamed_filename(
                sort_rank=row["sort_rank"],
                estimated_year=row["estimated_year"],
                original_name=original_path.name,
            )
            renamed_target = renamed_dir / renamed_name
            enriched_target = enriched_dir / original_path.name

            if not dry_run:
                shutil.copy2(original_path, renamed_target)
                shutil.copy2(original_path, enriched_target)

                signals = row.get("signals", {})
                write_exif_fields(
                    target_path=enriched_target,
                    year=row["estimated_year"],
                    month=row["estimated_month"],
                    confidence=row["confidence"],
                    signals=signals,
                    review_needed=row["review_needed"],
                    full_result=row["raw_ranking"],
                )
            written += 1

            report_rows.append(
                {
                    "photo_id": row["photo_id"],
                    "original_name": original_path.name,
                    "original_path": str(original_path),
                    "sort_rank": row["sort_rank"],
                    "estimated_year": row["estimated_year"],
                    "estimated_month": row["estimated_month"],
                    "confidence": row["confidence"],
                    "review_needed": row["review_needed"],
                    "output_renamed": str(renamed_target),
                    "output_enriched": str(enriched_target),
                }
            )

        report_payload = build_report(run_id, report_rows)
        timeline_csv = build_timeline_rows(report_rows)

        if not dry_run:
            (output_dir / "photochron_report.json").write_text(
                json.dumps(report_payload, indent=2), encoding="utf-8"
            )
            (output_dir / "photochron_timeline.csv").write_text(
                timeline_csv, encoding="utf-8"
            )

        logger.info(
            "Output layer complete: {} photos written, {} skipped (dry_run={})",
            written,
            skipped,
            dry_run,
        )
        self.mark_complete(run_id, photos_processed=written)

    def _load_rankings(self) -> list[dict[str, Any]]:
        """Load ranking rows joined with photos metadata, sorted by sort_rank."""
        store = get_store()
        with store.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT
                    r.photo_id, r.sort_rank, r.estimated_year, r.estimated_month,
                    r.confidence, r.review_needed, r.ranking_json,
                    p.file_path
                FROM rankings r
                JOIN photos p ON p.id = r.photo_id
                ORDER BY r.sort_rank
                """
            )
            raw_rows = [dict(row) for row in cursor.fetchall()]

        enriched: list[dict[str, Any]] = []
        for raw in raw_rows:
            try:
                parsed = json.loads(raw["ranking_json"]) if raw["ranking_json"] else {}
            except json.JSONDecodeError:
                parsed = {}
            enriched.append(
                {
                    "photo_id": raw["photo_id"],
                    "sort_rank": raw["sort_rank"],
                    "estimated_year": raw["estimated_year"],
                    "estimated_month": raw["estimated_month"],
                    "confidence": raw["confidence"],
                    "review_needed": bool(raw["review_needed"]),
                    "file_path": raw["file_path"],
                    "signals": parsed.get("signals", {}),
                    "raw_ranking": parsed,
                }
            )
        return enriched
