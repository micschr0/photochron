"""
Ranking engine stage: Combine signals and produce chronological ranking.
"""

import json
from datetime import date

from loguru import logger

from photochron.anchor import ConstraintSet
from photochron.config import get_config
from photochron.models import RankingCreate
from photochron.pipeline import PipelineStage, register_stage
from photochron.ranking.constraints import apply_constraints
from photochron.ranking.estimator import (
    DateEstimate,
    apply_review_overrides,
    combine_signals,
    face_year_estimate,
    rank_estimates,
)
from photochron.store import get_store


@register_stage
class RankingEngineStage(PipelineStage):
    """Stage 5: Chronological ranking computation."""

    @property
    def name(self) -> str:
        return "ranking_engine"

    @property
    def dependencies(self) -> list[str]:
        return ["context_layer", "anchor_layer"]

    def run(self, run_id: str, config_hash: str) -> None:
        """Compute weighted date estimates, apply constraints, and write rankings."""
        logger.info("Starting ranking engine stage")

        config = get_config()
        pipeline_cfg = config.pipeline
        weights = {
            "face": pipeline_cfg.face_age_weight,
            "llm": pipeline_cfg.llm_decade_weight,
            "medium": pipeline_cfg.photo_medium_weight,
        }

        constraint_set = self._load_constraint_set(run_id)

        photos = self._load_photo_signals()
        if not photos:
            logger.info("No photos available for ranking; stage complete")
            self.mark_complete(run_id, photos_processed=0)
            return

        estimates: list[tuple[int, str, DateEstimate]] = []
        today = date.today()
        for photo in photos:
            face_year, face_conf = self._best_face_year(photo, today)
            estimate = combine_signals(
                exif_datetime=photo.get("exif_datetime"),
                face_year=face_year,
                face_confidence=face_conf,
                decade=photo.get("decade"),
                decade_confidence=photo.get("decade_confidence"),
                photo_medium=photo.get("photo_medium"),
                photo_medium_confidence=photo.get("photo_medium_confidence"),
                weights=weights,
                min_confidence_threshold=pipeline_cfg.min_confidence_threshold,
            )
            estimates.append((photo["id"], photo["file_path"], estimate))

        apply_constraints(estimates, constraint_set)

        # Last: pin photos the user manually corrected via `photochron review`.
        # Overrides are intentionally applied *after* anchor constraints so the
        # user's explicit correction is the final word.
        overrides = self._load_review_overrides()
        if overrides:
            n = apply_review_overrides(estimates, overrides)
            logger.info("Applied {} user review override(s)", n)

        ranked = rank_estimates([(photo_id, estimate) for photo_id, _, estimate in estimates])
        rank_by_photo: dict[int, int] = dict(ranked)

        self._write_rankings(estimates, rank_by_photo)

        review_count = sum(1 for _, _, est in estimates if est.review_needed)
        logger.info(
            "Ranking engine complete: {} photos ranked, {} flagged for review",
            len(estimates),
            review_count,
        )
        self.mark_complete(run_id, photos_processed=len(estimates))

    def _load_constraint_set(self, run_id: str) -> ConstraintSet:
        store = get_store()
        with store.transaction() as conn:
            helper = store.get_query_helper(conn)
            raw = helper.get_anchor_constraints_json(run_id)
        if not raw:
            logger.info("No anchor constraints stored for run {}; using empty set", run_id)
            return ConstraintSet()
        return ConstraintSet.model_validate_json(raw)

    def _load_photo_signals(self) -> list[dict]:
        """Join photos with their context row for all ranked signals."""
        store = get_store()
        with store.transaction() as conn:
            cursor = conn.execute(
                """
                SELECT
                    p.id, p.file_path, p.exif_datetime,
                    c.decade, c.decade_confidence,
                    c.photo_medium, c.photo_medium_confidence
                FROM photos p
                LEFT JOIN context c ON p.id = c.photo_id
                ORDER BY p.id
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def _load_review_overrides(self) -> dict[int, dict[str, int | None]]:
        """Read review_overrides table; tolerate absence (lazy table)."""
        store = get_store()
        with store.transaction() as conn:
            try:
                cursor = conn.execute("SELECT photo_id, estimated_year, estimated_month FROM review_overrides")
                rows = cursor.fetchall()
            except Exception:  # noqa: BLE001 — table created lazily by `photochron review`
                return {}
        return {
            int(row["photo_id"]): {
                "estimated_year": row["estimated_year"],
                "estimated_month": row["estimated_month"],
            }
            for row in rows
        }

    def _best_face_year(self, photo: dict, today: date) -> tuple[int | None, float | None]:
        """Pick the highest-confidence face with a birthday and return (year, conf)."""
        store = get_store()
        with store.transaction() as conn:
            helper = store.get_query_helper(conn)
            faces = helper.get_faces_with_person_by_photo(photo["id"])
        best_year = None
        best_conf: float | None = None
        for face in faces:
            if not face.get("birthday") or face.get("age_estimate") is None:
                continue
            year = face_year_estimate(today, face["birthday"], face["age_estimate"])
            if year is None:
                continue
            conf = float(face.get("confidence") or 0.0)
            if best_conf is None or conf > best_conf:
                best_conf = conf
                best_year = year
        return best_year, best_conf

    def _write_rankings(
        self,
        estimates: list[tuple[int, str, DateEstimate]],
        rank_by_photo: dict[int, int],
    ) -> None:
        store = get_store()
        with store.transaction() as conn:
            helper = store.get_query_helper(conn)
            helper.clear_rankings()
            for photo_id, file_path, est in estimates:
                payload = {
                    "file_path": file_path,
                    "year": est.year,
                    "month": est.month,
                    "confidence": est.confidence,
                    "signals": est.signals,
                    "notes": est.notes,
                }
                helper.insert_ranking(
                    RankingCreate(
                        photo_id=photo_id,
                        sort_rank=rank_by_photo[photo_id],
                        estimated_year=est.year,
                        estimated_month=est.month,
                        confidence=float(est.confidence),
                        review_needed=bool(est.review_needed),
                        ranking_json=json.dumps(payload),
                    )
                )
