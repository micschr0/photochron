"""
Query helper functions for common database operations.
"""

import json
import sqlite3
from datetime import datetime
from typing import Any

from ..models import (
    Context,
    ContextCreate,
    Face,
    FaceCreate,
    Person,
    PersonCreate,
    Photo,
    PhotoCreate,
    PipelineRun,
    PipelineRunCreate,
    Ranking,
    RankingCreate,
)
from .schema import migrate_schema


class QueryHelper:
    """Helper class for common database operations."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    # Photo operations
    def insert_photo(self, photo: PhotoCreate) -> int:
        """Insert a new photo record and return its ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO photos (content_hash, file_path, downsample_path, exif_datetime, make, model, perceptual_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                photo.content_hash,
                photo.file_path,
                photo.downsample_path,
                photo.exif_datetime,
                photo.make,
                photo.model,
                photo.perceptual_hash,
            ),
        )
        return cursor.lastrowid

    def get_photo_by_id(self, photo_id: int) -> Photo | None:
        """Get photo by ID."""
        cursor = self.conn.execute("SELECT * FROM photos WHERE id = ?", (photo_id,))
        row = cursor.fetchone()
        return Photo.model_validate(dict(row)) if row else None

    def get_photo_by_hash(self, content_hash: str) -> Photo | None:
        """Get photo by content hash."""
        cursor = self.conn.execute("SELECT * FROM photos WHERE content_hash = ?", (content_hash,))
        row = cursor.fetchone()
        return Photo.model_validate(dict(row)) if row else None

    def get_all_photos(self) -> list[Photo]:
        """Get all photos."""
        cursor = self.conn.execute("SELECT * FROM photos ORDER BY created_at")
        return [Photo.model_validate(dict(row)) for row in cursor.fetchall()]

    # Person operations
    def insert_person(self, person: PersonCreate) -> int:
        """Insert a new person record and return its ID."""
        cursor = self.conn.execute(
            "INSERT INTO persons (person_id, name, birthday) VALUES (?, ?, ?)",
            (person.person_id, person.name, person.birthday),
        )
        return cursor.lastrowid

    def get_person_by_id(self, person_id: int) -> Person | None:
        """Get person by database ID."""
        cursor = self.conn.execute("SELECT * FROM persons WHERE id = ?", (person_id,))
        row = cursor.fetchone()
        return Person.model_validate(dict(row)) if row else None

    def get_person_by_person_id(self, person_id: str) -> Person | None:
        """Get person by user-defined person_id."""
        cursor = self.conn.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,))
        row = cursor.fetchone()
        return Person.model_validate(dict(row)) if row else None

    def upsert_person(self, person: PersonCreate) -> int:
        """Insert or update a person record by person_id and return its row id."""
        cursor = self.conn.execute(
            """
            INSERT INTO persons (person_id, name, birthday) VALUES (?, ?, ?)
            ON CONFLICT(person_id) DO UPDATE SET
                name = excluded.name,
                birthday = excluded.birthday
            """,
            (person.person_id, person.name, person.birthday),
        )
        if cursor.lastrowid:
            return cursor.lastrowid
        existing = self.get_person_by_person_id(person.person_id)
        assert existing is not None
        return existing.id

    # Anchor constraint operations
    def upsert_anchor_constraints(self, run_id: str, constraints_json: str, source_path: str | None = None) -> None:
        """Persist serialized ConstraintSet for the given run (replaces existing)."""
        self.conn.execute(
            """
            INSERT INTO anchor_constraints (run_id, source_path, constraints_json)
            VALUES (?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                source_path = excluded.source_path,
                constraints_json = excluded.constraints_json
            """,
            (run_id, source_path, constraints_json),
        )

    def get_anchor_constraints_json(self, run_id: str) -> str | None:
        """Return serialized ConstraintSet JSON for run, or None if not stored."""
        cursor = self.conn.execute(
            "SELECT constraints_json FROM anchor_constraints WHERE run_id = ?",
            (run_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None

    # Face operations
    def insert_face(self, face: FaceCreate) -> int:
        """Insert a new face record and return its ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO faces (photo_id, person_id, embedding, age_estimate, age_std, confidence,
                               bbox_x1, bbox_y1, bbox_x2, bbox_y2)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                face.photo_id,
                face.person_id,
                face.embedding,
                face.age_estimate,
                face.age_std,
                face.confidence,
                face.bbox_x1,
                face.bbox_y1,
                face.bbox_x2,
                face.bbox_y2,
            ),
        )
        return cursor.lastrowid

    def get_faces_by_photo_id(self, photo_id: int) -> list[Face]:
        """Get all faces for a photo."""
        cursor = self.conn.execute("SELECT * FROM faces WHERE photo_id = ? ORDER BY id", (photo_id,))
        return [Face.model_validate(dict(row)) for row in cursor.fetchall()]

    def get_faces_by_person_id(self, person_id: int) -> list[Face]:
        """Get all faces for a person."""
        cursor = self.conn.execute("SELECT * FROM faces WHERE person_id = ? ORDER BY id", (person_id,))
        return [Face.model_validate(dict(row)) for row in cursor.fetchall()]

    # Context operations
    def insert_context(self, context: ContextCreate) -> int:
        """Insert a new context record and return its ID."""
        # Convert list fields to JSON strings for storage
        visual_evidence_json = json.dumps(context.visual_evidence) if context.visual_evidence else None
        alternative_decades_json = json.dumps(context.alternative_decades) if context.alternative_decades else None

        cursor = self.conn.execute(
            """
            INSERT INTO context (
                photo_id, decade, decade_confidence, season, season_confidence,
                event_hint, event_confidence, photo_medium, photo_medium_confidence,
                visual_evidence, alternative_decades, uncertainty_flag, hypothesis_notes, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                context.photo_id,
                context.decade,
                context.decade_confidence,
                context.season,
                context.season_confidence,
                context.event_hint,
                context.event_confidence,
                context.photo_medium,
                context.photo_medium_confidence,
                visual_evidence_json,
                alternative_decades_json,
                context.uncertainty_flag,
                context.hypothesis_notes,
                context.raw_json,
            ),
        )
        return cursor.lastrowid

    def upsert_context(self, context: ContextCreate) -> int:
        """Insert or update a context record and return its ID.

        Uses INSERT OR REPLACE to handle conflicts on the photo_id UNIQUE constraint.
        Returns the ID of the inserted/updated record.
        """
        # Convert list fields to JSON strings for storage
        visual_evidence_json = json.dumps(context.visual_evidence) if context.visual_evidence else None
        alternative_decades_json = json.dumps(context.alternative_decades) if context.alternative_decades else None

        cursor = self.conn.execute(
            """
            INSERT OR REPLACE INTO context (
                photo_id, decade, decade_confidence, season, season_confidence,
                event_hint, event_confidence, photo_medium, photo_medium_confidence,
                visual_evidence, alternative_decades, uncertainty_flag, hypothesis_notes, raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                context.photo_id,
                context.decade,
                context.decade_confidence,
                context.season,
                context.season_confidence,
                context.event_hint,
                context.event_confidence,
                context.photo_medium,
                context.photo_medium_confidence,
                visual_evidence_json,
                alternative_decades_json,
                context.uncertainty_flag,
                context.hypothesis_notes,
                context.raw_json,
            ),
        )
        return cursor.lastrowid

    def get_context_by_photo_id(self, photo_id: int) -> Context | None:
        """Get context for a photo."""
        cursor = self.conn.execute("SELECT * FROM context WHERE photo_id = ?", (photo_id,))
        row = cursor.fetchone()
        if not row:
            return None

        # Convert row to dict and parse JSON fields
        row_dict = dict(row)

        # Parse JSON fields if they exist
        if row_dict.get("visual_evidence"):
            try:
                row_dict["visual_evidence"] = json.loads(row_dict["visual_evidence"])
            except (json.JSONDecodeError, TypeError):
                row_dict["visual_evidence"] = None

        if row_dict.get("alternative_decades"):
            try:
                row_dict["alternative_decades"] = json.loads(row_dict["alternative_decades"])
            except (json.JSONDecodeError, TypeError):
                row_dict["alternative_decades"] = None

        return Context.model_validate(row_dict)

    def get_photos_without_context(self) -> list[Photo]:
        """Get all photos that don't have context analysis."""
        cursor = self.conn.execute(
            """
            SELECT p.* 
            FROM photos p
            LEFT JOIN context c ON p.id = c.photo_id
            WHERE c.id IS NULL
            ORDER BY p.created_at
            """
        )
        return [Photo.model_validate(dict(row)) for row in cursor.fetchall()]

    def get_photos_without_context_batch(self, batch_size: int = 100, offset: int = 0) -> list[Photo]:
        """Get photos without context analysis in batches.

        Args:
            batch_size: Number of photos to return per batch
            offset: Starting offset for pagination

        Returns:
            List of Photo objects for the current batch
        """
        cursor = self.conn.execute(
            """
            SELECT p.* 
            FROM photos p
            LEFT JOIN context c ON p.id = c.photo_id
            WHERE c.id IS NULL
            ORDER BY p.created_at
            LIMIT ? OFFSET ?
            """,
            (batch_size, offset),
        )
        return [Photo.model_validate(dict(row)) for row in cursor.fetchall()]

    def get_faces_with_person_by_photo(self, photo_id: int) -> list[dict[str, Any]]:
        """Return face rows joined with person.birthday for a given photo."""
        cursor = self.conn.execute(
            """
            SELECT f.age_estimate, f.age_std, f.confidence, p.birthday, p.person_id
            FROM faces f
            LEFT JOIN persons p ON f.person_id = p.id
            WHERE f.photo_id = ?
            """,
            (photo_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def clear_rankings(self) -> None:
        """Delete all ranking rows (used before a fresh ranking pass)."""
        self.conn.execute("DELETE FROM rankings")

    # Ranking operations
    def insert_ranking(self, ranking: RankingCreate) -> int:
        """Insert a new ranking record and return its ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO rankings (photo_id, sort_rank, estimated_year, estimated_month, confidence,
                                  review_needed, ranking_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ranking.photo_id,
                ranking.sort_rank,
                ranking.estimated_year,
                ranking.estimated_month,
                ranking.confidence,
                ranking.review_needed,
                ranking.ranking_json,
            ),
        )
        return cursor.lastrowid

    def get_ranking_by_photo_id(self, photo_id: int) -> Ranking | None:
        """Get ranking for a photo."""
        cursor = self.conn.execute("SELECT * FROM rankings WHERE photo_id = ?", (photo_id,))
        row = cursor.fetchone()
        return Ranking.model_validate(dict(row)) if row else None

    def get_all_rankings(self) -> list[Ranking]:
        """Get all rankings sorted by sort_rank."""
        cursor = self.conn.execute("SELECT * FROM rankings ORDER BY sort_rank")
        return [Ranking.model_validate(dict(row)) for row in cursor.fetchall()]

    # Pipeline run operations
    def insert_pipeline_run(self, run: PipelineRunCreate) -> int:
        """Insert a new pipeline run record and return its ID."""
        cursor = self.conn.execute(
            """
            INSERT INTO pipeline_runs (run_id, schema_version, config_hash, insightface_version,
                                       ollama_version, start_time, status, photos_processed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.schema_version,
                run.config_hash,
                run.insightface_version,
                run.ollama_version,
                run.start_time.isoformat() if isinstance(run.start_time, datetime) else run.start_time,
                run.status,
                run.photos_processed,
            ),
        )
        return cursor.lastrowid

    def update_pipeline_run(self, run_id: str, **updates) -> None:
        """Update pipeline run fields."""
        if not updates:
            return

        set_clause = ", ".join(f"{key} = ?" for key in updates.keys())
        values = list(updates.values())
        values.append(run_id)

        self.conn.execute(f"UPDATE pipeline_runs SET {set_clause} WHERE run_id = ?", values)

    def get_pipeline_run(self, run_id: str) -> PipelineRun | None:
        """Get pipeline run by ID."""
        cursor = self.conn.execute("SELECT * FROM pipeline_runs WHERE run_id = ?", (run_id,))
        row = cursor.fetchone()
        return PipelineRun.model_validate(dict(row)) if row else None

    def get_latest_pipeline_run(self) -> PipelineRun | None:
        """Get the most recent pipeline run."""
        cursor = self.conn.execute("SELECT * FROM pipeline_runs ORDER BY start_time DESC LIMIT 1")
        row = cursor.fetchone()
        return PipelineRun.model_validate(dict(row)) if row else None

    # Cache invalidation helpers
    def mark_photo_invalid(self, photo_id: int) -> None:
        """Mark all dependent features for a photo as invalid."""
        # For now, we'll delete dependent rows
        # In a more sophisticated implementation, we'd mark them as stale
        self.conn.execute("DELETE FROM faces WHERE photo_id = ?", (photo_id,))
        self.conn.execute("DELETE FROM context WHERE photo_id = ?", (photo_id,))
        self.conn.execute("DELETE FROM rankings WHERE photo_id = ?", (photo_id,))

    def get_photo_count(self) -> int:
        """Get total number of photos in database."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM photos")
        return cursor.fetchone()[0]

    def get_face_count(self) -> int:
        """Get total number of face detections."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM faces")
        return cursor.fetchone()[0]

    def get_context_count(self) -> int:
        """Get total number of context analyses."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM context")
        return cursor.fetchone()[0]

    def get_ranking_count(self) -> int:
        """Get total number of rankings."""
        cursor = self.conn.execute("SELECT COUNT(*) FROM rankings")
        return cursor.fetchone()[0]


def initialize_database(conn: sqlite3.Connection) -> QueryHelper:
    """Initialize database schema and return query helper."""

    migrate_schema(conn)
    return QueryHelper(conn)
