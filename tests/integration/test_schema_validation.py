"""
Schema validation tests to ensure SQLite schema matches specification.
"""

import sqlite3
import pytest

from photochron.store.schema import SCHEMA_SQL


def extract_table_schema(sql: str) -> dict:
    """
    Extract table definitions from SQL schema.

    Returns dict mapping table name to list of column definitions.
    """
    tables = {}
    current_table = None
    current_columns = []

    lines = sql.split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("CREATE TABLE"):
            # Start new table
            if current_table is not None:
                tables[current_table] = current_columns

            # Extract table name
            parts = line.split()
            table_name = parts[2]  # CREATE TABLE table_name
            current_table = table_name
            current_columns = []
        elif line.startswith("(") or line.endswith("("):
            # Start of column definitions, skip
            continue
        elif line.startswith(")"):
            # End of table definition
            if current_table is not None:
                tables[current_table] = current_columns
                current_table = None
                current_columns = []
        elif current_table is not None and line and not line.startswith("--"):
            # Column definition (simplified)
            # Remove trailing comma if present
            if line.endswith(","):
                line = line[:-1]
            current_columns.append(line)

    return tables


def test_schema_contains_all_tables(database_store):
    """Verify all required tables exist in schema."""
    with database_store.transaction() as conn:
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND name NOT LIKE 'sqlite_%'
        """)
        tables = {row[0] for row in cursor.fetchall()}

    required_tables = {
        "photos",
        "faces",
        "context",
        "rankings",
        "pipeline_runs",
        "persons",
        "anchor_constraints",
    }

    missing_tables = required_tables - tables
    extra_tables = tables - required_tables

    assert len(missing_tables) == 0, f"Missing tables: {missing_tables}"
    assert len(extra_tables) == 0, f"Unexpected tables: {extra_tables}"


def test_photos_table_columns(database_store):
    """Verify photos table has all required columns."""
    with database_store.transaction() as conn:
        cursor = conn.execute("PRAGMA table_info(photos)")
        columns = {row[1] for row in cursor.fetchall()}  # row[1] is column name

    required_columns = {
        "id",
        "content_hash",
        "file_path",
        "downsample_path",
        "exif_datetime",
        "make",
        "model",
        "perceptual_hash",
        "created_at",
    }

    missing_columns = required_columns - columns
    assert len(missing_columns) == 0, (
        f"Missing columns in photos table: {missing_columns}"
    )

    # Check that content_hash is UNIQUE
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='photos'"
    )
    table_sql = cursor.fetchone()[0]
    assert (
        "content_hash TEXT NOT NULL UNIQUE" in table_sql
        or "UNIQUE(content_hash)" in table_sql
    )


def test_faces_table_columns(database_store):
    """Verify faces table has all required columns."""
    with database_store.transaction() as conn:
        cursor = conn.execute("PRAGMA table_info(faces)")
        columns = {row[1] for row in cursor.fetchall()}

    required_columns = {
        "id",
        "photo_id",
        "person_id",
        "embedding",
        "age_estimate",
        "age_std",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "created_at",
    }

    missing_columns = required_columns - columns
    assert len(missing_columns) == 0, (
        f"Missing columns in faces table: {missing_columns}"
    )

    # Check foreign keys
    cursor = conn.execute("PRAGMA foreign_key_list(faces)")
    foreign_keys = [
        (row[2], row[3], row[4]) for row in cursor.fetchall()
    ]  # (table, from, to)

    # Should have foreign key from photo_id to photos.id
    photo_fk = any(
        table == "photos" and from_col == "photo_id" and to_col == "id"
        for table, from_col, to_col in foreign_keys
    )
    assert photo_fk, "Missing foreign key from photo_id to photos.id"

    # Should have foreign key from person_id to persons.id (can be NULL)
    person_fk = any(
        table == "persons" and from_col == "person_id" and to_col == "id"
        for table, from_col, to_col in foreign_keys
    )
    assert person_fk, "Missing foreign key from person_id to persons.id"


def test_context_table_columns(database_store):
    """Verify context table has all required columns."""
    with database_store.transaction() as conn:
        cursor = conn.execute("PRAGMA table_info(context)")
        columns = {row[1] for row in cursor.fetchall()}

    required_columns = {
        "id",
        "photo_id",
        "decade",
        "decade_confidence",
        "season",
        "season_confidence",
        "event_hint",
        "event_confidence",
        "photo_medium",
        "photo_medium_confidence",
        "visual_evidence",
        "alternative_decades",
        "uncertainty_flag",
        "hypothesis_notes",
        "raw_json",
        "created_at",
    }

    missing_columns = required_columns - columns
    assert len(missing_columns) == 0, (
        f"Missing columns in context table: {missing_columns}"
    )

    # Check that photo_id is UNIQUE
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='context'"
    )
    table_sql = cursor.fetchone()[0]
    assert (
        "photo_id INTEGER NOT NULL UNIQUE" in table_sql
        or "UNIQUE(photo_id)" in table_sql
    )


def test_rankings_table_columns(database_store):
    """Verify rankings table has all required columns."""
    with database_store.transaction() as conn:
        cursor = conn.execute("PRAGMA table_info(rankings)")
        columns = {row[1] for row in cursor.fetchall()}

    required_columns = {
        "id",
        "photo_id",
        "sort_rank",
        "estimated_year",
        "estimated_month",
        "confidence",
        "review_needed",
        "ranking_json",
        "created_at",
    }

    missing_columns = required_columns - columns
    assert len(missing_columns) == 0, (
        f"Missing columns in rankings table: {missing_columns}"
    )


def test_pipeline_runs_table_columns(database_store):
    """Verify pipeline_runs table has all required columns."""
    with database_store.transaction() as conn:
        cursor = conn.execute("PRAGMA table_info(pipeline_runs)")
        columns = {row[1] for row in cursor.fetchall()}

    required_columns = {
        "id",
        "run_id",
        "schema_version",
        "config_hash",
        "insightface_version",
        "ollama_version",
        "start_time",
        "end_time",
        "status",
        "photos_processed",
        "created_at",
    }

    missing_columns = required_columns - columns
    assert len(missing_columns) == 0, (
        f"Missing columns in pipeline_runs table: {missing_columns}"
    )

    # Check that run_id is UNIQUE
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='pipeline_runs'"
    )
    table_sql = cursor.fetchone()[0]
    assert "run_id TEXT NOT NULL UNIQUE" in table_sql or "UNIQUE(run_id)" in table_sql


def test_persons_table_columns(database_store):
    """Verify persons table has all required columns."""
    with database_store.transaction() as conn:
        cursor = conn.execute("PRAGMA table_info(persons)")
        columns = {row[1] for row in cursor.fetchall()}

    required_columns = {"id", "person_id", "name", "birthday", "created_at"}

    missing_columns = required_columns - columns
    assert len(missing_columns) == 0, (
        f"Missing columns in persons table: {missing_columns}"
    )

    # Check that person_id is UNIQUE
    cursor = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='persons'"
    )
    table_sql = cursor.fetchone()[0]
    assert (
        "person_id TEXT NOT NULL UNIQUE" in table_sql
        or "UNIQUE(person_id)" in table_sql
    )


def test_schema_indices(database_store):
    """Verify that important indices exist."""
    with database_store.transaction() as conn:
        cursor = conn.execute("""
            SELECT name, tbl_name, sql FROM sqlite_master 
            WHERE type='index' AND name NOT LIKE 'sqlite_%'
        """)
        indices = {
            (row[0], row[1]) for row in cursor.fetchall()
        }  # (index_name, table_name)

    # Check for critical indices
    critical_indices = [
        ("idx_photos_content_hash", "photos"),
        ("idx_faces_photo_id", "faces"),
        ("idx_faces_person_id", "faces"),
        ("idx_context_photo_id", "context"),
        ("idx_rankings_sort_rank", "rankings"),
        ("idx_pipeline_runs_run_id", "pipeline_runs"),
    ]

    for index_name, table_name in critical_indices:
        assert (index_name, table_name) in indices, (
            f"Missing index {index_name} on {table_name}"
        )


def test_schema_foreign_keys_enabled(database_store):
    """Verify foreign key constraints are enabled."""
    with database_store.transaction() as conn:
        cursor = conn.execute("PRAGMA foreign_keys")
        foreign_keys_enabled = cursor.fetchone()[0]

    assert foreign_keys_enabled == 1, "Foreign keys are not enabled"


def test_schema_version_tracking(database_store):
    """Verify schema version is tracked in pipeline_runs table."""
    with database_store.transaction() as conn:
        from photochron.store.schema import get_schema_version

        version = get_schema_version(conn)

        # Should be 1 for initial schema
        assert version == 1, f"Expected schema version 1, got {version}"

        # Check that schema_setup record exists
        cursor = conn.execute(
            "SELECT schema_version FROM pipeline_runs WHERE run_id = 'schema_setup'"
        )
        row = cursor.fetchone()
        assert row is not None, "schema_setup record not found"
        assert row[0] == 1, f"schema_setup record has wrong version: {row[0]}"
