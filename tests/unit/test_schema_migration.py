"""Unit tests for ``photochron.store.schema.migrate_schema`` paths.

Covers:
- Fresh (v0) database: create from scratch and stamp the version.
- Already-at-current (v2): no-op fast path.
- v1 → v2 upgrade: idempotent additive migration adds the ``error_message``
  column without losing existing data.
- ``get_schema_version`` returning 0 when ``pipeline_runs`` table is missing.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from photochron.store.schema import (
    SCHEMA_VERSION,
    create_schema,
    get_schema_version,
    migrate_schema,
)


def _open(tmp_path: Path, name: str = "test.db") -> sqlite3.Connection:
    conn = sqlite3.connect(tmp_path / name)
    conn.row_factory = sqlite3.Row
    return conn


def test_get_schema_version_returns_zero_on_empty_database(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        assert get_schema_version(conn) == 0
    finally:
        conn.close()


def test_migrate_schema_creates_fresh_database(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        migrate_schema(conn)
        # Schema version should now be the current version.
        assert get_schema_version(conn) == SCHEMA_VERSION
        # Core tables exist.
        tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        for required in {"photos", "persons", "faces", "context", "rankings", "pipeline_runs"}:
            assert required in tables
    finally:
        conn.close()


def test_migrate_schema_is_idempotent_when_already_current(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        migrate_schema(conn)
        # Insert sentinel row to verify we don't wipe data on second call.
        conn.execute(
            "INSERT INTO photos (content_hash, file_path) VALUES (?, ?)",
            ("hash_keep", "/tmp/keep.jpg"),
        )
        conn.commit()
        migrate_schema(conn)  # second invocation – fast path
        cnt = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
        assert cnt == 1
        assert get_schema_version(conn) == SCHEMA_VERSION
    finally:
        conn.close()


def test_migrate_schema_upgrades_v1_to_current(tmp_path: Path) -> None:
    """Simulate a v1 database (no error_message column) and migrate it."""
    conn = _open(tmp_path)
    try:
        # Build a "v1" pipeline_runs table missing the error_message column.
        conn.executescript(
            """
            CREATE TABLE pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                schema_version INTEGER NOT NULL DEFAULT 1,
                config_hash TEXT NOT NULL,
                insightface_version TEXT,
                ollama_version TEXT,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                status TEXT NOT NULL,
                photos_processed INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO pipeline_runs (run_id, schema_version, config_hash, start_time, status)
            VALUES ('schema_setup', 1, '', CURRENT_TIMESTAMP, 'completed');
            INSERT INTO pipeline_runs (run_id, schema_version, config_hash, start_time, status)
            VALUES ('run_v1_legacy', 1, 'h', CURRENT_TIMESTAMP, 'completed');
            """
        )
        conn.commit()
        assert get_schema_version(conn) == 1

        migrate_schema(conn)

        # Now at current version, error_message column exists, legacy row preserved.
        assert get_schema_version(conn) == SCHEMA_VERSION
        cols = {row[1] for row in conn.execute("PRAGMA table_info(pipeline_runs)").fetchall()}
        assert "error_message" in cols
        legacy = conn.execute("SELECT run_id FROM pipeline_runs WHERE run_id = 'run_v1_legacy'").fetchone()
        assert legacy is not None
    finally:
        conn.close()


def test_migrate_schema_tolerates_duplicate_alter(tmp_path: Path) -> None:
    """Calling migrate twice on v1 must not blow up on the ALTER TABLE re-attempt."""
    conn = _open(tmp_path)
    try:
        conn.executescript(
            """
            CREATE TABLE pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL UNIQUE,
                schema_version INTEGER NOT NULL DEFAULT 1,
                config_hash TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                status TEXT NOT NULL,
                photos_processed INTEGER DEFAULT 0
            );
            INSERT INTO pipeline_runs (run_id, schema_version, config_hash, start_time, status)
            VALUES ('schema_setup', 1, '', CURRENT_TIMESTAMP, 'completed');
            """
        )
        conn.commit()
        migrate_schema(conn)
        # Force the schema_version back to 1 to trigger the migration path again.
        conn.execute("UPDATE pipeline_runs SET schema_version = 1 WHERE run_id = 'schema_setup'")
        conn.commit()
        # Should reach the OperationalError-swallow branch.
        migrate_schema(conn)
        assert get_schema_version(conn) == SCHEMA_VERSION
    finally:
        conn.close()


def test_create_schema_stamps_version(tmp_path: Path) -> None:
    conn = _open(tmp_path)
    try:
        create_schema(conn)
        conn.commit()
        assert get_schema_version(conn) == SCHEMA_VERSION
    finally:
        conn.close()
