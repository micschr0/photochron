"""Unit tests for the per-stage ``pipeline_stage_runs`` ledger.

Replaces the old whole-run-only ``should_run`` semantics: a stage should now
be skipped if and only if *that specific stage* has already completed for the
given ``run_id``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from photochron.pipeline import PipelineStage
from photochron.store import DatabaseStore, close_store
from photochron.store.schema import create_schema


class _DummyStage(PipelineStage):
    @property
    def name(self) -> str:
        return "dummy"

    @property
    def dependencies(self) -> list[str]:
        return []

    def run(self, run_id: str, config_hash: str) -> None:  # pragma: no cover
        pass


@pytest.fixture
def store(tmp_path: Path):
    """Per-test database with the v2 schema applied."""
    close_store()
    db = DatabaseStore(tmp_path / "ledger.db")
    with db.transaction() as conn:
        create_schema(conn)
        # Provide a parent pipeline_runs row so foreign-key-ish updates pass.
        conn.execute(
            "INSERT INTO pipeline_runs (run_id, config_hash, start_time, status) "
            "VALUES ('r1', 'hash', CURRENT_TIMESTAMP, 'running')"
        )
    with patch("photochron.pipeline.get_store", return_value=db):
        yield db
    db.close()
    close_store()


def test_should_run_true_on_fresh_run(store) -> None:
    assert _DummyStage().should_run("r1") is True


def test_mark_complete_then_should_run_false(store) -> None:
    stage = _DummyStage()
    stage.mark_complete("r1", photos_processed=7)
    assert stage.should_run("r1") is False


def test_per_stage_isolation(store) -> None:
    """Completing one stage must not silently skip a *different* stage."""

    class _OtherStage(_DummyStage):
        @property
        def name(self) -> str:
            return "other"

    _DummyStage().mark_complete("r1", photos_processed=1)
    assert _OtherStage().should_run("r1") is True


def test_mark_failed_persists_error(store) -> None:
    stage = _DummyStage()
    stage.mark_failed("r1", "boom")
    with store.transaction() as conn:
        row = conn.execute(
            "SELECT status, error_message FROM pipeline_stage_runs "
            "WHERE run_id='r1' AND stage_name='dummy'"
        ).fetchone()
        assert row["status"] == "failed"
        assert row["error_message"] == "boom"
        run_row = conn.execute(
            "SELECT status, error_message FROM pipeline_runs WHERE run_id='r1'"
        ).fetchone()
        assert run_row["status"] == "failed"
        assert run_row["error_message"] == "boom"


def test_mark_failed_truncates_long_errors(store) -> None:
    """Keep DB rows bounded; users hit megabyte tracebacks otherwise."""
    stage = _DummyStage()
    stage.mark_failed("r1", "x" * 5000)
    with store.transaction() as conn:
        row = conn.execute(
            "SELECT error_message FROM pipeline_stage_runs WHERE run_id='r1'"
        ).fetchone()
        assert len(row["error_message"]) == 1024
