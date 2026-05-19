"""Integration tests for review-override loading in RankingEngineStage.

The ``review_overrides`` table is created lazily by ``photochron review`` —
the ranking stage must therefore tolerate its absence on the very first
``photochron run`` after a fresh install.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from photochron.pipeline.stages.ranking_engine import RankingEngineStage
from photochron.store import DatabaseStore, close_store
from photochron.store.schema import create_schema


@pytest.fixture
def store(tmp_path: Path):
    close_store()
    db = DatabaseStore(tmp_path / "ranking.db")
    with db.transaction() as conn:
        create_schema(conn)
    with patch("photochron.pipeline.stages.ranking_engine.get_store", return_value=db):
        yield db
    db.close()
    close_store()


def test_load_review_overrides_missing_table_returns_empty(store) -> None:
    """Fresh DB: review_overrides hasn't been created yet — return {}, don't raise."""
    stage = RankingEngineStage()
    assert stage._load_review_overrides() == {}


def test_load_review_overrides_reads_rows(store) -> None:
    """After ``photochron review`` populated the table, the ranking stage sees it."""
    with store.transaction() as conn:
        conn.execute(
            """
            CREATE TABLE review_overrides (
                photo_id INTEGER PRIMARY KEY,
                estimated_year INTEGER NOT NULL,
                estimated_month INTEGER,
                note TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute("INSERT INTO review_overrides (photo_id, estimated_year, estimated_month) VALUES (1, 1985, 7)")
        conn.execute("INSERT INTO review_overrides (photo_id, estimated_year, estimated_month) VALUES (2, 1992, NULL)")

    stage = RankingEngineStage()
    overrides = stage._load_review_overrides()
    assert overrides == {
        1: {"estimated_year": 1985, "estimated_month": 7},
        2: {"estimated_year": 1992, "estimated_month": None},
    }
