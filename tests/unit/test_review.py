"""Unit tests for the review TUI module.

Drives ``photochron.review.run_review_tui`` end-to-end against an isolated
SQLite database and a stubbed ``rich.prompt`` so each user-input branch
(accept / skip / edit / quit) can be exercised without a real terminal.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from photochron.review import _ensure_overrides_table, _fetch_candidates, run_review_tui
from photochron.store import DatabaseStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[DatabaseStore]:
    """Isolated DatabaseStore backed by a temp SQLite file."""
    db_path = tmp_path / "test_review.db"
    s = DatabaseStore(db_path=db_path)
    # Minimal schema needed by the review module.
    with s.transaction() as conn:
        conn.executescript(
            """
            CREATE TABLE photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL
            );
            CREATE TABLE rankings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id INTEGER NOT NULL,
                estimated_year INTEGER,
                estimated_month INTEGER,
                confidence REAL NOT NULL
            );
            """
        )
    yield s
    s.close()


def _seed_candidates(store: DatabaseStore, rows: list[tuple[str, int | None, int | None, float]]) -> list[int]:
    """Insert (file_path, est_year, est_month, confidence) rows; return photo_ids."""
    ids: list[int] = []
    with store.transaction() as conn:
        for file_path, year, month, conf in rows:
            cur = conn.execute("INSERT INTO photos (file_path) VALUES (?)", (file_path,))
            photo_id = cur.lastrowid
            assert photo_id is not None
            conn.execute(
                "INSERT INTO rankings (photo_id, estimated_year, estimated_month, confidence) VALUES (?, ?, ?, ?)",
                (photo_id, year, month, conf),
            )
            ids.append(photo_id)
    return ids


def test_ensure_overrides_table_is_idempotent(store: DatabaseStore) -> None:
    with store.transaction() as conn:
        _ensure_overrides_table(conn)
        # Calling a second time must not raise.
        _ensure_overrides_table(conn)
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='review_overrides'")
        assert cur.fetchone() is not None


def test_fetch_candidates_orders_by_confidence_and_honours_limit(store: DatabaseStore) -> None:
    _seed_candidates(
        store,
        [
            ("a.jpg", 2000, None, 0.10),
            ("b.jpg", 2001, 5, 0.40),
            ("c.jpg", 2002, None, 0.30),
            ("d.jpg", 2003, None, 0.80),  # above threshold
        ],
    )
    with store.transaction() as conn:
        all_ = _fetch_candidates(conn, threshold=0.5, limit=None)
        # 3 below threshold, sorted ascending by confidence.
        assert [c["file_path"] for c in all_] == ["a.jpg", "c.jpg", "b.jpg"]

        limited = _fetch_candidates(conn, threshold=0.5, limit=2)
        assert len(limited) == 2
        assert limited[0]["file_path"] == "a.jpg"


def test_run_review_tui_empty_candidates_returns_zero(store: DatabaseStore) -> None:
    console = Console(record=True, width=120)
    with patch("photochron.review.get_store", return_value=store):
        n = run_review_tui(threshold=0.5, limit=None, console=console)
    assert n == 0
    assert "No photos with confidence below" in console.export_text()


def test_run_review_tui_accept_persists_existing_year(store: DatabaseStore) -> None:
    ids = _seed_candidates(store, [("a.jpg", 1995, 7, 0.20)])
    console = Console(record=True, width=120)
    with (
        patch("photochron.review.get_store", return_value=store),
        patch("photochron.review.Prompt.ask", return_value="a"),
    ):
        n = run_review_tui(threshold=0.5, limit=None, console=console)
    assert n == 1
    with store.transaction() as conn:
        row = conn.execute(
            "SELECT estimated_year, estimated_month, note FROM review_overrides WHERE photo_id = ?",
            (ids[0],),
        ).fetchone()
    assert row is not None
    assert row[0] == 1995
    assert row[1] == 7
    assert row[2] == "review-tui"


def test_run_review_tui_skip_does_not_persist(store: DatabaseStore) -> None:
    ids = _seed_candidates(store, [("a.jpg", 2000, None, 0.10)])
    console = Console(record=True, width=120)
    with (
        patch("photochron.review.get_store", return_value=store),
        patch("photochron.review.Prompt.ask", return_value="s"),
    ):
        n = run_review_tui(threshold=0.5, limit=None, console=console)
    assert n == 0
    with store.transaction() as conn:
        row = conn.execute("SELECT 1 FROM review_overrides WHERE photo_id = ?", (ids[0],)).fetchone()
    assert row is None


def test_run_review_tui_quit_stops_iteration(store: DatabaseStore) -> None:
    _seed_candidates(
        store,
        [
            ("a.jpg", 2000, None, 0.10),
            ("b.jpg", 2001, None, 0.20),
            ("c.jpg", 2002, None, 0.30),
        ],
    )
    console = Console(record=True, width=120)
    with (
        patch("photochron.review.get_store", return_value=store),
        patch("photochron.review.Prompt.ask", return_value="q"),
    ):
        n = run_review_tui(threshold=0.5, limit=None, console=console)
    # Quit on first prompt → nothing reviewed.
    assert n == 0


def test_run_review_tui_edit_writes_user_year_and_month(store: DatabaseStore) -> None:
    ids = _seed_candidates(store, [("a.jpg", 1990, None, 0.10)])
    console = Console(record=True, width=120)
    # Prompt.ask is called for "action" first, then for "month" (string).
    prompt_outputs = iter(["e", "3"])
    with (
        patch("photochron.review.get_store", return_value=store),
        patch("photochron.review.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_outputs)),
        patch("photochron.review.IntPrompt.ask", return_value=1985),
    ):
        n = run_review_tui(threshold=0.5, limit=None, console=console)
    assert n == 1
    with store.transaction() as conn:
        row = conn.execute(
            "SELECT estimated_year, estimated_month FROM review_overrides WHERE photo_id = ?",
            (ids[0],),
        ).fetchone()
    assert tuple(row) == (1985, 3)


def test_run_review_tui_edit_with_blank_month_stores_null(store: DatabaseStore) -> None:
    ids = _seed_candidates(store, [("a.jpg", None, None, 0.10)])
    console = Console(record=True, width=120)
    prompt_outputs = iter(["e", ""])  # action then blank month
    with (
        patch("photochron.review.get_store", return_value=store),
        patch("photochron.review.Prompt.ask", side_effect=lambda *a, **kw: next(prompt_outputs)),
        patch("photochron.review.IntPrompt.ask", return_value=2010),
    ):
        n = run_review_tui(threshold=0.5, limit=None, console=console)
    assert n == 1
    with store.transaction() as conn:
        row = conn.execute(
            "SELECT estimated_year, estimated_month FROM review_overrides WHERE photo_id = ?",
            (ids[0],),
        ).fetchone()
    assert tuple(row) == (2010, None)


def test_run_review_tui_handles_integrity_error_gracefully(store: DatabaseStore) -> None:
    """If the INSERT raises IntegrityError, the override is skipped, loop continues."""
    _seed_candidates(
        store,
        [
            ("a.jpg", 2000, None, 0.10),
            ("b.jpg", 2001, None, 0.20),
        ],
    )
    # Drop the FK target so an INSERT into review_overrides with a non-existent
    # photo_id would fail FK — but we want the simpler reliable path: wrap the
    # connection's execute to raise IntegrityError once for the INSERT.
    console = Console(record=True, width=120)

    call_count = {"n": 0}
    original = sqlite3.Connection.executemany  # sentinel use only

    def make_conn_wrapper(real_conn: sqlite3.Connection):
        class Wrapper:
            def __init__(self) -> None:
                self._real = real_conn

            def execute(self, sql: str, *args, **kwargs):  # type: ignore[no-untyped-def]
                if "INSERT INTO review_overrides" in sql:
                    call_count["n"] += 1
                    if call_count["n"] == 1:
                        raise sqlite3.IntegrityError("simulated")
                return self._real.execute(sql, *args, **kwargs)

            def __getattr__(self, name: str):  # type: ignore[no-untyped-def]
                return getattr(self._real, name)

        return Wrapper()

    # Patch DatabaseStore.transaction so the second `with store.transaction()`
    # inside run_review_tui (the per-photo write) yields our wrapper.
    from contextlib import contextmanager

    real_transaction = store.transaction
    transaction_count = {"n": 0}

    @contextmanager
    def patched_transaction():
        transaction_count["n"] += 1
        # Wrap only the per-photo transactions (calls #2 and #3); the first
        # call is the initial candidate fetch + table creation.
        with real_transaction() as conn:
            if transaction_count["n"] >= 2:
                yield make_conn_wrapper(conn)
            else:
                yield conn

    with (
        patch("photochron.review.get_store", return_value=store),
        patch.object(store, "transaction", patched_transaction),
        patch("photochron.review.Prompt.ask", return_value="a"),
    ):
        n = run_review_tui(threshold=0.5, limit=None, console=console)
    # First insert raised → continue counter incremented but reviewed not;
    # second insert succeeded.
    assert n == 1
    assert call_count["n"] == 2
    _ = original  # keep ruff happy
