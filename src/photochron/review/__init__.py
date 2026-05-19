"""Interactive review of low-confidence photos.

photochron flags photos whose final confidence falls below a threshold as
``review_needed=true`` in the JSON report. The UX promise is that a human
can quickly walk through those photos and either accept the AI guess or
correct the year — and have those corrections persisted so the next
pipeline run honours them.

This module implements the "walk + collect" half. Application of overrides
lives in :mod:`photochron.ranking.estimator` (follow-up).
"""

from __future__ import annotations

import sqlite3
from typing import Any

from loguru import logger
from rich.console import Console
from rich.prompt import IntPrompt, Prompt

from photochron.store import get_store


def _ensure_overrides_table(conn: sqlite3.Connection) -> None:
    """Idempotently create the ``review_overrides`` table.

    Kept here rather than in the main schema migration because the table is
    only ever written by the ``photochron review`` command — users who never
    invoke that command never pay the cost.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS review_overrides (
            photo_id INTEGER PRIMARY KEY,
            estimated_year INTEGER NOT NULL,
            estimated_month INTEGER,
            note TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
        )
        """
    )


def _fetch_candidates(conn: sqlite3.Connection, threshold: float, limit: int | None) -> list[dict[str, Any]]:
    sql = (
        "SELECT r.photo_id, r.estimated_year, r.estimated_month, r.confidence, p.file_path "
        "FROM rankings r JOIN photos p ON p.id = r.photo_id "
        "WHERE r.confidence < ? ORDER BY r.confidence ASC"
    )
    params: tuple[Any, ...] = (threshold,)
    if limit:
        sql += " LIMIT ?"
        params = (threshold, limit)
    return [dict(row) for row in conn.execute(sql, params).fetchall()]


def run_review_tui(threshold: float, limit: int | None, console: Console) -> int:
    """Walk every photo with confidence < *threshold*; return the count handled."""
    store = get_store()
    with store.transaction() as conn:
        _ensure_overrides_table(conn)
        candidates = _fetch_candidates(conn, threshold, limit)

    if not candidates:
        console.print(
            f"[green]No photos with confidence below {threshold:.2f}.[/green] "
            "Either the pipeline hasn't run yet or every photo is confident."
        )
        return 0

    console.print(f"[bold]Reviewing {len(candidates)} low-confidence photo(s)[/bold] (threshold={threshold:.2f})")
    console.print("[dim]At each prompt: [a]ccept, [s]kip, [e]dit year, [q]uit.[/dim]\n")

    reviewed = 0
    for c in candidates:
        console.print(
            f"[bold]#{c['photo_id']}[/bold]  {c['file_path']}\n"
            f"  current guess: year={c['estimated_year']!s} month={c['estimated_month']!s} "
            f"confidence={c['confidence']:.2f}"
        )
        choice = Prompt.ask("  action", choices=["a", "s", "e", "q"], default="s")
        if choice == "q":
            break
        if choice == "s":
            continue
        if choice == "a":
            year = c["estimated_year"]
            month = c["estimated_month"]
        else:  # 'e'
            year = IntPrompt.ask("  year", default=c["estimated_year"] or 2000)
            month_str = Prompt.ask("  month (1-12, blank = unknown)", default="")
            month = int(month_str) if month_str.strip() else None

        with store.transaction() as conn:
            _ensure_overrides_table(conn)
            try:
                conn.execute(
                    """
                    INSERT INTO review_overrides (photo_id, estimated_year, estimated_month, note)
                    VALUES (?, ?, ?, 'review-tui')
                    ON CONFLICT(photo_id) DO UPDATE SET
                        estimated_year=excluded.estimated_year,
                        estimated_month=excluded.estimated_month,
                        updated_at=CURRENT_TIMESTAMP
                    """,
                    (c["photo_id"], year, month),
                )
            except sqlite3.IntegrityError as e:
                logger.warning("Could not persist override for photo {}: {}", c["photo_id"], e)
                continue
        reviewed += 1
        console.print(f"  [green]✓[/green] saved override year={year} month={month}\n")

    return reviewed


__all__ = ["run_review_tui"]
