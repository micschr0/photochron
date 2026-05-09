"""
SQLite Feature Store for photochron pipeline.

This module provides database connection management, schema definition,
and query helpers for the 6-stage pipeline.
"""

import sqlite3
import threading
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from photochron import CACHE_DIR

from .queries import QueryHelper


class DatabaseStore:
    """Manages SQLite database connections with connection pooling."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or CACHE_DIR / "cache.db"
        self._local = threading.local()

    @property
    def connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn"):
            self._local.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
            # Enable foreign keys
            self._local.conn.execute("PRAGMA foreign_keys = ON")
            # Enable WAL mode for better concurrency
            self._local.conn.execute("PRAGMA journal_mode = WAL")
        return self._local.conn

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database transactions."""
        conn = self.connection
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def close(self) -> None:
        """Close thread-local connection if it exists."""
        if hasattr(self._local, "conn"):
            self._local.conn.close()
            delattr(self._local, "conn")

    def get_query_helper(self, conn: sqlite3.Connection) -> QueryHelper:
        """Get a QueryHelper instance for the given connection."""
        return QueryHelper(conn)


# Global store instance
_store: DatabaseStore | None = None


def get_store() -> DatabaseStore:
    """Get global database store instance."""
    global _store
    if _store is None:
        _store = DatabaseStore()
    return _store


def close_store() -> None:
    """Close global database store."""
    global _store
    if _store is not None:
        _store.close()
        _store = None


__all__ = ["DatabaseStore", "get_store", "close_store"]
