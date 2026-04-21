"""
Unit tests for DatabaseStore methods.
"""

import sqlite3
from unittest.mock import MagicMock, Mock, patch

from photochron.store import DatabaseStore
from photochron.store.queries import QueryHelper


class TestDatabaseStoreMethods:
    """Test suite for DatabaseStore methods."""

    def test_get_query_helper_returns_query_helper(self):
        """Test get_query_helper returns a QueryHelper instance."""
        # Create a DatabaseStore instance
        store = DatabaseStore()

        # Create a mock connection
        mock_conn = Mock(spec=sqlite3.Connection)

        # Call get_query_helper
        helper = store.get_query_helper(mock_conn)

        # Verify it returns a QueryHelper
        assert isinstance(helper, QueryHelper)

        # Verify QueryHelper was initialized with the connection
        assert helper.conn is mock_conn

    def test_get_query_helper_with_real_connection(self, tmp_path):
        """Test get_query_helper with a real SQLite connection."""
        # Create a temporary database file
        db_path = tmp_path / "test.db"

        # Create DatabaseStore with the temporary path
        store = DatabaseStore(db_path)

        # Get a real connection from the store
        conn = store.connection

        # Call get_query_helper
        helper = store.get_query_helper(conn)

        # Verify it returns a QueryHelper
        assert isinstance(helper, QueryHelper)

        # Verify QueryHelper was initialized with the connection
        assert helper.conn is conn

        # Clean up
        store.close()

    def test_get_query_helper_preserves_connection_state(self):
        """Test get_query_helper doesn't modify the connection."""
        # Create a DatabaseStore instance
        store = DatabaseStore()

        # Create a mock connection with specific attributes
        mock_conn = Mock(spec=sqlite3.Connection)
        mock_conn.row_factory = sqlite3.Row
        mock_conn.isolation_level = "DEFERRED"

        # Call get_query_helper
        helper = store.get_query_helper(mock_conn)

        # Verify connection attributes are preserved
        assert mock_conn.row_factory == sqlite3.Row
        assert mock_conn.isolation_level == "DEFERRED"

        # Verify QueryHelper was initialized with the connection
        assert isinstance(helper, QueryHelper)
        assert helper.conn is mock_conn

    def test_get_query_helper_multiple_calls(self):
        """Test get_query_helper can be called multiple times with different connections."""
        # Create a DatabaseStore instance
        store = DatabaseStore()

        # Create multiple mock connections
        mock_conn1 = Mock(spec=sqlite3.Connection)
        mock_conn2 = Mock(spec=sqlite3.Connection)

        # Call get_query_helper with first connection
        helper1 = store.get_query_helper(mock_conn1)

        # Call get_query_helper with second connection
        helper2 = store.get_query_helper(mock_conn2)

        # Verify both return QueryHelper instances
        assert isinstance(helper1, QueryHelper)
        assert isinstance(helper2, QueryHelper)

        # Verify they were initialized with different connections
        assert helper1.conn is mock_conn1
        assert helper2.conn is mock_conn2
        assert helper1 is not helper2  # Different instances

    def test_get_query_helper_in_transaction_context(self):
        """Test get_query_helper works within a transaction context."""
        # Create a DatabaseStore instance
        store = DatabaseStore()

        # Create a mock connection
        mock_conn = Mock(spec=sqlite3.Connection)

        # Mock the transaction context manager
        mock_transaction_context = MagicMock()
        mock_transaction_context.__enter__.return_value = mock_conn
        mock_transaction_context.__exit__.return_value = None

        with patch.object(store, "transaction", return_value=mock_transaction_context):
            with store.transaction() as conn:
                # Call get_query_helper within transaction
                helper = store.get_query_helper(conn)

                # Verify it returns a QueryHelper
                assert isinstance(helper, QueryHelper)
                assert helper.conn is conn

    def test_get_query_helper_error_handling(self):
        """Test get_query_helper handles invalid input."""
        # Create a DatabaseStore instance
        store = DatabaseStore()

        # Test with None connection - QueryHelper will fail when trying to use it
        # but get_query_helper itself doesn't validate
        helper = store.get_query_helper(None)  # type: ignore
        assert isinstance(helper, QueryHelper)
        assert helper.conn is None

        # Test with wrong type - same behavior
        helper = store.get_query_helper("not a connection")  # type: ignore
        assert isinstance(helper, QueryHelper)
        assert helper.conn == "not a connection"
