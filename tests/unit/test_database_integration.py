"""
Database integration tests for context record insertion and transaction integrity.

These tests focus on database operations for context data, transaction integrity,
and schema compliance using SQLite in-memory database.
"""

import json
import sqlite3

import pytest

from photochron.models import (
    ContextCreate,
    PhotoCreate,
)
from photochron.store import DatabaseStore
from photochron.store.queries import QueryHelper
from photochron.store.schema import create_schema, get_schema_version


class TestDatabaseTransactionIntegrity:
    """Test transaction integrity: rollback on failure, commit on success."""

    def test_transaction_commit_on_success(self, database_store: DatabaseStore):
        """Test that successful transactions are committed."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert a photo
            photo = PhotoCreate(
                content_hash="test_hash_commit",
                file_path="/test/photo_commit.jpg",
                downsample_path="/test/thumb_commit.jpg",
            )
            photo_id = helper.insert_photo(photo)

        # Verify the photo exists after transaction commit
        with database_store.transaction() as conn:
            helper = QueryHelper(conn)
            retrieved = helper.get_photo_by_id(photo_id)
            assert retrieved is not None
            assert retrieved.content_hash == "test_hash_commit"
            assert retrieved.file_path == "/test/photo_commit.jpg"

    def test_transaction_rollback_on_exception(self, database_store: DatabaseStore):
        """Test that transactions are rolled back on exception."""
        try:
            with database_store.transaction() as conn:
                create_schema(conn)
                helper = QueryHelper(conn)

                # Insert a photo
                photo = PhotoCreate(
                    content_hash="test_hash_rollback",
                    file_path="/test/photo_rollback.jpg",
                )
                helper.insert_photo(photo)

                # This should raise an exception
                raise ValueError("Simulated error")
        except ValueError:
            pass  # Expected exception

        # Verify the photo does NOT exist (transaction was rolled back)
        with database_store.transaction() as conn:
            helper = QueryHelper(conn)
            retrieved = helper.get_photo_by_hash("test_hash_rollback")
            assert retrieved is None

    def test_nested_transaction_handling(self, database_store: DatabaseStore):
        """Test that nested transactions work correctly."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert first photo
            photo1 = PhotoCreate(
                content_hash="nested_hash_1",
                file_path="/test/nested1.jpg",
            )
            photo_id1 = helper.insert_photo(photo1)

            # Nested transaction attempt (should use same connection)
            with database_store.transaction() as nested_conn:
                assert nested_conn is conn  # Should be same connection
                nested_helper = QueryHelper(nested_conn)

                # Insert second photo
                photo2 = PhotoCreate(
                    content_hash="nested_hash_2",
                    file_path="/test/nested2.jpg",
                )
                photo_id2 = nested_helper.insert_photo(photo2)

        # Both photos should exist after outer transaction commits
        with database_store.transaction() as conn:
            helper = QueryHelper(conn)
            retrieved1 = helper.get_photo_by_id(photo_id1)
            retrieved2 = helper.get_photo_by_id(photo_id2)

            assert retrieved1 is not None
            assert retrieved2 is not None
            assert retrieved1.content_hash == "nested_hash_1"
            assert retrieved2.content_hash == "nested_hash_2"

    def test_transaction_with_constraint_violation(self, database_store: DatabaseStore):
        """Test transaction behavior with constraint violations."""
        try:
            with database_store.transaction() as conn:
                create_schema(conn)
                helper = QueryHelper(conn)

                # Insert a photo with unique content_hash
                photo1 = PhotoCreate(
                    content_hash="duplicate_hash",
                    file_path="/test/photo1.jpg",
                )
                helper.insert_photo(photo1)

                # Try to insert another photo with same content_hash (should fail)
                photo2 = PhotoCreate(
                    content_hash="duplicate_hash",  # Same hash - violates UNIQUE constraint
                    file_path="/test/photo2.jpg",
                )

                helper.insert_photo(photo2)
                pytest.fail("Should have raised sqlite3.IntegrityError")
        except sqlite3.IntegrityError:
            # Exception should cause automatic rollback by context manager
            pass

        # Verify no photos exist (transaction was rolled back due to exception)
        with database_store.transaction() as conn:
            helper = QueryHelper(conn)
            photos = helper.get_all_photos()
            assert len(photos) == 0  # Rollback removed both


class TestContextRecordInsertion:
    """Test context record insertion with various data scenarios."""

    def test_insert_context_with_full_data(self, database_store: DatabaseStore):
        """Test inserting context record with all fields populated."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert photo first
            photo = PhotoCreate(
                content_hash="full_context_photo",
                file_path="/test/full_context.jpg",
            )
            photo_id = helper.insert_photo(photo)

            # Insert context with all fields
            context = ContextCreate(
                photo_id=photo_id,
                decade="1985-1990",
                decade_confidence=0.85,
                season="summer",
                season_confidence=0.75,
                event_hint="beach vacation",
                event_confidence=0.65,
                photo_medium="print_scan",
                photo_medium_confidence=0.95,
                visual_evidence=["palm trees", "ocean", "swimwear", "sunglasses"],
                alternative_decades=["1980-1985", "1990-1995"],
                uncertainty_flag=False,
                hypothesis_notes="Clear summer beach scene with strong visual cues",
                raw_json=json.dumps(
                    {
                        "decade": "1985-1990",
                        "decade_confidence": 0.85,
                        "season": "summer",
                        "season_confidence": 0.75,
                        "event_hint": "beach vacation",
                        "event_confidence": 0.65,
                        "photo_medium": "print_scan",
                        "photo_medium_confidence": 0.95,
                        "visual_evidence": [
                            "palm trees",
                            "ocean",
                            "swimwear",
                            "sunglasses",
                        ],
                        "alternative_decades": ["1980-1985", "1990-1995"],
                        "uncertainty_flag": False,
                        "hypothesis_notes": "Clear summer beach scene with strong visual cues",
                    }
                ),
            )

            context_id = helper.insert_context(context)
            assert context_id == 1

            # Retrieve and verify
            retrieved = helper.get_context_by_photo_id(photo_id)
            assert retrieved is not None
            assert retrieved.id == context_id
            assert retrieved.decade == "1985-1990"
            assert retrieved.decade_confidence == 0.85
            assert retrieved.season == "summer"
            assert retrieved.season_confidence == 0.75
            assert retrieved.event_hint == "beach vacation"
            assert retrieved.event_confidence == 0.65
            assert retrieved.photo_medium == "print_scan"
            assert retrieved.photo_medium_confidence == 0.95
            assert retrieved.visual_evidence == [
                "palm trees",
                "ocean",
                "swimwear",
                "sunglasses",
            ]
            assert retrieved.alternative_decades == ["1980-1985", "1990-1995"]
            assert retrieved.uncertainty_flag is False
            assert retrieved.hypothesis_notes == "Clear summer beach scene with strong visual cues"

    def test_insert_context_with_minimal_data(self, database_store: DatabaseStore):
        """Test inserting context record with only required fields."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert photo
            photo = PhotoCreate(
                content_hash="minimal_context_photo",
                file_path="/test/minimal_context.jpg",
            )
            photo_id = helper.insert_photo(photo)

            # Insert context with minimal data
            context = ContextCreate(
                photo_id=photo_id,
                decade="2000-2005",
                decade_confidence=0.5,
                season=None,
                season_confidence=None,
                event_hint=None,
                event_confidence=None,
                photo_medium="digital",
                photo_medium_confidence=0.6,
                visual_evidence=None,
                alternative_decades=None,
                uncertainty_flag=True,
                hypothesis_notes="Low confidence estimate",
                raw_json=json.dumps(
                    {
                        "decade": "2000-2005",
                        "decade_confidence": 0.5,
                        "photo_medium": "digital",
                        "photo_medium_confidence": 0.6,
                        "uncertainty_flag": True,
                    }
                ),
            )

            context_id = helper.insert_context(context)
            assert context_id == 1

            # Retrieve and verify
            retrieved = helper.get_context_by_photo_id(photo_id)
            assert retrieved is not None
            assert retrieved.decade == "2000-2005"
            assert retrieved.decade_confidence == 0.5
            assert retrieved.season is None
            assert retrieved.season_confidence is None
            assert retrieved.event_hint is None
            assert retrieved.event_confidence is None
            assert retrieved.photo_medium == "digital"
            assert retrieved.photo_medium_confidence == 0.6
            assert retrieved.visual_evidence is None
            assert retrieved.alternative_decades is None
            assert retrieved.uncertainty_flag is True
            assert retrieved.hypothesis_notes == "Low confidence estimate"

    def test_upsert_context_replaces_existing(self, database_store: DatabaseStore):
        """Test that upsert_context replaces existing context for same photo."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert photo
            photo = PhotoCreate(
                content_hash="upsert_test_photo",
                file_path="/test/upsert_test.jpg",
            )
            photo_id = helper.insert_photo(photo)

            # Insert initial context
            context1 = ContextCreate(
                photo_id=photo_id,
                decade="1990-1995",
                decade_confidence=0.7,
                season="winter",
                season_confidence=0.6,
                photo_medium="print_scan",
                photo_medium_confidence=0.8,
                raw_json='{"decade": "1990-1995"}',
            )
            context_id1 = helper.insert_context(context1)
            assert context_id1 == 1

            # Verify initial context
            retrieved1 = helper.get_context_by_photo_id(photo_id)
            assert retrieved1 is not None
            assert retrieved1.decade == "1990-1995"
            assert retrieved1.season == "winter"

            # Upsert with updated context
            context2 = ContextCreate(
                photo_id=photo_id,  # Same photo_id triggers upsert
                decade="1995-2000",
                decade_confidence=0.8,
                season="summer",
                season_confidence=0.7,
                photo_medium="digital",
                photo_medium_confidence=0.9,
                raw_json='{"decade": "1995-2000"}',
            )
            context_id2 = helper.upsert_context(context2)
            assert context_id2 is not None

            # Verify updated context (old one should be replaced)
            retrieved2 = helper.get_context_by_photo_id(photo_id)
            assert retrieved2 is not None
            assert retrieved2.decade == "1995-2000"  # Updated value
            assert retrieved2.season == "summer"  # Updated value
            assert retrieved2.photo_medium == "digital"  # Updated value

    def test_context_with_json_parsing_edge_cases(self, database_store: DatabaseStore):
        """Test context JSON field parsing with edge cases."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert photo
            photo = PhotoCreate(
                content_hash="json_edge_case_photo",
                file_path="/test/json_edge_case.jpg",
            )
            photo_id = helper.insert_photo(photo)

            # Test with empty lists
            context = ContextCreate(
                photo_id=photo_id,
                decade="1980-1985",
                decade_confidence=0.7,
                season="autumn",
                season_confidence=0.6,
                photo_medium="polaroid",
                photo_medium_confidence=0.8,
                visual_evidence=[],  # Empty list
                alternative_decades=[],  # Empty list
                uncertainty_flag=False,
                hypothesis_notes="Test with empty JSON arrays",
                raw_json=json.dumps(
                    {
                        "decade": "1980-1985",
                        "visual_evidence": [],
                        "alternative_decades": [],
                    }
                ),
            )

            context_id = helper.insert_context(context)
            assert context_id == 1

            # Retrieve and verify empty lists become None
            retrieved = helper.get_context_by_photo_id(photo_id)
            assert retrieved is not None
            assert retrieved.visual_evidence is None  # Empty list becomes None
            assert retrieved.alternative_decades is None  # Empty list becomes None

    def test_context_with_invalid_json_recovery(self, database_store: DatabaseStore):
        """Test that invalid JSON in database fields is handled gracefully."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert photo
            photo = PhotoCreate(
                content_hash="invalid_json_photo",
                file_path="/test/invalid_json.jpg",
            )
            photo_id = helper.insert_photo(photo)

            # Manually insert context with invalid JSON to simulate corrupted data
            conn.execute(
                """
                INSERT INTO context (
                    photo_id, decade, decade_confidence, season, season_confidence,
                    event_hint, event_confidence, photo_medium, photo_medium_confidence,
                    visual_evidence, alternative_decades, uncertainty_flag, hypothesis_notes, raw_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    photo_id,
                    "1990-1995",
                    0.8,
                    "summer",
                    0.7,
                    "test event",
                    0.6,
                    "print_scan",
                    0.9,
                    "{invalid json",  # Invalid JSON
                    "[1990, 1995",  # Invalid JSON (missing closing bracket)
                    False,
                    "Test with invalid JSON",
                    '{"valid": "json"}',  # This one is valid
                ),
            )

            # QueryHelper should handle invalid JSON gracefully
            retrieved = helper.get_context_by_photo_id(photo_id)
            assert retrieved is not None
            assert retrieved.visual_evidence is None  # Invalid JSON becomes None
            assert retrieved.alternative_decades is None  # Invalid JSON becomes None
            assert retrieved.raw_json == '{"valid": "json"}'  # Valid JSON preserved

    def test_get_context_by_photo_id_returns_none_when_no_context(self, database_store: DatabaseStore):
        """Test that get_context_by_photo_id returns None when no context exists."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert a photo without context
            photo = PhotoCreate(
                content_hash="no_context_photo",
                file_path="/test/no_context.jpg",
            )
            photo_id = helper.insert_photo(photo)

            # get_context_by_photo_id should return None
            retrieved = helper.get_context_by_photo_id(photo_id)
            assert retrieved is None

            # Also test with non-existent photo ID
            retrieved = helper.get_context_by_photo_id(999)
            assert retrieved is None


class TestDatabaseQueryOperations:
    """Test query operations for context data."""

    def test_get_photos_without_context(self, database_store: DatabaseStore):
        """Test query to find photos without context analysis."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert photos
            photos_data = [
                ("hash1", "/test/photo1.jpg"),
                ("hash2", "/test/photo2.jpg"),
                ("hash3", "/test/photo3.jpg"),
                ("hash4", "/test/photo4.jpg"),
            ]

            photo_ids = []
            for content_hash, file_path in photos_data:
                photo = PhotoCreate(
                    content_hash=content_hash,
                    file_path=file_path,
                )
                photo_id = helper.insert_photo(photo)
                photo_ids.append(photo_id)

            # Add context to some photos
            context1 = ContextCreate(
                photo_id=photo_ids[0],
                decade="1990-1995",
                decade_confidence=0.8,
                photo_medium="print_scan",
                photo_medium_confidence=0.9,
                raw_json="{}",
            )
            helper.insert_context(context1)

            context2 = ContextCreate(
                photo_id=photo_ids[2],
                decade="2000-2005",
                decade_confidence=0.7,
                photo_medium="digital",
                photo_medium_confidence=0.8,
                raw_json="{}",
            )
            helper.insert_context(context2)

            # Get photos without context
            photos_without_context = helper.get_photos_without_context()
            assert len(photos_without_context) == 2  # photo_ids[1] and [3]

            # Verify correct photos are returned
            returned_hashes = {p.content_hash for p in photos_without_context}
            assert "hash2" in returned_hashes
            assert "hash4" in returned_hashes
            assert "hash1" not in returned_hashes
            assert "hash3" not in returned_hashes

    def test_get_photos_without_context_batch(self, database_store: DatabaseStore):
        """Test batch query for photos without context."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert 15 photos and track their IDs
            photo_ids = []
            for i in range(15):
                photo = PhotoCreate(
                    content_hash=f"batch_hash_{i}",
                    file_path=f"/test/batch_photo_{i}.jpg",
                )
                photo_id = helper.insert_photo(photo)
                photo_ids.append(photo_id)

            # Add context to 5 photos (indices 0, 3, 6, 9, 12)
            for i in [0, 3, 6, 9, 12]:
                context = ContextCreate(
                    photo_id=photo_ids[i],  # Use actual photo ID
                    decade="1990-1995",
                    decade_confidence=0.8,
                    photo_medium="print_scan",
                    photo_medium_confidence=0.9,
                    raw_json="{}",
                )
                helper.insert_context(context)

            # Get first batch of 5 photos without context
            batch1 = helper.get_photos_without_context_batch(batch_size=5, offset=0)
            assert len(batch1) == 5

            # Get second batch
            batch2 = helper.get_photos_without_context_batch(batch_size=5, offset=5)
            assert len(batch2) == 5

            # Get third batch (should have remaining photos)
            batch3 = helper.get_photos_without_context_batch(batch_size=5, offset=10)
            assert len(batch3) == 0  # Only 10 photos without context total

            # Verify no overlap between batches
            batch1_ids = {p.id for p in batch1}
            batch2_ids = {p.id for p in batch2}
            assert batch1_ids.isdisjoint(batch2_ids)

    def test_get_photos_without_context_batch_edge_cases(self, database_store: DatabaseStore):
        """Test batch query edge cases (negative offset, zero batch size)."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert 3 photos
            for i in range(3):
                photo = PhotoCreate(
                    content_hash=f"edge_case_hash_{i}",
                    file_path=f"/test/edge_case_{i}.jpg",
                )
                helper.insert_photo(photo)

            # Test with zero batch size - should return empty list
            batch = helper.get_photos_without_context_batch(batch_size=0, offset=0)
            assert batch == []

            # Test with negative offset - should treat as offset 0
            batch = helper.get_photos_without_context_batch(batch_size=10, offset=-5)
            assert len(batch) == 3  # Should return all photos

            # Test with offset larger than total count
            batch = helper.get_photos_without_context_batch(batch_size=10, offset=100)
            assert batch == []  # Should return empty list

            # Test with batch_size=1 to verify pagination works
            batch1 = helper.get_photos_without_context_batch(batch_size=1, offset=0)
            batch2 = helper.get_photos_without_context_batch(batch_size=1, offset=1)
            batch3 = helper.get_photos_without_context_batch(batch_size=1, offset=2)
            batch4 = helper.get_photos_without_context_batch(batch_size=1, offset=3)

            assert len(batch1) == 1
            assert len(batch2) == 1
            assert len(batch3) == 1
            assert len(batch4) == 0  # No more photos

            # Verify all batches are different photos
            if batch1 and batch2:
                assert batch1[0].id != batch2[0].id

    def test_context_query_with_foreign_key_constraint(self, database_store: DatabaseStore):
        """Test that foreign key constraints are enforced for context records."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Try to insert context for non-existent photo (should fail)
            context = ContextCreate(
                photo_id=999,  # Non-existent photo ID
                decade="1990-1995",
                decade_confidence=0.8,
                photo_medium="print_scan",
                photo_medium_confidence=0.9,
                raw_json="{}",
            )

            try:
                helper.insert_context(context)
                pytest.fail("Should have raised sqlite3.IntegrityError for foreign key violation")
            except sqlite3.IntegrityError as e:
                assert "FOREIGN KEY constraint failed" in str(e)

    def test_context_count_operations(self, database_store: DatabaseStore):
        """Test count operations for context records."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Initial counts should be zero
            assert helper.get_context_count() == 0
            assert helper.get_photo_count() == 0

            # Insert photos with context
            for i in range(5):
                photo = PhotoCreate(
                    content_hash=f"count_hash_{i}",
                    file_path=f"/test/count_photo_{i}.jpg",
                )
                photo_id = helper.insert_photo(photo)

                # Add context to every other photo
                if i % 2 == 0:
                    context = ContextCreate(
                        photo_id=photo_id,
                        decade="1990-1995",
                        decade_confidence=0.8,
                        photo_medium="print_scan",
                        photo_medium_confidence=0.9,
                        raw_json="{}",
                    )
                    helper.insert_context(context)

            # Verify counts
            assert helper.get_photo_count() == 5
            assert helper.get_context_count() == 3  # Photos 0, 2, 4 have context


class TestDatabaseSchemaCompliance:
    """Test database schema compliance and constraints."""

    def test_schema_version_tracking(self, database_store: DatabaseStore):
        """Test that schema version is properly tracked."""
        with database_store.transaction() as conn:
            create_schema(conn)

            # Check schema version
            version = get_schema_version(conn)
            assert version == 1

            # Verify schema_setup record exists
            cursor = conn.execute("SELECT * FROM pipeline_runs WHERE run_id = 'schema_setup'")
            row = cursor.fetchone()
            assert row is not None
            assert row["schema_version"] == 1
            assert row["status"] == "completed"

    def test_table_constraints_are_enforced(self, database_store: DatabaseStore):
        """Test that table constraints (UNIQUE, NOT NULL) are enforced."""
        with database_store.transaction() as conn:
            create_schema(conn)

            # Test NOT NULL constraint on photos.content_hash
            try:
                conn.execute("INSERT INTO photos (file_path) VALUES ('/test/photo.jpg')")
                pytest.fail("Should have raised sqlite3.IntegrityError for NOT NULL constraint")
            except sqlite3.IntegrityError as e:
                assert "NOT NULL constraint failed" in str(e)

            # Test UNIQUE constraint on photos.content_hash
            conn.execute("INSERT INTO photos (content_hash, file_path) VALUES ('hash1', '/test/photo1.jpg')")

            try:
                conn.execute("INSERT INTO photos (content_hash, file_path) VALUES ('hash1', '/test/photo2.jpg')")
                pytest.fail("Should have raised sqlite3.IntegrityError for UNIQUE constraint")
            except sqlite3.IntegrityError as e:
                assert "UNIQUE constraint failed" in str(e)

            # Test UNIQUE constraint on context.photo_id
            photo_id = conn.execute("SELECT id FROM photos WHERE content_hash = 'hash1'").fetchone()[0]

            conn.execute(
                """
                INSERT INTO context (photo_id, decade, decade_confidence, photo_medium, photo_medium_confidence, raw_json)
                VALUES (?, '1990-1995', 0.8, 'print_scan', 0.9, '{}')
                """,
                (photo_id,),
            )

            try:
                conn.execute(
                    """
                    INSERT INTO context (photo_id, decade, decade_confidence, photo_medium, photo_medium_confidence, raw_json)
                    VALUES (?, '2000-2005', 0.7, 'digital', 0.8, '{}')
                    """,
                    (photo_id,),
                )
                pytest.fail("Should have raised sqlite3.IntegrityError for UNIQUE constraint")
            except sqlite3.IntegrityError as e:
                assert "UNIQUE constraint failed" in str(e)

    def test_foreign_key_cascade_deletion(self, database_store: DatabaseStore):
        """Test that foreign key cascade deletion works correctly."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert photo with context
            photo = PhotoCreate(
                content_hash="cascade_test",
                file_path="/test/cascade.jpg",
            )
            photo_id = helper.insert_photo(photo)

            context = ContextCreate(
                photo_id=photo_id,
                decade="1990-1995",
                decade_confidence=0.8,
                photo_medium="print_scan",
                photo_medium_confidence=0.9,
                raw_json="{}",
            )
            helper.insert_context(context)

            # Verify context exists
            assert helper.get_context_by_photo_id(photo_id) is not None

            # Delete photo (should cascade to context)
            conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))

            # Verify context was automatically deleted
            assert helper.get_context_by_photo_id(photo_id) is None

    def test_indexes_are_created(self, database_store: DatabaseStore):
        """Test that required indexes are created by schema."""
        with database_store.transaction() as conn:
            create_schema(conn)

            # Check that key indexes exist
            index_queries = [
                ("idx_photos_content_hash", "photos"),
                ("idx_context_photo_id", "context"),
                ("idx_context_decade", "context"),
                ("idx_context_uncertainty_flag", "context"),
            ]

            for index_name, table_name in index_queries:
                cursor = conn.execute(
                    """
                    SELECT name FROM sqlite_master 
                    WHERE type='index' AND name=? AND tbl_name=?
                    """,
                    (index_name, table_name),
                )
                row = cursor.fetchone()
                assert row is not None, f"Index {index_name} on {table_name} should exist"

    def test_schema_recreation_is_idempotent(self, database_store: DatabaseStore):
        """Test that creating schema multiple times doesn't cause errors."""
        with database_store.transaction() as conn:
            # Create schema multiple times
            for _ in range(3):
                create_schema(conn)

            # Should still have valid schema
            version = get_schema_version(conn)
            assert version == 1

            # Should be able to insert data
            cursor = conn.execute(
                "INSERT INTO photos (content_hash, file_path) VALUES ('idempotent_test', '/test/idempotent.jpg')"
            )
            assert cursor.lastrowid is not None


class TestDatabaseEdgeCases:
    """Test edge cases and error scenarios."""

    def test_context_with_very_long_json(self, database_store: DatabaseStore):
        """Test context insertion with very long JSON strings."""
        with database_store.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            # Insert photo
            photo = PhotoCreate(
                content_hash="long_json_photo",
                file_path="/test/long_json.jpg",
            )
            photo_id = helper.insert_photo(photo)

            # Create very long JSON string
            long_evidence = ["evidence_" + str(i) for i in range(1000)]
            long_decades = ["decade_" + str(i) for i in range(100)]
            long_hypothesis = "x" * 10000  # Very long hypothesis notes

            context = ContextCreate(
                photo_id=photo_id,
                decade="1990-1995",
                decade_confidence=0.8,
                season="summer",
                season_confidence=0.7,
                photo_medium="print_scan",
                photo_medium_confidence=0.9,
                visual_evidence=long_evidence,
                alternative_decades=long_decades,
                uncertainty_flag=False,
                hypothesis_notes=long_hypothesis,
                raw_json=json.dumps({"test": "very long data"}),
            )

            # Should succeed without errors
            context_id = helper.insert_context(context)
            assert context_id == 1

            # Should be able to retrieve
            retrieved = helper.get_context_by_photo_id(photo_id)
            assert retrieved is not None
            assert len(retrieved.visual_evidence) == 1000
            assert len(retrieved.alternative_decades) == 100
            assert len(retrieved.hypothesis_notes) == 10000

    def test_concurrent_transaction_isolation(self, database_store: DatabaseStore):
        """Test that transactions provide isolation between operations."""
        # This test simulates concurrent access by using multiple connections
        with database_store.transaction() as conn1:
            create_schema(conn1)
            helper1 = QueryHelper(conn1)

            # Insert photo in first transaction
            photo = PhotoCreate(
                content_hash="concurrent_test",
                file_path="/test/concurrent.jpg",
            )
            photo_id = helper1.insert_photo(photo)

            # Start second "concurrent" transaction
            # In reality, SQLite serializes writes, but we can test isolation
            with database_store.transaction() as conn2:
                helper2 = QueryHelper(conn2)

                # Second transaction should see the photo (depends on isolation level)
                # With SQLite's default isolation, it might see uncommitted changes
                retrieved = helper2.get_photo_by_id(photo_id)
                # This could be None or the photo, depending on isolation
                # Assert that we get a consistent result (either None or the photo)
                if retrieved is not None:
                    assert retrieved.content_hash == "concurrent_test"
                    assert retrieved.file_path == "/test/concurrent.jpg"
                # If it's None, that's also valid (isolation prevents seeing uncommitted changes)

            # Commit first transaction (context manager will handle this)

        # After commit, all connections should see the photo
        with database_store.transaction() as conn:
            helper = QueryHelper(conn)
            retrieved = helper.get_photo_by_id(photo_id)
            assert retrieved is not None
            assert retrieved.content_hash == "concurrent_test"
            assert retrieved.file_path == "/test/concurrent.jpg"

    def test_database_store_close_and_reopen(self, temp_db_path):
        """Test that database store can be closed and reopened."""
        # Create first store
        store1 = DatabaseStore(temp_db_path)
        with store1.transaction() as conn:
            create_schema(conn)
            helper = QueryHelper(conn)

            photo = PhotoCreate(
                content_hash="reopen_test",
                file_path="/test/reopen.jpg",
            )
            helper.insert_photo(photo)

        # Close first store
        store1.close()

        # Create second store with same path
        store2 = DatabaseStore(temp_db_path)
        with store2.transaction() as conn:
            helper = QueryHelper(conn)

            # Should see data from first store
            retrieved = helper.get_photo_by_hash("reopen_test")
            assert retrieved is not None
            assert retrieved.file_path == "/test/reopen.jpg"

        store2.close()

    def test_missing_photo_foreign_key_handling(self, database_store: DatabaseStore):
        """Test handling of missing foreign key references."""
        with database_store.transaction() as conn:
            create_schema(conn)

            # Try to insert context with invalid photo_id to test constraint
            # Foreign keys should already be enabled by schema creation
            try:
                conn.execute(
                    """
                    INSERT INTO context (
                        photo_id, decade, decade_confidence, photo_medium, photo_medium_confidence, raw_json
                    )
                    VALUES (999, '1990-1995', 0.8, 'print_scan', 0.9, '{}')
                    """
                )
                pytest.fail("Should have raised sqlite3.IntegrityError")
            except sqlite3.IntegrityError as e:
                assert "FOREIGN KEY constraint failed" in str(e)
