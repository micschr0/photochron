"""
Unit tests for database store operations.
"""

from datetime import datetime

from photochron.models import (
    ContextCreate,
    FaceCreate,
    PersonCreate,
    PhotoCreate,
    PipelineRunCreate,
)
from photochron.store.queries import QueryHelper
from photochron.store.schema import create_schema, get_schema_version


def test_database_store_creation(database_store):
    """Test DatabaseStore creation and connection."""
    assert database_store.db_path.exists()

    # Test connection property
    conn = database_store.connection
    assert conn is not None

    # Test connection is thread-local
    conn2 = database_store.connection
    assert conn2 is conn  # Same thread, should be same connection


def test_transaction_context_manager(database_store):
    """Test transaction context manager."""
    with database_store.transaction() as conn:
        # Should be able to execute queries
        cursor = conn.execute("SELECT 1")
        result = cursor.fetchone()
        assert result[0] == 1

        # Changes should be committable
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY)")

    # Table should exist after transaction
    with database_store.transaction() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='test'")
        assert cursor.fetchone() is not None


def test_schema_creation(database_store):
    """Test database schema creation."""
    with database_store.transaction() as conn:
        create_schema(conn)

        # Check that tables exist
        cursor = conn.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' 
            AND name IN ('photos', 'faces', 'context', 'rankings', 'pipeline_runs', 'persons')
        """)
        tables = {row[0] for row in cursor.fetchall()}

        assert "photos" in tables
        assert "faces" in tables
        assert "context" in tables
        assert "rankings" in tables
        assert "pipeline_runs" in tables
        assert "persons" in tables

        # Check schema version
        version = get_schema_version(conn)
        assert version == 1


def test_query_helper_photo_operations(database_store):
    """Test QueryHelper photo operations."""
    with database_store.transaction() as conn:
        create_schema(conn)
        helper = QueryHelper(conn)

        # Insert photo
        photo = PhotoCreate(
            content_hash="abc123",
            file_path="/test/photo.jpg",
            downsample_path="/test/thumb.jpg",
            exif_datetime="2020:01:01 12:00:00",
            make="Test Camera",
            model="Test Model",
            perceptual_hash="perceptual123",
        )

        photo_id = helper.insert_photo(photo)
        assert photo_id == 1

        # Retrieve photo by ID
        retrieved = helper.get_photo_by_id(photo_id)
        assert retrieved is not None
        assert retrieved.content_hash == "abc123"
        assert retrieved.file_path == "/test/photo.jpg"
        assert retrieved.make == "Test Camera"

        # Retrieve photo by hash
        by_hash = helper.get_photo_by_hash("abc123")
        assert by_hash is not None
        assert by_hash.id == photo_id

        # Get all photos
        all_photos = helper.get_all_photos()
        assert len(all_photos) == 1
        assert all_photos[0].id == photo_id


def test_query_helper_person_operations(database_store):
    """Test QueryHelper person operations."""
    with database_store.transaction() as conn:
        create_schema(conn)
        helper = QueryHelper(conn)

        # Insert person
        person = PersonCreate(
            person_id="person_mama",
            name="Mama",
            birthday="1983-03-15",
        )

        person_id = helper.insert_person(person)
        assert person_id == 1

        # Retrieve person by ID
        retrieved = helper.get_person_by_id(person_id)
        assert retrieved is not None
        assert retrieved.person_id == "person_mama"
        assert retrieved.name == "Mama"

        # Retrieve person by person_id
        by_person_id = helper.get_person_by_person_id("person_mama")
        assert by_person_id is not None
        assert by_person_id.id == person_id


def test_query_helper_face_operations(database_store):
    """Test QueryHelper face operations."""
    with database_store.transaction() as conn:
        create_schema(conn)
        helper = QueryHelper(conn)

        # First insert a photo
        photo = PhotoCreate(
            content_hash="photo1",
            file_path="/test/photo.jpg",
            downsample_path="/test/thumb.jpg",
        )
        photo_id = helper.insert_photo(photo)

        # Insert person
        person = PersonCreate(
            person_id="person1",
            name="Person One",
            birthday="1990-01-01",
        )
        person_id = helper.insert_person(person)

        # Insert face
        face = FaceCreate(
            photo_id=photo_id,
            person_id=person_id,
            embedding=b"fake_embedding",
            age_estimate=30.0,
            age_std=2.5,
            confidence=0.9,
            bbox_x1=10.0,
            bbox_y1=20.0,
            bbox_x2=110.0,
            bbox_y2=120.0,
        )

        face_id = helper.insert_face(face)
        assert face_id == 1

        # Get faces by photo ID
        faces = helper.get_faces_by_photo_id(photo_id)
        assert len(faces) == 1
        assert faces[0].age_estimate == 30.0
        assert faces[0].confidence == 0.9

        # Get faces by person ID
        person_faces = helper.get_faces_by_person_id(person_id)
        assert len(person_faces) == 1
        assert person_faces[0].photo_id == photo_id


def test_query_helper_context_operations(database_store):
    """Test QueryHelper context operations."""
    with database_store.transaction() as conn:
        create_schema(conn)
        helper = QueryHelper(conn)

        # Insert photo
        photo = PhotoCreate(
            content_hash="photo1",
            file_path="/test/photo.jpg",
        )
        photo_id = helper.insert_photo(photo)

        # Insert context
        context = ContextCreate(
            photo_id=photo_id,
            decade="1990-1995",
            decade_confidence=0.8,
            season="summer",
            season_confidence=0.7,
            event_hint="beach vacation",
            event_confidence=0.6,
            photo_medium="print_scan",
            photo_medium_confidence=0.9,
            visual_evidence=["palm trees", "ocean", "swimwear"],
            alternative_decades=["1985-1990", "1995-2000"],
            uncertainty_flag=False,
            hypothesis_notes="Clear summer beach scene",
            raw_json='{"decade": "1990-1995", "confidence": 0.8}',
        )

        context_id = helper.insert_context(context)
        assert context_id == 1

        # Get context by photo ID
        retrieved = helper.get_context_by_photo_id(photo_id)
        assert retrieved is not None
        assert retrieved.decade == "1990-1995"
        assert retrieved.decade_confidence == 0.8
        assert retrieved.season == "summer"
        assert retrieved.season_confidence == 0.7
        assert retrieved.event_hint == "beach vacation"
        assert retrieved.event_confidence == 0.6
        assert retrieved.photo_medium == "print_scan"
        assert retrieved.photo_medium_confidence == 0.9
        assert retrieved.visual_evidence == ["palm trees", "ocean", "swimwear"]
        assert retrieved.alternative_decades == ["1985-1990", "1995-2000"]
        assert retrieved.uncertainty_flag is False
        assert retrieved.hypothesis_notes == "Clear summer beach scene"


def test_query_helper_pipeline_run_operations(database_store):
    """Test QueryHelper pipeline run operations."""
    with database_store.transaction() as conn:
        create_schema(conn)
        helper = QueryHelper(conn)

        # Insert pipeline run
        run = PipelineRunCreate(
            run_id="test_run_123",
            schema_version=1,
            config_hash="abc123",
            insightface_version="buffalo_l",
            ollama_version="llava-next:7b",
            start_time=datetime.now(),
            status="running",
            photos_processed=0,
        )

        run_id = helper.insert_pipeline_run(run)
        assert run_id > 0

        # Get pipeline run
        retrieved = helper.get_pipeline_run("test_run_123")
        assert retrieved is not None
        assert retrieved.run_id == "test_run_123"
        assert retrieved.status == "running"

        # Update pipeline run
        helper.update_pipeline_run("test_run_123", status="completed", photos_processed=10)

        # Verify update
        updated = helper.get_pipeline_run("test_run_123")
        assert updated.status == "completed"
        assert updated.photos_processed == 10

        # Get latest pipeline run
        latest = helper.get_latest_pipeline_run()
        assert latest is not None
        assert latest.run_id == "test_run_123"


def test_cache_invalidation(database_store):
    """Test cache invalidation helpers."""
    with database_store.transaction() as conn:
        create_schema(conn)
        helper = QueryHelper(conn)

        # Insert photo with dependent data
        photo = PhotoCreate(
            content_hash="photo1",
            file_path="/test/photo.jpg",
        )
        photo_id = helper.insert_photo(photo)

        # Insert face for this photo
        face = FaceCreate(
            photo_id=photo_id,
            confidence=0.9,
            bbox_x1=0,
            bbox_y1=0,
            bbox_x2=100,
            bbox_y2=100,
        )
        helper.insert_face(face)

        # Insert context
        context = ContextCreate(
            photo_id=photo_id,
            decade="1990-1995",
            decade_confidence=0.8,
            season=None,
            season_confidence=None,
            event_hint=None,
            event_confidence=None,
            photo_medium="print_scan",
            photo_medium_confidence=0.8,
            visual_evidence=None,
            alternative_decades=None,
            uncertainty_flag=True,
            hypothesis_notes="Low confidence estimate",
            raw_json="{}",
        )
        helper.insert_context(context)

        # Verify data exists
        assert len(helper.get_faces_by_photo_id(photo_id)) == 1
        assert helper.get_context_by_photo_id(photo_id) is not None

        # Mark photo invalid (should delete dependent data)
        helper.mark_photo_invalid(photo_id)

        # Verify dependent data is gone
        assert len(helper.get_faces_by_photo_id(photo_id)) == 0
        assert helper.get_context_by_photo_id(photo_id) is None


def test_count_helpers(database_store):
    """Test count helper functions."""
    with database_store.transaction() as conn:
        create_schema(conn)
        helper = QueryHelper(conn)

        # Initial counts should be zero
        assert helper.get_photo_count() == 0
        assert helper.get_face_count() == 0
        assert helper.get_context_count() == 0
        assert helper.get_ranking_count() == 0

        # Add some data
        photo = PhotoCreate(
            content_hash="photo1",
            file_path="/test/photo.jpg",
        )
        helper.insert_photo(photo)

        # Counts should update
        assert helper.get_photo_count() == 1
        assert helper.get_face_count() == 0  # No faces yet
