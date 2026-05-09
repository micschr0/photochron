"""
SQL schema definition for photochron Feature Store.
"""

import sqlite3

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Photos table: stores metadata about ingested photos
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content_hash TEXT NOT NULL UNIQUE,  -- MD5 hash of file content
    file_path TEXT NOT NULL,            -- Original file path
    downsample_path TEXT,               -- Path to downsampled thumbnail
    exif_datetime TEXT,                 -- EXIF DateTimeOriginal if present
    make TEXT,                          -- Camera make
    model TEXT,                         -- Camera model
    perceptual_hash TEXT,               -- Perceptual hash for near-duplicate detection
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Persons table: known persons from anchors.yaml + user-assigned clusters
CREATE TABLE IF NOT EXISTS persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL UNIQUE,     -- User-defined ID (e.g., 'person_mama')
    name TEXT NOT NULL,                 -- Display name
    birthday TEXT,                      -- YYYY-MM-DD format
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Faces table: face detections and age estimates
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,
    person_id INTEGER,                  -- NULL for unknown faces
    embedding BLOB,                     -- Face embedding vector (biometric data)
    age_estimate REAL,                  -- Estimated age in years
    age_std REAL,                       -- Standard deviation of age estimate
    confidence REAL NOT NULL,           -- Detection confidence 0.0-1.0
    bbox_x1 REAL NOT NULL,              -- Bounding box coordinates
    bbox_y1 REAL NOT NULL,
    bbox_x2 REAL NOT NULL,
    bbox_y2 REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE,
    FOREIGN KEY (person_id) REFERENCES persons (id) ON DELETE SET NULL
);

-- Context table: LLM analysis results
CREATE TABLE IF NOT EXISTS context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL UNIQUE,   -- One context per photo
    decade TEXT,                        -- Estimated decade range (e.g., '1985-1990')
    decade_confidence REAL NOT NULL,    -- Confidence 0.0-1.0
    season TEXT,                        -- 'spring', 'summer', 'autumn', 'winter'
    season_confidence REAL,             -- Confidence in season estimate (0.0-1.0)
    event_hint TEXT,                    -- Event hint from LLM
    event_confidence REAL,              -- Confidence in event hint (0.0-1.0)
    photo_medium TEXT NOT NULL,         -- 'print_scan', 'digital', 'polaroid', etc.
    photo_medium_confidence REAL,       -- Confidence in photo medium estimate (0.0-1.0)
    visual_evidence TEXT,               -- JSON list of visual cues that informed analysis
    alternative_decades TEXT,           -- JSON list of alternative decade possibilities
    uncertainty_flag BOOLEAN,           -- Flag indicating high uncertainty in analysis
    hypothesis_notes TEXT,              -- Explanation when multiple hypotheses exist
    raw_json TEXT NOT NULL,             -- Full LLM response JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
);

-- Rankings table: final chronological ranking
CREATE TABLE IF NOT EXISTS rankings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL UNIQUE,
    sort_rank INTEGER NOT NULL,         -- Final chronological rank (0-based)
    estimated_year INTEGER,             -- Estimated year
    estimated_month INTEGER,            -- Estimated month if known (1-12)
    confidence REAL NOT NULL,           -- Overall confidence 0.0-1.0
    review_needed BOOLEAN DEFAULT FALSE, -- Flag for low confidence results
    ranking_json TEXT NOT NULL,         -- Full ranking details JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (photo_id) REFERENCES photos (id) ON DELETE CASCADE
);

-- Anchor constraints table: serialized ConstraintSet per pipeline run
CREATE TABLE IF NOT EXISTS anchor_constraints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,       -- One constraint set per run
    source_path TEXT,                   -- Path to anchors.yaml
    constraints_json TEXT NOT NULL,     -- Serialized ConstraintSet
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline runs table: tracks pipeline execution history
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,        -- Unique identifier for this run
    schema_version INTEGER NOT NULL DEFAULT 1,
    config_hash TEXT NOT NULL,          -- Hash of config used
    insightface_version TEXT,           -- Model versions for cache invalidation
    ollama_version TEXT,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    status TEXT NOT NULL,               -- 'running', 'completed', 'failed'
    photos_processed INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indices for frequent lookups
CREATE INDEX IF NOT EXISTS idx_photos_content_hash ON photos (content_hash);
CREATE INDEX IF NOT EXISTS idx_photos_exif_datetime ON photos (exif_datetime);

CREATE INDEX IF NOT EXISTS idx_persons_person_id ON persons (person_id);

CREATE INDEX IF NOT EXISTS idx_faces_photo_id ON faces (photo_id);
CREATE INDEX IF NOT EXISTS idx_faces_person_id ON faces (person_id);
CREATE INDEX IF NOT EXISTS idx_faces_confidence ON faces (confidence);

CREATE INDEX IF NOT EXISTS idx_context_photo_id ON context (photo_id);
CREATE INDEX IF NOT EXISTS idx_context_decade ON context (decade);
CREATE INDEX IF NOT EXISTS idx_context_season_confidence ON context (season_confidence);
CREATE INDEX IF NOT EXISTS idx_context_event_confidence ON context (event_confidence);
CREATE INDEX IF NOT EXISTS idx_context_photo_medium_confidence ON context (photo_medium_confidence);
CREATE INDEX IF NOT EXISTS idx_context_uncertainty_flag ON context (uncertainty_flag);

CREATE INDEX IF NOT EXISTS idx_rankings_photo_id ON rankings (photo_id);
CREATE INDEX IF NOT EXISTS idx_rankings_sort_rank ON rankings (sort_rank);
CREATE INDEX IF NOT EXISTS idx_rankings_confidence ON rankings (confidence);

CREATE INDEX IF NOT EXISTS idx_anchor_constraints_run_id ON anchor_constraints (run_id);

CREATE INDEX IF NOT EXISTS idx_pipeline_runs_run_id ON pipeline_runs (run_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_start_time ON pipeline_runs (start_time);
"""


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all database tables if they don't exist."""
    conn.executescript(SCHEMA_SQL)
    # Record schema version
    conn.execute(
        "INSERT OR REPLACE INTO pipeline_runs (run_id, schema_version, config_hash, start_time, status) "
        "VALUES ('schema_setup', ?, '', CURRENT_TIMESTAMP, 'completed')",
        (SCHEMA_VERSION,),
    )


def get_schema_version(conn: sqlite3.Connection) -> int:
    """Get current schema version from pipeline_runs table."""
    try:
        cursor = conn.execute(
            "SELECT schema_version FROM pipeline_runs WHERE run_id = 'schema_setup' ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Migrate database schema to current version if needed."""
    current_version = get_schema_version(conn)

    if current_version == SCHEMA_VERSION:
        return  # Already up to date

    # Migration logic for future versions
    # For now, just recreate schema if version mismatch
    if current_version == 0:
        # Fresh database, create schema
        create_schema(conn)
    else:
        # Future migrations would go here
        # For now, we'll recreate (in development)
        # In production, we would apply incremental migrations
        pass

    # Update to current version
    conn.execute(
        "UPDATE pipeline_runs SET schema_version = ? WHERE run_id = 'schema_setup'",
        (SCHEMA_VERSION,),
    )
