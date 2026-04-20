Data Formats

SQLite Feature Store: `.photochron/cache.db`

Table: photos
CREATE TABLE photos (
photo_id       TEXT PRIMARY KEY,  -- MD5 content hash
file_path      TEXT NOT NULL,     -- original absolute path
exif_date      TEXT,              -- DateTimeOriginal if present, else NULL
width          INTEGER,
height         INTEGER,
thumb_path     TEXT,              -- path to 1024px downsampled copy
is_duplicate   BOOLEAN DEFAULT 0,
created_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

Table: faces
CREATE TABLE faces (
face_id        TEXT PRIMARY KEY,  -- UUID
photo_id       TEXT REFERENCES photos(photo_id),
person_id      TEXT REFERENCES persons(person_id),  -- NULL if unmatched
age_estimate   REAL,
age_std        REAL,              -- std dev = confidence proxy
embedding      BLOB,             -- 512-dim float32 vector
bbox           TEXT              -- JSON: {"x":0,"y":0,"w":100,"h":100}
);

Table: persons
CREATE TABLE persons (
person_id      TEXT PRIMARY KEY,  -- slug from anchors.yaml
name           TEXT NOT NULL,
birthday       TEXT,             -- ISO date string or NULL
ref_embedding  BLOB             -- representative embedding
);

Table: context
CREATE TABLE context (
photo_id          TEXT PRIMARY KEY REFERENCES photos(photo_id),
decade_estimate   TEXT,
decade_confidence REAL,
season            TEXT,
event_hint        TEXT,
photo_medium      TEXT,
llm_model         TEXT,         -- model tag used for this row
llm_raw_json      TEXT
);

Table: rankings
CREATE TABLE rankings (
photo_id           TEXT PRIMARY KEY REFERENCES photos(photo_id),
estimated_year     INTEGER,
estimated_month    INTEGER,     -- NULL if unknown
confidence         REAL,
confidence_band    TEXT,        -- e.g. "1985-1988"
sort_rank          INTEGER,
review_needed      BOOLEAN DEFAULT 0,
ranking_notes      TEXT         -- JSON: signal breakdown
);

Table: pipeline_runs
CREATE TABLE pipeline_runs (
run_id         TEXT PRIMARY KEY,
started_at     TIMESTAMP,
config_hash    TEXT,
face_model     TEXT,
llm_model      TEXT,
photo_count    INTEGER,
status         TEXT            -- running / complete / failed
);

anchors.yaml Format
persons:

- id: person_mama
  name: "Mama"
  birthday: "1983-03-15"

events:

- name: "Umzug nach Osnabrück"
  date: "1991-08-01"
  type: hard
  photos_after:
    - "IMG_042.jpg"

known_dates:

- file: "Weihnachten_gross.jpg"
  month: 12
  type: soft

EXIF Fields Written (Mode B output)
Field             | Tag ID | Value format
DateTimeOriginal  | 36867  | "YYYY:01:01 00:00:00" (month/day if known)
ImageDescription  | 270    | "Est. 1987 ±2yr – Mama ~4yr, summer, print_scan"
UserComment       | 37510  | JSON blob (full pipeline result)

Always work on copies. Validate JPEG integrity after write.