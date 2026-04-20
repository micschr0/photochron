## ADDED Requirements

### Requirement: Detect faces in downsampled photos
The system SHALL detect human faces in downsampled images (max 1024 px) using InsightFace.
Each detection MUST include a bounding box (x1, y1, x2, y2) and a confidence score between 0.0 and 1.0.
The system SHALL skip images where no face is detected with confidence above the configured threshold.

#### Scenario: Single face detected
- **WHEN** a downsampled photo contains one clear human face
- **THEN** exactly one face detection is recorded with bounding box and confidence > threshold

#### Scenario: Multiple faces detected
- **WHEN** a downsampled photo contains multiple human faces
- **THEN** each face is detected and recorded with its own bounding box and confidence score

#### Scenario: No faces detected
- **WHEN** a downsampled photo contains no human faces (e.g., landscape)
- **THEN** no face records are created for that photo, and processing continues to the next photo

#### Scenario: Low confidence face skipped
- **WHEN** a potential face is detected with confidence below the configured threshold
- **THEN** the detection is ignored and not stored in the faces table

### Requirement: Compute face embeddings
For each detected face, the system SHALL compute a 512‑dimensional embedding vector that captures biometric identity.
The embedding MUST be normalized (L2‑norm) and stored as a BLOB in the database.
Embeddings MUST be robust to minor variations in pose, lighting, and expression.

#### Scenario: Embedding computed for detected face
- **WHEN** a face is detected with confidence above threshold
- **THEN** a 512‑dimension embedding vector is computed and stored

#### Scenario: Same person yields similar embeddings
- **WHEN** two different photos contain the same person’s face
- **THEN** the cosine similarity between their embeddings is greater than the matching threshold

#### Scenario: Different persons yield dissimilar embeddings
- **WHEN** two photos contain faces of different persons
- **THEN** the cosine similarity between their embeddings is below the matching threshold

### Requirement: Estimate age for each detected face
The system SHALL estimate the age (in years) of each detected face using InsightFace’s age head.
Each estimate MUST include a mean age (real number) and a standard deviation representing uncertainty.
The system SHALL store both values in the faces table.

#### Scenario: Age estimated for adult face
- **WHEN** a detected face appears to be an adult (e.g., 30–50 years)
- **THEN** the age estimate is within ±10 years of the true age (if known)

#### Scenario: Age estimate includes confidence interval
- **WHEN** an age estimate is stored
- **THEN** both `age_estimate` (mean) and `age_std` (standard deviation) fields are populated

#### Scenario: Age estimation works for child faces
- **WHEN** a detected face appears to be a child (e.g., 5–12 years)
- **THEN** the age estimate reflects a younger age range (not mis‑classified as adult)

### Requirement: Match faces to known persons
The system SHALL attempt to match each detected face to a known person in the `persons` table.
Matching MUST use cosine similarity between the face embedding and stored reference embeddings (if available).
If similarity exceeds the configured matching threshold, the face SHALL be assigned that `person_id`; otherwise it SHALL remain `NULL` (unknown).

#### Scenario: Known person matched
- **WHEN** a detected face’s embedding closely matches a known person’s reference embedding
- **THEN** the face record is linked to that person (`person_id` set)

#### Scenario: Unknown person not matched
- **WHEN** a detected face’s embedding does not match any known person above threshold
- **THEN** the face record’s `person_id` remains `NULL`

#### Scenario: Multiple faces matched to same person
- **WHEN** multiple faces in the same or different photos belong to the same known person
- **THEN** all those face records are linked to the same `person_id`

### Requirement: Store face data in feature store
All face detection results SHALL be stored in the existing `faces` table.
Each record MUST include: `photo_id` (foreign key), `person_id` (nullable), `embedding` BLOB, `age_estimate`, `age_std`, `confidence`, bounding box coordinates, and timestamps.
The system SHALL ensure referential integrity with `photos` and `persons` tables.

#### Scenario: New face record inserted
- **WHEN** a face is detected in a photo that previously had no face data
- **THEN** a new row is inserted into the `faces` table with all required fields

#### Scenario: Duplicate detection prevented
- **WHEN** the same face (same photo, same bounding box) is detected again in a later pipeline run
- **THEN** the existing record is updated (upsert) rather than creating a duplicate

#### Scenario: Database transaction integrity
- **WHEN** face processing fails midway through a photo
- **THEN** no partial face data for that photo is committed to the database

### Requirement: Integrate with pipeline framework
The face layer SHALL implement the `PipelineStage` abstract base class.
The stage SHALL declare `ingestion` as a dependency.
The stage SHALL report progress (photos processed, faces detected) through the pipeline’s progress tracking system.

#### Scenario: Stage runs after ingestion
- **WHEN** the pipeline executes with both ingestion and face stages enabled
- **THEN** the face layer runs only after the ingestion stage completes successfully

#### Scenario: Progress reported during processing
- **WHEN** processing a batch of photos
- **THEN** the stage reports incremental progress (e.g., “5/100 photos, 12 faces”)

#### Scenario: Configuration passed correctly
- **WHEN** the face stage is initialized
- **THEN** it receives the `face` configuration section from `config.yaml`

### Requirement: Provide configurable thresholds
The system SHALL allow configuration of detection confidence threshold, age confidence scale, and matching similarity threshold via `config.yaml`.
All thresholds MUST have sensible defaults that work for typical photo collections.

#### Scenario: Detection threshold filters weak detections
- **WHEN** detection threshold is set to 0.7
- **THEN** faces with confidence < 0.7 are ignored

#### Scenario: Matching threshold adjusts person assignment
- **WHEN** matching threshold is set to 0.5
- **THEN** faces with similarity ≥ 0.5 are linked to known persons

#### Scenario: Age confidence scale influences stored uncertainty
- **WHEN** age_confidence_scale is set to 0.15
- **THEN** the stored `age_std` reflects this scaling factor