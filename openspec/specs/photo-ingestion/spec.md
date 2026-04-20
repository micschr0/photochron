## ADDED Requirements

### Requirement: Read and validate image files
The ingestion stage SHALL read image files from the configured input directory.
The system MUST support common image formats: JPEG, PNG, HEIC, and RAW (via Pillow plugins).
The system SHALL skip files that are not valid images or unsupported formats, logging a warning.

#### Scenario: Valid JPEG file processed
- **WHEN** a valid JPEG file exists in the input directory
- **THEN** the file is successfully read and its dimensions are extracted

#### Scenario: Unsupported file format skipped
- **WHEN** a file with unsupported extension (e.g., .txt, .pdf) exists in the input directory
- **THEN** the file is skipped with a warning logged to console

#### Scenario: Corrupt image file handled gracefully
- **WHEN** a corrupt or unreadable image file exists
- **THEN** the file is skipped with an error logged, processing continues with next file

### Requirement: Compute perceptual hash
The system SHALL compute a 64-bit perceptual hash (pHash) for each valid image.
The hash MUST be robust to resizing, format conversion, and minor modifications.
The system SHALL store the hash as a hexadecimal string in the feature store.

#### Scenario: Hash computed for image
- **WHEN** a valid image is read
- **THEN** a 64-bit perceptual hash is computed and stored

#### Scenario: Identical images produce same hash
- **WHEN** two visually identical images (different compression levels) are processed
- **THEN** both images receive the same perceptual hash value

#### Scenario: Different images produce different hashes
- **WHEN** two visually distinct images are processed
- **THEN** the images receive different perceptual hash values

### Requirement: Create downsampled version
The system SHALL create a downsampled version of each image with maximum dimension of 1024 pixels.
The downsampled image MUST maintain the original aspect ratio.
The downsampled image SHALL be stored in the cache directory under `downsampled/` with a filename derived from the perceptual hash.

#### Scenario: Large image downsampled
- **WHEN** an image with dimensions 4000x3000 pixels is processed
- **THEN** a downsampled version with dimensions 1024x768 pixels is created and saved

#### Scenario: Small image not upscaled
- **WHEN** an image with dimensions 800x600 pixels is processed
- **THEN** the image is saved as-is (no upscaling) in the downsampled directory

#### Scenario: Downsampled file naming
- **WHEN** an image with hash `abc123` is processed
- **THEN** the downsampled file is saved as `{cache_dir}/downsampled/abc123.jpg`

### Requirement: Extract EXIF metadata
The system SHALL extract EXIF metadata from images where available.
The system MUST extract timestamp (DateTimeOriginal), camera model, and GPS coordinates.
The system SHALL convert EXIF timestamps to ISO 8601 format for storage.
If EXIF metadata is unavailable, the system SHALL fall back to file modification time.

#### Scenario: EXIF timestamp extracted
- **WHEN** an image contains EXIF DateTimeOriginal tag
- **THEN** the timestamp is extracted and converted to ISO 8601 format

#### Scenario: GPS coordinates extracted
- **WHEN** an image contains GPS latitude and longitude tags
- **THEN** coordinates are extracted as decimal degrees and stored

#### Scenario: Missing EXIF falls back to file time
- **WHEN** an image contains no EXIF timestamp
- **THEN** the file's modification time is used as the timestamp

### Requirement: Store data in feature store
The system SHALL store all extracted data in the `photos` table of the SQLite feature store.
Each record MUST include: perceptual hash, original dimensions, timestamp, camera model, GPS coordinates (if available), and path to downsampled image.
The system SHALL ensure no duplicate records for the same perceptual hash (upsert operation).

#### Scenario: New image creates record
- **WHEN** a new image (hash not in database) is processed
- **THEN** a new record is inserted into the `ingestion` table

#### Scenario: Duplicate image updates record
- **WHEN** an image with existing hash is processed
- **THEN** the existing record is updated (or skipped) with latest metadata

#### Scenario: Database transaction integrity
- **WHEN** processing fails mid-batch
- **THEN** no partial data is committed to the database

### Requirement: Non-destructive operation
The system SHALL never modify original input files.
The system SHALL only read original files and write to cache/output directories.
Original file permissions, timestamps, and contents MUST remain unchanged.

#### Scenario: Original file unchanged
- **WHEN** an image is processed through ingestion
- **THEN** the original file's content, modification time, and permissions are identical before and after processing

#### Scenario: Cache directory usage
- **WHEN** downsampled images are created
- **THEN** they are stored in the configured cache directory, not the input directory

### Requirement: Integration with pipeline framework
The ingestion stage SHALL implement the `PipelineStage` abstract base class.
The stage SHALL accept configuration from the main pipeline configuration.
The stage SHALL report progress through the pipeline's progress tracking system.

#### Scenario: Stage runs as part of pipeline
- **WHEN** the pipeline executes with ingestion stage enabled
- **THEN** the ingestion stage's `run()` method is called with appropriate context

#### Scenario: Configuration passed correctly
- **WHEN** the ingestion stage is initialized
- **THEN** it receives input_dir, cache_dir, and other relevant configuration values

#### Scenario: Progress reporting
- **WHEN** processing multiple files
- **THEN** the stage reports progress (files processed/total) through the pipeline's progress interface