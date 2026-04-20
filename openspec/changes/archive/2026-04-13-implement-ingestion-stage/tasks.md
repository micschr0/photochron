## 1. Dependencies and Project Setup

- [x] 1.1 Add Pillow (PIL) dependency to pyproject.toml for image reading/manipulation
- [x] 1.2 Add imagehash dependency to pyproject.toml for perceptual hashing (pHash)
- [x] 1.3 Add piexif dependency to pyproject.toml for detailed EXIF parsing
- [x] 1.4 Install dependencies locally and verify imports work

## 2. IngestionStage Class Implementation

- [x] 2.1 Create concrete IngestionStage class in `src/photochron/pipeline/stages/ingestion.py` replacing placeholder
- [x] 2.2 Implement `__init__` method to accept and store configuration
- [x] 2.3 Implement `run` method signature matching PipelineStage abstract base class
- [x] 2.4 Add progress reporting using pipeline's progress interface
- [x] 2.5 Update stage registry in `src/photochron/pipeline/__init__.py` to use real implementation

## 3. Image Reading and Validation

- [x] 3.1 Implement method to scan `config.input_dir` for image files
- [x] 3.2 Add file filtering by extension (jpg, jpeg, png, heic, raw, etc.)
- [x] 3.3 Implement image validation using Pillow to detect corrupt/unsupported files
- [x] 3.4 Add error handling: skip invalid files with warning, continue processing
- [x] 3.5 Extract basic image metadata (width, height, format) from valid images

## 4. Perceptual Hashing

- [x] 4.1 Implement pHash computation using imagehash library
- [x] 4.2 Convert hash to hexadecimal string for storage
- [x] 4.3 Handle edge cases: very small images, monochrome images
- [x] 4.4 Add hash caching to avoid recomputing for same file content

## 5. Image Downsampling

- [x] 5.1 Implement downsampling logic: resize to max 1024px preserving aspect ratio
- [x] 5.2 Create cache directory structure: `{cache_dir}/downsampled/`
- [x] 5.3 Generate unique filename using perceptual hash (e.g., `{hash}.jpg`)
- [x] 5.4 Save downsampled image with appropriate quality settings
- [x] 5.5 Handle cases where image is already smaller than 1024px (save as-is)

## 6. EXIF Metadata Extraction

- [x] 6.1 Implement EXIF extraction using Pillow's built-in EXIF support
- [x] 6.2 Add detailed EXIF parsing using piexif for GPS and camera data
- [x] 6.3 Extract DateTimeOriginal timestamp and convert to ISO 8601 format
- [x] 6.4 Extract GPS coordinates and convert to decimal degrees
- [x] 6.5 Implement fallback to file modification time when EXIF missing
- [x] 6.6 Handle various EXIF tag formats and encoding issues

## 7. Feature Store Integration

- [x] 7.1 Implement database operations for `photos` table using existing store module
- [x] 7.2 Create Pydantic model for ingestion records (if not already in models)
- [x] 7.3 Implement upsert logic: insert new records, update existing ones by hash
- [x] 7.4 Ensure transaction integrity: commit only after successful processing
- [x] 7.5 Add database error handling and retry logic

## 8. Unit Tests

- [x] 8.1 Create test file `tests/test_pipeline/test_ingestion.py`
- [x] 8.2 Write tests for image reading and validation with mock files
- [x] 8.3 Write tests for perceptual hash computation
- [x] 8.4 Write tests for downsampling logic (dimension calculations)
- [x] 8.5 Write tests for EXIF extraction with sample EXIF data
- [x] 8.6 Write tests for database operations using test database fixture
- [x] 8.7 Write tests for error handling (corrupt files, missing EXIF)

## 9. Integration Tests

- [x] 9.1 Create integration test that runs full ingestion stage on mock image directory
- [x] 9.2 Test end-to-end flow: scan → hash → downsample → EXIF → store
- [x] 9.3 Verify non-destructive principle: original files unchanged
- [x] 9.4 Test duplicate detection (same hash results in skip/update)
- [x] 9.5 Test with various image formats (JPEG, PNG) using test fixtures

## 10. Configuration and Documentation

- [x] 10.1 Add ingestion-specific configuration options to config.yaml (max_downsample_size, supported_formats)
- [x] 10.2 Update CLAUDE.md documentation with new stage details and usage
- [x] 10.3 Add example configuration snippet for ingestion settings
- [x] 10.4 Update any API documentation for pipeline stages

## 11. Validation and Quality

- [x] 11.1 Run existing test suite to ensure no regressions
- [x] 11.2 Run type checking (mypy) on new code
- [x] 11.3 Run linting (ruff) and fix any issues
- [x] 11.4 Verify test coverage for ingestion module (>80%)
- [x] 11.5 Perform manual test with sample photo directory