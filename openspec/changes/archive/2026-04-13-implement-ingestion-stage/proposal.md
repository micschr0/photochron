## Why

The project foundation (architecture, configuration, CLI, feature store, and pipeline skeleton) is complete, but all pipeline stages are currently placeholders. To start processing actual photos through the 6-stage pipeline, we need to implement the first critical stage: Ingestion. This stage reads input photos, computes unique identifiers, extracts metadata, and prepares images for downstream AI processing. Without ingestion, the pipeline cannot accept real-world input, making the entire system unusable.

## What Changes

- Implement the `IngestionStage` class with concrete logic replacing the current placeholder
- Add image reading and validation (supporting JPEG, PNG, HEIC, RAW formats via Pillow)
- Compute perceptual hash (pHash) for duplicate detection and file identification
- Create downsampled versions (max 1024px) for efficient AI processing
- Extract EXIF metadata (timestamp, camera, GPS) for temporal context
- Write all extracted data to the SQLite feature store (`ingestion` table)
- Add comprehensive unit and integration tests with mock images
- Ensure non-destructive operation (original files never modified)

## Capabilities

### New Capabilities
- `photo-ingestion`: Reading input image files, computing perceptual hashes, creating downsampled versions, extracting EXIF metadata, and storing results in the feature store.

### Modified Capabilities
<!-- No existing capabilities are being modified -->

## Impact

**Code Impact:**
- New: `src/photochron/pipeline/stages/ingestion.py` (concrete implementation)
- Modified: `src/photochron/pipeline/__init__.py` (update registry)
- Tests: `tests/test_pipeline/test_ingestion.py` (new test module)

**Dependencies:**
- Pillow (PIL) for image reading and manipulation
- imagehash for perceptual hashing (phash)
- piexif for EXIF extraction (or Pillow's EXIF support)
- No new external services (maintains local-only principle)

**Data Flow:**
- Input: Files from `config.input_dir`
- Output: Records in `ingestion` table (hash, dimensions, timestamp, downsampled_path)
- Downstream: Provides essential data for Face Layer (Stage 2)

**Configuration:**
- May add ingestion-specific settings to `config.yaml` (e.g., max_downsample_size, supported_formats)