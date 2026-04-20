## Context

The PhotoChron project has a complete foundation with SQLite feature store, configuration system, CLI, and pipeline skeleton. The pipeline follows a strict 6-stage architecture (Ingestion → Face → Context → Anchor → Ranking → Output) where each stage communicates only through the feature store. Currently, all stages are placeholder implementations that log but perform no real work. The Ingestion stage is the first stage that must process real input files (photos) to enable the rest of the pipeline.

**Current State:**
- PipelineStage abstract base class exists with `run()` method signature
- `IngestionStage` placeholder in `src/photochron/pipeline/stages/ingestion.py` logs "Ingestion stage would run here"
- SQLite feature store has an `ingestion` table ready (created during foundation)
- Configuration includes `input_dir` path where user photos are located
- Non-destructive principle: original files must never be modified

**Constraints:**
- Local-only inference: No external HTTP APIs, all processing on-device
- Heavy dependencies (InsightFace, Ollama/MLX) are mocked for now but will be integrated later
- Must support common image formats: JPEG, PNG, HEIC, RAW (via Pillow)
- Downsampled images stored in `cache_dir` for efficient AI processing
- Perceptual hash (pHash) used for duplicate detection and file identity

## Goals / Non-Goals

**Goals:**
1. Replace the placeholder `IngestionStage` with concrete implementation that reads image files from `config.input_dir`
2. Compute perceptual hash (pHash) for each image to detect duplicates and establish unique identity
3. Create downsampled version (max 1024px on longest side) stored in cache directory
4. Extract EXIF metadata (timestamp, camera model, GPS coordinates) for temporal context
5. Write all extracted data to the `ingestion` table in the SQLite feature store
6. Maintain non-destructive operation: original files untouched, only read
7. Provide comprehensive test coverage with mock images

**Non-Goals:**
1. No facial recognition or AI processing (that's Face Layer, Stage 2)
2. No image enhancement, cropping, or editing
3. No support for video files (photos only)
4. No external API calls or cloud services
5. No modification of existing pipeline architecture or stage communication pattern

## Decisions

**1. Image Processing Library: Pillow (PIL)**
- **Why**: Standard, well-maintained, supports all required formats (JPEG, PNG, HEIC with plugins, RAW basic), includes EXIF extraction capabilities
- **Alternatives considered**: OpenCV (heavy, focused on CV), imageio (simpler but less format support), rawpy (RAW-specific but extra dependency)
- **Rationale**: Pillow balances simplicity, format support, and EXIF access. It's already a common dependency in Python image projects.

**2. Perceptual Hashing: `imagehash` library**
- **Why**: Provides pHash algorithm that's robust to resizing, format conversion, and minor modifications
- **Alternatives considered**: Custom hashing (complex), `dhash` (simpler but less robust), cryptographic hashes (MD5, SHA - not perceptual)
- **Rationale**: pHash is ideal for duplicate detection where images may be resized or recompressed. The library is lightweight and well-tested.

**3. EXIF Extraction: Pillow's built-in EXIF + `piexif` for detailed parsing**
- **Why**: Pillow provides basic EXIF access; `piexif` offers full EXIF tag decoding including GPS coordinates
- **Alternatives considered**: `exif` library (similar to piexif), custom parsing (error-prone)
- **Rationale**: Use Pillow for simple cases, fall back to `piexif` when detailed GPS or camera data needed. Both are pure Python.

**4. Downsampling Strategy: Max dimension 1024px, maintain aspect ratio**
- **Why**: Balances AI processing efficiency with retaining enough detail for face detection
- **Alternatives considered**: Fixed width/height (distortion), multiple resolutions (complexity), no downsampling (performance issues)
- **Rationale**: 1024px is sufficient for InsightFace face detection (per architecture spec) while significantly reducing memory/CPU.

**5. Cache Storage: Subdirectory under `config.cache_dir` (`downsampled/`)**
- **Why**: Keeps downsampled images organized and separate from other cache data
- **Alternatives considered**: Same directory as originals (violates non-destructive), temp files (volatile)
- **Rationale**: Persistent cache allows re-running pipeline without re-downsampling. Follows established cache pattern from foundation.

**6. Error Handling: Skip invalid files with warning, continue processing**
- **Why**: Large photo collections may contain corrupt or unsupported files; pipeline should be resilient
- **Alternatives considered**: Fail-fast (stops entire pipeline), silent skip (user unaware)
- **Rationale**: Warn users about problematic files but continue processing valid ones. Errors logged to console and potentially to a summary report.

**7. Database Schema: Use existing `ingestion` table from foundation**
- **Why**: Schema already defined with appropriate columns (hash, width, height, timestamp, etc.)
- **Alternatives considered**: Modify schema (would break compatibility), separate table (unnecessary)
- **Rationale**: Foundation's schema matches requirements exactly; no changes needed.

## Risks / Trade-offs

**1. [Risk] HEIC and RAW format support may require additional system libraries**
- **Mitigation**: Document requirements (libheif for HEIC, rawpy for RAW) and provide clear error messages. Consider falling back to "supported formats" list if dependencies missing.

**2. [Risk] Perceptual hashing collisions (different images with same hash)**
- **Mitigation**: Use 64-bit pHash (reasonable collision resistance). Supplement with file size and modification time as secondary checks.

**3. [Risk] Large photo collections could overwhelm memory if all loaded at once**
- **Mitigation**: Process files sequentially, not in parallel. Use generators to limit memory footprint.

**4. [Trade-off] Downsampling loses detail that might be needed for very small faces**
- **Acceptance**: Architecture spec defines 1024px as sufficient for InsightFace. If users have extreme close-ups of tiny faces, they can adjust config.

**5. [Trade-off] EXIF extraction incomplete for some proprietary RAW formats**
- **Acceptance**: Extract what's available, log warnings for missing data. Fall back to file modification time as timestamp approximation.

**6. [Risk] Concurrent pipeline runs could conflict on cache files**
- **Mitigation**: Include hash in downsampled filename to make collisions unlikely. Consider file locking if needed later.

**7. [Trade-off] No progress indicator for large collections**
- **Acceptance**: CLI already shows stage progress via Rich. Ingestion stage can provide file count updates.

## Migration Plan

**Deployment Steps:**
1. Add new dependencies to `pyproject.toml`: `pillow`, `imagehash`, `piexif`
2. Implement `IngestionStage` class in `src/photochron/pipeline/stages/ingestion.py`
3. Update stage registry in `src/photochron/pipeline/__init__.py` to use real implementation
4. Write comprehensive tests in `tests/test_pipeline/test_ingestion.py`
5. Run existing test suite to ensure no regressions
6. Update `CLAUDE.md` documentation with new stage details

**Rollback Strategy:**
- Revert to placeholder implementation (existing code already there)
- No data migration required (feature store schema unchanged)
- Downsampled cache files can be deleted manually if needed

## Open Questions

1. Should we include a "dry run" mode that lists files without processing?
   - Could be useful for users to see what will be processed before running full pipeline
   - Consider adding to CLI options in future iteration

2. How to handle duplicate files (same perceptual hash)?
   - Options: Skip duplicates, process but mark as duplicate, create relationship table
   - Initial implementation: Skip with warning message

3. Should we compute multiple hash types (pHash, dHash, color histogram) for robustness?
   - Adds complexity; start with pHash only, expand later if needed

4. How to handle images without EXIF timestamp?
   - Fall back to file modification time, creation time, or user-provided default?
   - Initial: Use file modification time with clear warning.