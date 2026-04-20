## 1. Dependencies and Project Setup

- [x] 1.1 Add insightface dependency to pyproject.toml (with optional onnxruntime extras)
- [x] 1.2 Add onnxruntime dependency (CPU version) for cross‑platform compatibility
- [x] 1.3 Install dependencies locally and verify imports work
- [x] 1.4 Download InsightFace buffalo_l model files (or implement automatic download)
- [x] 1.5 Verify model loading works in a minimal test script

## 2. Configuration System

- [x] 2.1 Create Pydantic model for face configuration (ConfigFace) in src/photochron/config/models.py
- [x] 2.2 Add face section to config.yaml with default values (detection_threshold, matching_threshold, age_confidence_scale, use_gpu, model_name)
- [x] 2.3 Update config loading in src/photochron/config/__init__.py to include face config
- [x] 2.4 Add validation for threshold ranges (0.0‑1.0) and sensible defaults

## 3. FaceLayerStage Class Implementation

- [x] 3.1 Update FaceLayerStage placeholder in src/photochron/pipeline/stages/face_layer.py with proper __init__ that stores config
- [x] 3.2 Implement run() method signature matching PipelineStage abstract base class
- [x] 3.3 Add progress reporting using pipeline's progress interface
- [x] 3.4 Implement method to query photos without face data from the database
- [x] 3.5 Add error handling for corrupt images or model failures (skip with warning)
- [x] 3.6 Ensure stage declares correct dependencies (["ingestion"]) in property

## 4. Model Loading and Inference Helper

- [x] 4.1 Create module src/photochron/face/insightface_wrapper.py (or similar) to encapsulate InsightFace usage
- [x] 4.2 Implement model loading (buffalo_l) with configurable backend (CPU/GPU)
- [x] 4.3 Implement detect_faces() function that takes image array and returns detections (bbox, confidence)
- [x] 4.4 Implement compute_embedding() function that extracts 512‑dim embedding from a cropped face
- [x] 4.5 Implement estimate_age() function that returns mean and std deviation
- [x] 4.6 Add batch inference support for multiple faces in one photo
- [x] 4.7 Add model caching (load once per pipeline run, not per photo)

## 5. Face Detection and Processing

- [x] 5.1 Implement method to load downsampled image from disk using downsample_path from photos table
- [x] 5.2 Apply detection threshold to filter low‑confidence faces
- [x] 5.3 For each detected face, crop image using bounding box (with optional margin)
- [x] 5.4 Compute embedding and age estimate for each cropped face
- [x] 5.5 Convert bounding box coordinates to normalized form (0‑1) or store as absolute pixels

## 6. Person Matching Logic

- [x] 6.1 Query known persons from persons table (currently empty, but schema exists)
- [x] 6.2 Implement cosine similarity calculation between face embedding and reference embeddings
- [x] 6.3 Apply matching threshold to decide if face belongs to a known person
- [x] 6.4 Assign person_id or leave as NULL based on match result
- [ ] 6.5 Store reference embeddings for future matching (if persons table gets embedding column later)

## 7. Database Operations

- [x] 7.1 Define Pydantic model for face records (if not already in models)
- [x] 7.2 Implement upsert logic for faces table: insert new records, update existing ones by (photo_id, bounding box hash)
- [x] 7.3 Ensure foreign‑key constraints are satisfied (photo_id references photos.id, person_id references persons.id)
- [x] 7.4 Store embedding as BLOB (serialized numpy array or bytes)
- [x] 7.5 Use database transaction to commit all faces of a photo atomically
- [x] 7.6 Add database error handling and retry logic

## 8. Integration with Pipeline

- [x] 8.1 Verify stage registration via @register_stage decorator works
- [x] 8.2 Test that stage runs after ingestion stage (dependency ordering)
- [x] 8.3 Integrate progress reporting: update after each photo and each batch of faces
- [x] 8.4 Add logging for key events (model loaded, faces detected, errors)
- [x] 8.5 Ensure non‑destructive principle: only read downsampled images, never modify originals

## 9. Unit Tests

- [x] 9.1 Create test file tests/test_pipeline/test_face_layer.py
- [x] 9.2 Write tests for model wrapper with mock detections
- [x] 9.3 Write tests for face detection logic using synthetic images
- [x] 9.4 Write tests for embedding computation and cosine similarity
- [x] 9.5 Write tests for age estimation (mock outputs)
- [x] 9.6 Write tests for person matching logic
- [ ] 9.7 Write tests for database operations using test database fixture
- [ ] 9.8 Write tests for error handling (corrupt images, missing model files)

## 10. Integration Tests

- [x] 10.1 Create integration test that runs full face layer on a small set of test photos with faces
- [x] 10.2 Test end‑to‑end flow: load model → detect faces → compute embeddings → estimate ages → store
- [x] 10.3 Verify that faces table is populated with correct columns and foreign keys
- [x] 10.4 Test duplicate detection handling (re‑running stage on same photos)
- [ ] 10.5 Test configuration changes (thresholds) affect detection results

## 11. Performance and Optimization

- [ ] 11.1 Profile face detection on CPU with typical image sizes (1024px)
- [ ] 11.2 Implement optional GPU acceleration detection and fallback
- [ ] 11.3 Add batch size configuration to balance memory usage and speed
- [ ] 11.4 Consider caching embeddings for known persons to avoid recomputation

## 12. Documentation and Validation

- [ ] 12.1 Update CLAUDE.md with face layer details, configuration options, and usage notes
- [ ] 12.2 Add example configuration snippet for face settings
- [ ] 12.3 Run existing test suite to ensure no regressions
- [ ] 12.4 Run type checking (mypy) on new code
- [ ] 12.5 Run linting (ruff) and fix any issues
- [ ] 12.6 Verify test coverage for face module (>80%)
- [ ] 12.7 Perform manual test with a small directory of portrait photos