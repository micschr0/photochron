## Context

The PhotoChron pipeline currently has a working Ingestion stage (Stage 1) that reads photos, computes perceptual hashes, downsamples images, extracts EXIF metadata, and stores records in the `photos` table. Stage 2 (Face Layer) is a placeholder that must be implemented to detect faces, compute embeddings, estimate ages, and match faces to known persons. The pipeline follows a strict 6‑stage architecture where stages communicate only via the SQLite feature store. The `faces` table schema already exists, so no database migrations are required. The system must remain fully on‑device (no external HTTP APIs) and preserve the non‑destructive principle (original files are never modified).

## Goals / Non-Goals

**Goals:**
- Detect faces in downsampled photos (max 1024 px) with bounding boxes and confidence scores
- Compute 512‑dimensional face embeddings suitable for person matching and clustering
- Estimate age for each detected face with confidence intervals (mean ± standard deviation)
- Match faces to known persons (from `persons` table) or mark as unknown
- Store all results in the `faces` table with appropriate foreign‑key relationships
- Integrate seamlessly with the pipeline’s progress tracking and dependency system (depends on `ingestion`)
- Provide configurable confidence thresholds for detection, age estimation, and matching
- Run entirely on‑device using InsightFace models with optional GPU acceleration

**Non-Goals:**
- Real‑time face detection (batch processing is sufficient)
- Training custom models (use pretrained InsightFace models)
- Face recognition across large‑scale databases (matching is limited to known persons in `persons` table)
- Handling of video files or live camera feeds (only static images)
- Modification of original image files (read‑only access to downsampled versions)

## Decisions

1. **Model selection: InsightFace with ONNX runtime**
   - **Why**: InsightFace provides state‑of‑the‑art face detection, embedding, and age estimation in a single, well‑maintained Python package. The ONNX runtime offers cross‑platform compatibility and decent CPU performance; GPU acceleration is optional via CUDA/cuDNN.
   - **Alternatives considered**: `face_recognition` (dlib) is easier to install but lacks age estimation. MediaPipe offers face detection but not embeddings. Training a custom model is out of scope.
   - **Model variant**: Use `buffalo_l` (the largest publicly available InsightFace model) for best accuracy. It provides detection, 512‑dim embedding, and age/gender estimation.

2. **Backend: ONNX Runtime (CPU‑first, optional GPU)**
   - **Why**: ONNX models are portable and can run on CPU with acceptable speed. If a GPU is available, the same model can be accelerated with CUDA provider.
   - **Alternatives considered**: MXNet backend is more performant on GPU but adds extra dependency and installation complexity. ONNX strikes a balance between ease of deployment and performance.

3. **Person matching: cosine similarity with threshold**
   - **Why**: For each detected face, compute its embedding and compare with embeddings of known persons (stored in a future stage). Use cosine similarity and a configurable threshold (default 0.6) to assign a `person_id` or leave as `NULL` (unknown).
   - **Alternatives considered**: Euclidean distance, SVM classifier, or k‑NN. Cosine similarity is standard for face embeddings and computationally cheap.

4. **Age estimation: use InsightFace’s built‑in age head**
   - **Why**: The selected `buffalo_l` model includes an age estimation head that outputs a continuous value (years). We’ll also compute a confidence‑based standard deviation (fixed for now, could be derived from model uncertainty later).
   - **Alternatives considered**: Separate age‑estimation models (DEX, etc.) would increase complexity. InsightFace’s integrated age estimation is sufficient for relative chronological ordering.

5. **Batch processing: process one photo at a time, but batch faces within a photo**
   - **Why**: The pipeline processes photos sequentially, but InsightFace can detect multiple faces in a single image forward pass. We’ll load the model once per stage run and reuse it across all photos.
   - **Alternatives considered**: Batch multiple photos together could improve GPU utilization but complicates error handling and progress tracking. Sequential per‑photo processing aligns with the pipeline’s incremental design.

6. **Configuration: separate `face` section in config.yaml**
   - **Why**: Centralized configuration for model path, detection threshold, age confidence, matching threshold, and batch size. Allows users to tune performance/accuracy trade‑offs without code changes.
   - **Example**:
     ```yaml
     face:
       model_name: "buffalo_l"
       detection_threshold: 0.5
       age_confidence_scale: 0.1
       matching_threshold: 0.6
       use_gpu: false
     ```

7. **Error handling: skip photos where face detection fails**
   - **Why**: If an image cannot be processed (corrupt, no faces, model error), log a warning and continue to the next photo. The pipeline must be robust to partial failures.
   - **Alternatives considered**: Halt the entire stage – too strict. Retry mechanisms – overkill for a local batch process.

## Risks / Trade-offs

- **Performance risk**: Face detection on CPU may be slow for large photo collections (thousands of images). **Mitigation**: Use downsampled images (max 1024 px), optional GPU acceleration, and provide progress feedback.
- **Dependency risk**: InsightFace + ONNX runtime are non‑trivial dependencies that may cause installation issues across different platforms (macOS ARM, Windows, Linux). **Mitigation**: Document installation steps, provide fallback to CPU‑only installation, and consider packaging with Docker later.
- **Accuracy trade‑off**: The chosen model may have bias in age estimation across different ethnicities or lighting conditions. **Mitigation**: Acknowledge the limitation; age estimates are used as relative signals, not absolute truth. Confidence scores reflect uncertainty.
- **Memory usage**: Loading the InsightFace model (≈200 MB) may strain memory‑constrained environments. **Mitigation**: Load model once per pipeline run, not per photo. Provide configuration to use smaller model variants (`buffalo_s`).
- **Privacy concern**: Face embeddings are biometric data. **Mitigation**: All data stays local (SQLite database). No external transmission. Users can delete the database at any time.