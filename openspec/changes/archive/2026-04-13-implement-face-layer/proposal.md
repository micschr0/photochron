## Why

The PhotoChron pipeline's second stage (Face Layer) is critical for establishing photo chronology through people analysis. Currently, the pipeline can ingest photos and extract basic metadata, but cannot detect faces, estimate ages, or compute embeddings needed for person clustering. Without this capability, the system cannot leverage the most important chronological signal: the appearance and aging of people across photos. This change implements the face analysis stage to enable person‑based dating and clustering.

## What Changes

- **Add InsightFace dependency** for face detection, embedding extraction, and age estimation (on‑device, no external APIs)
- **Implement FaceLayerStage** that processes downsampled images from the ingestion stage
- **Detect faces** in each photo, compute bounding boxes and confidence scores
- **Extract face embeddings** (512‑dim vectors) for person matching and clustering
- **Estimate age** for each detected face with confidence intervals
- **Match faces to known persons** (from `persons` table) or mark as unknown
- **Store results** in the existing `faces` table with foreign‑key relationships
- **Add configuration options** for model selection, confidence thresholds, and batch processing
- **Integrate with pipeline progress tracking** and dependency system (depends on `ingestion` stage)

## Capabilities

### New Capabilities

- `face-analysis`: Detect faces in photos, compute embeddings, estimate ages, and match to known persons. This capability covers the entire face‑processing workflow required by Stage 2 of the PhotoChron pipeline.

### Modified Capabilities

<!-- No existing capabilities are being modified at the requirement level. -->

## Impact

- **Dependencies**: Adds `insightface` Python package (and potentially `onnxruntime` or `mxnet` as backends). This is a significant new dependency but runs entirely on‑device.
- **Code**: New implementation in `src/photochron/pipeline/stages/face_layer.py` replacing the current placeholder. Additional helper modules for model loading, inference, and matching.
- **Database**: Uses the existing `faces` table schema; no schema changes required.
- **Configuration**: New `face` section in `config.yaml` with model paths, confidence thresholds, and batch‑size settings.
- **Performance**: Face detection is computationally intensive; will leverage downsampled images (max 1024 px) and optional GPU acceleration if available.
- **Testing**: Requires unit tests with mock faces, integration tests with sample photos, and verification of confidence‑score integrity.