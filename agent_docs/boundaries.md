Boundaries – What agents MUST, MAY, and MUST NOT do

🚫 Never (hard stops – no exceptions)

- Never modify files in the input directory. Read-only.
- Never call external HTTP APIs for image analysis (no GPT-4o Vision, no AWS Rekognition, etc.).
  All inference is local via InsightFace and Ollama.
- Never produce a result without a confidence score. Every DB row, every output file annotation
  must include a confidence float 0.0–1.0. A result without confidence is invalid.
- Never export or transmit face embeddings outside the local SQLite Feature Store.
  Embeddings are biometric data.
- Never overwrite an existing output file silently. Either version it or raise an error.

⚠️ Ask before doing

- Adding a new Python dependency (update `requirements.txt` and ask for confirmation).
- Changing the SQLite schema (requires migration logic + version bump in `pipeline_runs`).
- Changing `config.yaml` default values (downstream effects on cached results).
- Changing `anchors.yaml` schema (user-facing format – needs changelog entry).
- Switching the default LLM model (affects reproducibility of cached runs).
- Any change that would invalidate existing cache entries.

✅ Always do

- Write all output to `{output_dir}/` – never anywhere else.
- Log which model version and config hash was used in each `pipeline_runs` DB row.
- Propagate confidence scores from every upstream stage to every downstream stage.
- Downsample images to max 1024px longest edge before any inference.
- Validate JPEG integrity after any EXIF write operation.
- Flag photos with `confidence < 0.5` as `review_needed = TRUE` in the DB.
- Write both renamed copies AND exif-enriched copies on a full run (both output modes active).