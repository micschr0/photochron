Pipeline Architecture

Each stage is independently re-runnable via `photochron rerun --stage <name>`.
All inter-stage communication goes through the SQLite Feature Store – no direct function calls between stages.

Stage 1: Ingestion
Trigger: new run or `--rerun-stage ingestion`
Input: JPEG files in input directory
Key logic:

- MD5 content hash per file (rename-safe cache key)
- Downsample to 1024px longest edge → save to `.photochron/thumbs/`
- Read existing EXIF (DateTimeOriginal, Make, Model)
- Perceptual hash for near-duplicate detection (threshold: 0.95)
- Skip if hash already in `photos` table

Output table: `photos`
Invalidation: content hash change

Stage 2: Face Layer
Trigger: new photos in `photos` without `faces` rows
Model: InsightFace buffalo_l via ONNX Runtime (CoreML EP on Apple Silicon)
Key logic:

- Detect all faces, compute embeddings + age estimate per face
- Person identity: compare embedding to known persons (cosine similarity > threshold)
- Unknown faces go to cluster pool → resolved by user in `cluster` command
- Output: age_estimate (float), age_std (float), person_id (FK or NULL), bbox

Output table: `faces`
Latency: ~100–300ms/image on M3

Stage 3: Context Layer
Trigger: new photos in `photos` without `context` rows
Model: Ollama (llava-next:7b via MLX), fallback moondream2
Key logic:

- Structured JSON prompt – never free-text output
- Extract: decade_estimate, decade_confidence, season, event_hint, photo_medium
- Pass anchor context in prompt when person birthdays are known
- Retry once on JSON parse failure; mark as failed if retry fails

Output table: `context`
Latency: ~2–5s/image (7B model, MLX)

Prompt contract (output schema):
{
"decade": "1985-1990",
"decade_confidence": 0.75,
"season": "summer",
"event_hint": null,
"photo_medium": "print_scan"
}

Stage 4: Anchor Layer
Trigger: runs before Ranking Engine on every run (fast, no inference)
Input: `anchors.yaml`
Key logic:

- Load persons + birthdays → create `AnchorMap` (person_id → birthday)
- Resolve birthday constraints: age_estimate + birthday → estimated_photo_year
- Parse events → create `Constraint` list (type: hard | soft)
- Validate: no contradicting hard constraints; warn on soft conflicts

Output: in-memory `ConstraintSet` passed to Ranking Engine (not persisted separately)

Stage 5: Ranking Engine
Trigger: after Stages 2–4 complete
Key logic:

Step 1 – Weighted date estimate per photo:
estimated_date = weighted_combine(
face_age_estimate  × 0.45,
llm_decade         × 0.30,
photo_medium_prior × 0.10,
exif_date          × 1.00   # overrides all when present
)

Step 2 – Apply constraints (hard first, then soft)

Step 3 – Pairwise LLM comparison for ambiguous pairs
(confidence bands overlap → ask LLM: "Which photo is earlier?")
Cap: max 500 pairs per run.

Step 4 – Topological sort → final `sort_rank` per photo

Output table: `rankings`

Stage 6: Output Layer
Trigger: after Ranking Engine
Two output modes (both active on full run):

Mode A – Renamed copies:
`{output_dir}/renamed/{sort_rank:04d}_{estimated_year}-est_{original_name}.jpg`

Mode B – EXIF-enriched copies:
`{output_dir}/exif_enriched/{original_name}.jpg`
EXIF fields written:

- `DateTimeOriginal`: `{year}:01:01 00:00:00` (month/day if known)
- `ImageDescription`: "Est. 1987 ±2yr – Mama ~4yr, summer, print_scan"
- `UserComment`: full JSON result blob

Additional outputs:

- `{output_dir}/photochron_report.json`
- `{output_dir}/photochron_timeline.csv`