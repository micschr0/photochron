# FAQ

Common first-day questions, in roughly the order people hit them.

## "Is any of my data uploaded?"

No. All inference runs on-device:

- The vision LLM (`context.primary_model` / `context.fallback_model`)
  is served by a *local* Ollama daemon. The default
  `context.ollama_host` is `http://localhost:11434`.
- Face detection / age estimation uses the InsightFace ONNX model on
  your CPU, CUDA GPU, or Apple Neural Engine, depending on the
  `face.backend` you pick.
- GPS extraction is **opt-in** (`ingestion.extract_gps: false` by
  default) so coordinates never leave EXIF unless you ask.

See [SECURITY.md](../SECURITY.md) for the full privacy posture and the
known caveat about EXIF-enriched copies embedding the per-photo result
JSON.

## "`photochron run` says 'No AI model is configured' — what now?"

Either:

- Run `photochron init` and the wizard will set things up, **or**
- Open `config.yaml` and uncomment the `face.model_name`,
  `context.primary_model`, and `context.fallback_model` entries after
  verifying each model's license for your intended use.

Then run `photochron doctor` to confirm Ollama is reachable and the
named models are installed (`ollama pull <model>`).

## "Ollama isn't running — how do I install it?"

See [docs/ollama-setup.md](./ollama-setup.md) for the installation
walkthrough. Short version: install from https://ollama.com, then `ollama
pull llava-next:7b moondream2`. Run `photochron doctor` to confirm.

## "`photochron doctor` says CoreMLExecutionProvider is missing — am I running on CPU?"

Yes. The official `onnxruntime` wheel for macOS arm64 is CPU-only.
Install a CoreML-enabled wheel to use the Apple Neural Engine:

```bash
pip uninstall onnxruntime
pip install onnxruntime-silicon   # community project — verify the source
```

After installing, re-run `photochron doctor`; the providers list should
include `CoreMLExecutionProvider`.

## "How much memory does this need?"

For a small library (~500 photos) on Apple Silicon, ~4-6 GB of free
unified memory is comfortable. The vision LLM is the dominant cost.
Lower `context.num_ctx` (default 2048) to reduce Metal memory pressure;
on a 16 GB machine, dropping to 1024 is safe.

The pipeline also enforces a soft memory floor
(`context.memory_warning_threshold_mb` / `_critical_threshold_mb`) and
will skip+retry batches when free memory falls below the critical
threshold.

## "Why AGPL? Can I use this commercially?"

The AGPL-3.0-or-later license is intentional: photochron is a personal
tool, not a SaaS substrate. You can use it for any private or commercial
purpose as long as you respect the AGPL terms (most notably: if you
build a service on top of it that exposes a network UI, you must make
the source available to its users).

If AGPL doesn't fit your use-case, open a GitHub Discussion — we are
open to dual-licensing for specific scenarios.

## "Where do I find the chronological output?"

After `photochron run`, look at the configured `paths.output_dir`
(default `./photochron_output`):

- `renamed/` — chronologically-sorted copies with rank-prefixed
  filenames; drop into any photo viewer.
- `exif_enriched/` — original filenames with the estimated date written
  into EXIF `DateTimeOriginal` (Apple Photos / Lightroom / digiKam
  pick this up automatically).
- `photochron_report.json` — per-photo signals, confidence, and a
  `review_needed=true` flag for low-confidence photos.
- `photochron_timeline.csv` — flat timeline for spreadsheets.

Photos flagged `review_needed=true` are good candidates for
`photochron review`, which walks you through them one by one.

## "How do I fix a wrong year that photochron guessed?"

Two options:

1. **Run `photochron review`.** It walks every photo with confidence
   below a threshold (default 0.5) and lets you accept / edit / skip.
   Edits persist into a `review_overrides` table and are applied by the
   ranking engine on the next `photochron run` (you do *not* need to
   re-run the heavy face / context stages — the override is consumed
   purely by stage 5). Overrides outrank every AI signal *and* EXIF,
   by design: the human reviewer's correction is the last word.
2. **Add an anchor.** Open `anchors.yaml` and add a `known_dates` entry
   pinning that file to the right year (and month if you know it).
   Re-run `photochron run` and the ranking engine will honour the
   constraint.

To remove an override you regret, run:

```sql
sqlite3 .photochron/cache.db "DELETE FROM review_overrides WHERE photo_id = 42;"
```

## "Can I re-run only the slow context layer without re-doing ingestion?"

Not yet through `photochron rerun` — that command is a stub. As a
work-around, the per-stage `pipeline_stage_runs` ledger introduced in
SCHEMA_VERSION=2 already supports the semantics; deleting the
`pipeline_stage_runs` row for `context_layer` is enough to force a
re-run on the next invocation. Proper UX lives behind a flag in the
roadmap.

## "What's the cache directory? Can I delete it?"

`.photochron/` (in the cwd by default; configurable via
`paths.cache_dir`). It contains:

- `cache.db` — the SQLite feature store with photo metadata, face
  embeddings, context analyses, rankings, and the run ledger.
- `thumbs/` — downsampled images used by the LLM and face stages.
- `logs/photochron.log` — rotated loguru file sink.

Deleting it is safe but expensive: the next pipeline run will redo all
the inference work. If you only want to clear thumbnails, delete
`thumbs/` and re-run; ingestion will recreate them.
