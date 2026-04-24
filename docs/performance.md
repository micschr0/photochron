# Performance

This document collects the knobs that matter for PhotoChron throughput on
Apple Silicon (the primary supported platform) and how to measure them.

> **Target:** 1000 photos per run should feel responsive on a modern
> M-series Mac. That means ingestion in seconds, face layer in roughly a
> minute, and context layer bounded by Ollama + Metal inference on a
> warm model (the dominant cost).

---

## Tuning knobs

| Layer | Setting | Default | Why it matters |
|---|---|---|---|
| Ingestion | `ingestion.workers` | `4` | Threadpool over image decode + EXIF extraction. Pillow/imagehash/sqlite3 all release the GIL, so the speed-up is near-linear up to ~8 cores. |
| Face | `face.backend` | `auto` | `auto` picks CoreML on arm64 macOS and CPU elsewhere. **The official `onnxruntime` wheel for macOS arm64 ships CPU-only** — to actually use CoreML/ANE you need a wheel that exposes `CoreMLExecutionProvider`, for example the community [`onnxruntime-silicon`](https://github.com/cansik/onnxruntime-silicon) or a source build with `--use_coreml`. Run `photochron doctor` to confirm which providers are available on your host; if CoreML isn't in the list, the face layer silently falls back to CPU. |
| Context | `context.keep_alive` | `"30m"` | Ollama unloads the ~5 GB llava-next weights after ~5 min of idleness; the reload costs ~10–30 s per photo. Holding the model keeps Metal/MLX buffers warm. |
| Context | `context.num_ctx` | `2048` | Smaller context window reduces Metal/MLX memory pressure on 8–16 GB unified-memory machines. |
| Context | `context.num_gpu` | `-1` | Tells Ollama to use "all layers on GPU" (or the equivalent MLX placement in Ollama ≥ 0.19). On Apple Silicon that means all layers in unified memory. |
| Context | `context.model_options` | `{}` | Per-model overrides (e.g. make the lighter `moondream2` use `num_ctx: 1024`). |

See [configuration.md](configuration.md) for the full field list.

## Diagnostic commands

```bash
# Read-only snapshot of what's wired up on this host.
photochron doctor

# Shows the resolved face backend and pipeline cache state.
photochron status
```

`photochron doctor` is the first thing to check after an install. It
reports Python, platform, Apple-Silicon detection, the ONNX Runtime
providers actually available in your `onnxruntime` wheel, the resolved
`face.backend`, opt-in model gaps, and Ollama reachability. No models
are loaded – it is safe to run on a fresh setup before any download.

## Benchmark harness

Two helper scripts live in `scripts/`:

* `scripts/gen_bench_fixture.py` — generates synthetic JPEGs with random
  content, plausible dimensions, and partial EXIF. No faces, no personal
  data — safe to regenerate and check in as small samples if needed.
* `scripts/bench.py` — times the ingestion stage end-to-end against a
  fixture directory, sweeping across a list of `workers` values.

Typical flow:

```bash
# One-off: 500 reproducible photos
python scripts/gen_bench_fixture.py --count 500 --output bench_fixture --seed 1

# Compare workers=1 vs 2 vs 4 vs 8, median of 3 runs each
python scripts/bench.py --input bench_fixture --workers 1,2,4,8 --repeats 3
```

The report looks like:

```
 workers   files    wall (s)   per img (ms)    img/sec  speedup
---------------------------------------------------------------
       1      20        5.28          264.2        3.8   1.00×
       4      20        0.70           35.0       28.6   7.56×
```

The bench runs the real `IngestionStage.run`, so any future perf
improvement (libjpeg-turbo, direct-to-memory resize, SIMD) is picked up
automatically. Fresh SQLite store per worker count — duplicate
detection does not skew the numbers.

### Context-layer benchmarking (not covered by bench.py)

The context layer depends on a running Ollama daemon and opt-in model
configuration, both of which are host-specific. Bench it manually once
Ollama is installed and a model is uncommented in `config.yaml`:

```bash
# Warm the model, then time a full run
ollama run llava-next:7b "hello" > /dev/null
time photochron run --input bench_fixture --output /tmp/pc_out
```

Watch for the heartbeat log lines (`still working, Ns elapsed`) — they
confirm that Ollama is actively generating and not stuck on a reload.

## What to measure in a regression

When touching performance-critical code, capture:

1. `scripts/bench.py --input <fixture> --workers 1,4 --repeats 3` before
   and after the change.
2. `photochron doctor` output (so reviewers can tell what backend was
   resolved on your machine).
3. Wall time and `img/sec` at `workers=1` and `workers=4` — the first
   isolates single-thread speed, the second shows threading overhead.

Flag any regression larger than ~10% or any speedup that comes with a
new dependency in the PR description.
