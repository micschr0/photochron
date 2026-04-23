#!/usr/bin/env python3
"""
Benchmark PhotoChron's ingestion stage on a synthetic fixture.

Focus: measure the effect of ``config.ingestion.workers`` (the P1c
ThreadPoolExecutor) on a directory of synthetic JPEGs. The context
layer (Ollama) is not exercised here because it requires opt-in model
configuration, host-specific GPU/Metal behaviour, and a running Ollama
daemon – benchmark that separately once the environment is ready.

Typical use::

    # Generate a 500-photo fixture, benchmark ingestion at workers=1 and 8
    python scripts/gen_bench_fixture.py --count 500 --output bench_fixture --seed 1
    python scripts/bench.py --input bench_fixture --workers 1,2,4,8

Each run starts from a fresh SQLite store so caching between runs does
not skew the numbers. The resulting report lists wall time, per-image
time, and speedup relative to ``workers=1``.

The benchmark runs through the real ``IngestionStage`` code path, so
any future optimisation (SIMD, libjpeg-turbo, etc.) is automatically
reflected. No face or context inference happens – this is purely the
ingestion-layer benchmark.
"""

from __future__ import annotations

import argparse
import shutil
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from photochron.config import get_config  # noqa: E402
from photochron.pipeline import PipelineRunner  # noqa: E402
from photochron.pipeline.stages.ingestion import IngestionStage  # noqa: E402
from photochron.store import get_store  # noqa: E402


@dataclass
class Result:
    workers: int
    total_files: int
    wall_seconds: float

    @property
    def per_image_ms(self) -> float:
        return (self.wall_seconds / self.total_files) * 1000.0 if self.total_files else 0.0

    @property
    def images_per_sec(self) -> float:
        return self.total_files / self.wall_seconds if self.wall_seconds else 0.0


def _reset_cache(cache_dir: Path) -> None:
    """Purge the SQLite store + downsampled thumbnails so each run is cold.

    Without this the ingestion stage's duplicate-detection short-circuits
    everything after the first run and we measure "select and skip" times
    instead of real work.
    """
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _bench_once(input_dir: Path, workers: int) -> Result:
    """Run the ingestion stage once with the requested worker count."""
    import photochron.config as config_module

    # Force-refresh the config singleton so a changed workers value takes
    # effect (get_config() caches by default).
    config_module._config = None  # noqa: SLF001 – deliberate cache bust
    config = get_config()
    config.ingestion.workers = workers  # override (validate_assignment=True)

    cache_dir = Path(config.paths.cache_dir).resolve()
    _reset_cache(cache_dir)

    # Also bust the store singleton so it reopens against the fresh DB.
    import photochron.store as store_module

    store_module._store = None  # noqa: SLF001

    stage = IngestionStage()
    config.input_dir = str(input_dir)

    runner = PipelineRunner()
    run_id = runner.create_run(runner._compute_config_hash())  # noqa: SLF001

    total = sum(1 for p in input_dir.iterdir() if p.is_file())
    start = time.monotonic()
    stage.run(run_id=run_id, config_hash="bench")
    elapsed = time.monotonic() - start

    # Confirm the DB actually got populated.
    store = get_store()
    with store.transaction() as conn:
        rows = conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
    if rows != total:
        print(f"  [warn] ingested {rows}/{total} rows – some files may have failed")

    return Result(workers=workers, total_files=total, wall_seconds=elapsed)


def _print_report(results: list[Result]) -> None:
    if not results:
        print("No results.")
        return
    baseline = next((r for r in results if r.workers == 1), results[0])
    header = f"{'workers':>8}  {'files':>6}  {'wall (s)':>10}  {'per img (ms)':>13}  {'img/sec':>9}  speedup"
    print(header)
    print("-" * len(header))
    for r in results:
        speedup = baseline.wall_seconds / r.wall_seconds if r.wall_seconds else float("nan")
        print(
            f"{r.workers:>8}  {r.total_files:>6}  {r.wall_seconds:>10.2f}  "
            f"{r.per_image_ms:>13.1f}  {r.images_per_sec:>9.1f}  {speedup:>5.2f}×"
        )


def _parse_workers(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory of images to ingest (see scripts/gen_bench_fixture.py)",
    )
    parser.add_argument(
        "--workers",
        type=_parse_workers,
        default=[1, 4],
        help="Comma-separated worker counts to benchmark, e.g. '1,2,4,8' (default: '1,4')",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to run each configuration; the median is reported",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input directory not found: {args.input}", file=sys.stderr)
        return 2

    results: list[Result] = []
    for workers in args.workers:
        samples: list[float] = []
        per_total = 0
        for rep in range(args.repeats):
            print(f"▶ workers={workers} (run {rep + 1}/{args.repeats})")
            r = _bench_once(args.input, workers)
            samples.append(r.wall_seconds)
            per_total = r.total_files
        median = statistics.median(samples)
        results.append(Result(workers=workers, total_files=per_total, wall_seconds=median))

    print()
    _print_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
