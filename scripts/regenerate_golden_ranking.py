#!/usr/bin/env python3
"""
Regenerate the golden ranking fixture from its input definitions.

The golden fixture is a frozen snapshot of the photochron ranking math:
given a fixed set of synthetic signal-tuples, it records the year and
sort-rank the current code produces. Intentional changes to the ranking
algorithm (new weights, new medium priors, tweaked threshold) need to be
paired with a rerun of this script so the expected outputs track reality.

Unintentional changes fail the regression test — that is the point.

Usage::

    python scripts/regenerate_golden_ranking.py

Reads and writes ``tests/fixtures/golden_ranking.json`` in place. The
``expected`` block on each photo is recomputed from the current ranking
code; all other fields (inputs, labels, weights, tolerance) are preserved
so contributors have one editable source of truth.

Review the diff before committing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Allow running the script from a checkout without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from photochron.ranking.estimator import (  # noqa: E402
    combine_signals,
    rank_estimates,
)

GOLDEN_PATH = REPO_ROOT / "tests" / "fixtures" / "golden_ranking.json"


def main() -> int:
    raw = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    weights: dict[str, float] = raw["weights"]
    min_threshold: float = raw["min_confidence_threshold"]

    photos_with_estimates = []
    for photo in raw["photos"]:
        # Drop any stale ``expected`` block so it is fully rewritten below.
        photo.pop("expected", None)
        estimate = combine_signals(
            exif_datetime=photo["exif_datetime"],
            face_year=photo["face_year"],
            face_confidence=photo["face_confidence"],
            decade=photo["decade"],
            decade_confidence=photo["decade_confidence"],
            photo_medium=photo["photo_medium"],
            photo_medium_confidence=photo["photo_medium_confidence"],
            weights=weights,
            min_confidence_threshold=min_threshold,
        )
        photos_with_estimates.append((photo, estimate))

    ranked = rank_estimates([(photo["id"], estimate) for photo, estimate in photos_with_estimates])
    rank_by_id = dict(ranked)

    photos_out = []
    for photo, est in photos_with_estimates:
        photos_out.append(
            {
                **photo,
                "expected": {
                    "year": est.year,
                    "confidence": round(est.confidence, 6),
                    "review_needed": est.review_needed,
                    "sort_rank": rank_by_id[photo["id"]],
                },
            }
        )

    output = {
        "version": raw["version"],
        "weights": weights,
        "min_confidence_threshold": min_threshold,
        "year_tolerance": raw["year_tolerance"],
        "photos": photos_out,
    }

    GOLDEN_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(photos_out)} golden photos to {GOLDEN_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
