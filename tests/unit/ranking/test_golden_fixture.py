"""
Golden-fixture regression test for the ranking engine.

Freezes the current behaviour of ``combine_signals`` + ``rank_estimates``
against a committed JSON fixture (``tests/fixtures/golden_ranking.json``)
of 20 synthetic photos. The fixture covers five signal regimes:

* EXIF override (4 photos) — confidence should always be 1.0
* Full stack face+llm+medium (4 photos)
* Face-only (4 photos, all marked review_needed under default threshold)
* Llm+medium without face (4 photos)
* Medium-only and the all-None case (4 photos) — edge cases

Any accidental change to weights, medium priors, decade-midpoint math,
or sort tie-breaking trips this test. Intentional changes need a
paired run of ``scripts/regenerate_golden_ranking.py`` and a reviewer
who explicitly OKs the new snapshot.

Per-photo year tolerance is ``year_tolerance`` from the fixture (default
±2) so small floating-point reshuffles don't flap the test; ranks are
asserted exactly because ``rank_estimates`` produces a deterministic
permutation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from photochron.ranking.estimator import combine_signals, rank_estimates

GOLDEN_PATH = Path(__file__).resolve().parents[2] / "fixtures" / "golden_ranking.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    data: dict = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return data


def test_golden_fixture_has_expected_shape(golden: dict) -> None:
    """Sanity-check the fixture structure so a bad regen gets caught early."""
    assert golden["version"] == 1
    assert set(golden["weights"]) == {"face", "llm", "medium"}
    assert golden["year_tolerance"] >= 0
    assert len(golden["photos"]) == 20, "Golden fixture must keep 20 photos"
    for photo in golden["photos"]:
        assert "expected" in photo, f"photo {photo['id']} missing expected block"
        expected = photo["expected"]
        assert set(expected) == {"year", "confidence", "review_needed", "sort_rank"}


def test_golden_fixture_per_photo_estimates(golden: dict) -> None:
    """Each photo's year / confidence / review flag matches the snapshot."""
    weights: dict[str, float] = golden["weights"]
    min_threshold: float = golden["min_confidence_threshold"]
    tolerance: int = golden["year_tolerance"]

    mismatches: list[str] = []
    for photo in golden["photos"]:
        expected = photo["expected"]
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

        # Year: either both None, or within the tolerance band.
        if expected["year"] is None:
            if estimate.year is not None:
                mismatches.append(f"photo {photo['id']} ({photo['label']}): expected year=None, got {estimate.year}")
        else:
            if estimate.year is None:
                mismatches.append(f"photo {photo['id']} ({photo['label']}): expected year≈{expected['year']}, got None")
            elif abs(estimate.year - expected["year"]) > tolerance:
                mismatches.append(
                    f"photo {photo['id']} ({photo['label']}): "
                    f"year {estimate.year} outside ±{tolerance} of {expected['year']}"
                )

        # Confidence: tight float tolerance – the math is deterministic.
        assert estimate.confidence == pytest.approx(expected["confidence"], abs=1e-4), (
            f"photo {photo['id']} confidence drift: {estimate.confidence} vs {expected['confidence']}"
        )

        assert estimate.review_needed is expected["review_needed"], (
            f"photo {photo['id']} review_needed drift: {estimate.review_needed} vs {expected['review_needed']}"
        )

    assert not mismatches, (
        "Golden ranking drifted for one or more photos:\n  - "
        + "\n  - ".join(mismatches)
        + "\nIf this change is intentional, rerun "
        "`python scripts/regenerate_golden_ranking.py` and review the diff."
    )


def test_golden_fixture_sort_ranks(golden: dict) -> None:
    """Computed sort-rank matches the snapshot exactly."""
    weights: dict[str, float] = golden["weights"]
    min_threshold: float = golden["min_confidence_threshold"]

    photo_estimates = []
    expected_ranks: dict[int, int] = {}
    for photo in golden["photos"]:
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
        photo_estimates.append((photo["id"], estimate))
        expected_ranks[photo["id"]] = photo["expected"]["sort_rank"]

    actual_ranks = dict(rank_estimates(photo_estimates))

    drifted = [
        f"photo {pid}: expected rank {expected_ranks[pid]}, got {actual_ranks[pid]}"
        for pid in expected_ranks
        if actual_ranks[pid] != expected_ranks[pid]
    ]
    assert not drifted, (
        "Sort-rank drifted for one or more photos:\n  - "
        + "\n  - ".join(drifted)
        + "\nIf intentional, rerun `python scripts/regenerate_golden_ranking.py`."
    )
