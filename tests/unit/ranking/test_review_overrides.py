"""Unit tests for ``apply_review_overrides``.

Pinned semantics:
- A user override wins over every AI signal AND over EXIF (the human said so).
- The override row is recorded as a ``user_override`` signal so the per-photo
  report keeps a breadcrumb.
- Photos without an override pass through unchanged.
- An override with NULL year is treated as "no opinion" and skipped, not as
  an error.
"""

from __future__ import annotations

from photochron.ranking.estimator import DateEstimate, apply_review_overrides


def _est(year: int | None, *, confidence: float = 0.4, review_needed: bool = True) -> DateEstimate:
    return DateEstimate(
        year=year,
        month=None,
        confidence=confidence,
        signals={"face": {"year": float(year or 0), "confidence": confidence}},
        review_needed=review_needed,
        notes="",
    )


def test_override_replaces_year_and_pins_confidence() -> None:
    estimates = [
        (1, "a.jpg", _est(1990)),
        (2, "b.jpg", _est(2000)),
    ]
    n = apply_review_overrides(
        estimates,
        {1: {"estimated_year": 1985, "estimated_month": 6}},
    )
    assert n == 1
    _, _, est1 = estimates[0]
    assert est1.year == 1985
    assert est1.month == 6
    assert est1.confidence == 1.0
    assert est1.review_needed is False
    assert "user_override" in est1.signals
    # The original face signal is preserved as a breadcrumb.
    assert "face" in est1.signals
    # Photo 2 was untouched.
    assert estimates[1][2].year == 2000


def test_no_override_means_estimate_unchanged() -> None:
    original = _est(1995, confidence=0.42, review_needed=True)
    estimates = [(1, "a.jpg", original)]
    n = apply_review_overrides(estimates, {})
    assert n == 0
    assert estimates[0][2] is original


def test_override_outranks_high_confidence_signal() -> None:
    """Even when the original was already confident, the user wins."""
    estimates = [(1, "a.jpg", _est(2010, confidence=0.99, review_needed=False))]
    apply_review_overrides(estimates, {1: {"estimated_year": 1975, "estimated_month": None}})
    _, _, est = estimates[0]
    assert est.year == 1975
    assert est.month is None


def test_override_with_null_year_is_skipped() -> None:
    """A row that says 'no opinion' (NULL year) should be a no-op, not a crash."""
    estimates = [(1, "a.jpg", _est(1990))]
    n = apply_review_overrides(estimates, {1: {"estimated_year": None, "estimated_month": None}})
    assert n == 0
    assert estimates[0][2].year == 1990


def test_override_for_unknown_photo_id_is_ignored() -> None:
    """Overrides for photos that aren't in the current estimate list don't blow up."""
    estimates = [(1, "a.jpg", _est(1990))]
    n = apply_review_overrides(
        estimates,
        {99: {"estimated_year": 1850, "estimated_month": None}},
    )
    assert n == 0


def test_override_note_preserves_prior_note() -> None:
    """If the estimator already wrote a note, the override prefixes it."""
    est = DateEstimate(year=2000, confidence=0.2, signals={}, review_needed=True, notes="No usable signals")
    estimates = [(1, "a.jpg", est)]
    apply_review_overrides(estimates, {1: {"estimated_year": 1980, "estimated_month": None}})
    _, _, new = estimates[0]
    assert "user-override" in new.notes
    assert "No usable signals" in new.notes
