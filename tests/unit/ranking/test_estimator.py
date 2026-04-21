"""Tests for ranking estimator: decade midpoint, face year, signal combine."""

from datetime import date

from photochron.ranking import (
    DateEstimate,
    combine_signals,
    decade_midpoint,
    face_year_estimate,
    medium_prior_year,
)
from photochron.ranking.estimator import rank_estimates

WEIGHTS = {"face": 0.45, "llm": 0.30, "medium": 0.10}


def test_decade_midpoint_valid():
    assert decade_midpoint("1985-1990") == 1987
    assert decade_midpoint("2000-2010") == 2005


def test_decade_midpoint_invalid_returns_none():
    assert decade_midpoint(None) is None
    assert decade_midpoint("") is None
    assert decade_midpoint("not-a-decade") is None


def test_face_year_estimate_basic():
    # Mama born 1983, looks ~4 -> 1987
    year = face_year_estimate(date(2024, 1, 1), "1983-03-15", 4.2)
    assert year == 1987


def test_face_year_estimate_caps_at_present():
    # If age would predict future year, clamp to today's year
    year = face_year_estimate(date(2024, 1, 1), "2000-01-01", 99.0)
    assert year == 2024


def test_face_year_estimate_missing_inputs():
    assert face_year_estimate(date(2024, 1, 1), None, 4.0) is None
    assert face_year_estimate(date(2024, 1, 1), "1983-03-15", None) is None
    assert face_year_estimate(date(2024, 1, 1), "garbage", 4.0) is None


def test_medium_prior_year_lookup():
    assert medium_prior_year("digital")[0] == 2010
    assert medium_prior_year("POLAROID")[0] == 1978
    assert medium_prior_year("unseen_medium") is None
    assert medium_prior_year(None) is None


def test_combine_signals_exif_overrides_everything():
    est = combine_signals(
        exif_datetime="1995:07:04 10:00:00",
        face_year=1980,
        face_confidence=0.9,
        decade="2000-2010",
        decade_confidence=0.8,
        photo_medium="digital",
        photo_medium_confidence=0.9,
        weights=WEIGHTS,
        min_confidence_threshold=0.5,
    )
    assert est.year == 1995
    assert est.month == 7
    assert est.confidence == 1.0
    assert "exif" in est.signals


def test_combine_signals_weighted_combine():
    est = combine_signals(
        exif_datetime=None,
        face_year=1987,
        face_confidence=0.9,
        decade="1985-1990",
        decade_confidence=0.8,
        photo_medium="print_scan",
        photo_medium_confidence=0.5,
        weights=WEIGHTS,
        min_confidence_threshold=0.5,
    )
    assert est.year is not None
    assert 1985 <= est.year <= 1990
    assert "face" in est.signals
    assert "llm_decade" in est.signals
    assert est.confidence > 0


def test_combine_signals_no_usable_signals():
    est = combine_signals(
        exif_datetime=None,
        face_year=None,
        face_confidence=None,
        decade=None,
        decade_confidence=None,
        photo_medium=None,
        photo_medium_confidence=None,
        weights=WEIGHTS,
        min_confidence_threshold=0.5,
    )
    assert est.year is None
    assert est.review_needed
    assert est.confidence == 0.0


def test_combine_signals_low_confidence_flags_review():
    est = combine_signals(
        exif_datetime=None,
        face_year=None,
        face_confidence=None,
        decade="1985-1990",
        decade_confidence=0.2,
        photo_medium=None,
        photo_medium_confidence=None,
        weights=WEIGHTS,
        min_confidence_threshold=0.5,
    )
    assert est.year is not None
    assert est.review_needed is True


def test_rank_estimates_orders_by_year_then_month():
    e1 = DateEstimate(year=1990, month=5, confidence=0.6)
    e2 = DateEstimate(year=1985, month=1, confidence=0.8)
    e3 = DateEstimate(year=1990, month=2, confidence=0.7)
    e4 = DateEstimate(year=None, confidence=0.0)
    ranked = rank_estimates([(1, e1), (2, e2), (3, e3), (4, e4)])
    order = [photo_id for photo_id, _ in ranked]
    # e2 first (1985), then 1990/Feb, then 1990/May, then missing
    assert order == [2, 3, 1, 4]
