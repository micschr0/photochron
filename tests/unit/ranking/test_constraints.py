"""Tests for ranking constraint application."""

from photochron.anchor import Constraint, ConstraintSet, ConstraintType
from photochron.ranking import apply_constraints
from photochron.ranking.estimator import DateEstimate


def test_hard_year_pin_overrides_estimate():
    cs = ConstraintSet(
        constraints=[
            Constraint(
                kind="photo_year",
                file="xmas.jpg",
                year=1990,
                month=12,
                type=ConstraintType.HARD,
                source="known_date",
            )
        ]
    )
    est = DateEstimate(year=1985, confidence=0.4)
    apply_constraints([(1, "/photos/xmas.jpg", est)], cs)
    assert est.year == 1990
    assert est.month == 12
    assert est.confidence >= 0.95
    assert est.review_needed is False


def test_soft_year_pin_raises_floor():
    cs = ConstraintSet(
        constraints=[
            Constraint(
                kind="photo_year",
                file="xmas.jpg",
                year=1990,
                type=ConstraintType.SOFT,
                source="known_date",
            )
        ]
    )
    est = DateEstimate(year=1985, confidence=0.3)
    apply_constraints([(1, "xmas.jpg", est)], cs)
    assert est.year == 1990
    assert est.confidence >= 0.7


def test_after_hard_clamps_earlier_estimate():
    cs = ConstraintSet(
        constraints=[
            Constraint(
                kind="photo_after",
                file="IMG_042.jpg",
                reference_date="1991-08-01",
                type=ConstraintType.HARD,
                source="event:Umzug",
            )
        ]
    )
    est = DateEstimate(year=1988, confidence=0.5)
    apply_constraints([(1, "IMG_042.jpg", est)], cs)
    assert est.year == 1991
    assert est.review_needed is True


def test_before_hard_clamps_later_estimate():
    cs = ConstraintSet(
        constraints=[
            Constraint(
                kind="photo_before",
                file="IMG_010.jpg",
                reference_date="1991-08-01",
                type=ConstraintType.HARD,
                source="event:Umzug",
            )
        ]
    )
    est = DateEstimate(year=1995, confidence=0.5)
    apply_constraints([(1, "IMG_010.jpg", est)], cs)
    assert est.year == 1991


def test_non_matching_file_unaffected():
    cs = ConstraintSet(
        constraints=[
            Constraint(
                kind="photo_year",
                file="other.jpg",
                year=1950,
                type=ConstraintType.HARD,
            )
        ]
    )
    est = DateEstimate(year=2000, confidence=0.8)
    apply_constraints([(1, "mine.jpg", est)], cs)
    assert est.year == 2000


def test_file_match_by_basename_and_suffix():
    cs = ConstraintSet(
        constraints=[
            Constraint(
                kind="photo_year",
                file="photo.jpg",
                year=1980,
                type=ConstraintType.HARD,
            )
        ]
    )
    a = DateEstimate(year=2000)
    b = DateEstimate(year=2000)
    apply_constraints(
        [
            (1, "/home/user/photos/photo.jpg", a),
            (2, "photo.jpg", b),
        ],
        cs,
    )
    assert a.year == 1980
    assert b.year == 1980
