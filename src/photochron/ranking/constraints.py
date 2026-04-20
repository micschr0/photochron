"""
Apply anchor constraints to photo date estimates.

Hard constraints clamp the estimate to satisfy the rule (confidence set to 1.0
for exact year pins, or lifted for after/before bounds). Soft constraints nudge
the estimate within weight budget and may flag `review_needed`.
"""

from datetime import date
from pathlib import Path

from loguru import logger

from photochron.anchor import Constraint, ConstraintSet, ConstraintType

from .estimator import DateEstimate


def _match_file(file_path: str, constraint_file: str) -> bool:
    """Return True if photo's file path matches the constraint filename.

    Matches on basename (case-insensitive) or suffix of the original path.
    """
    if not file_path or not constraint_file:
        return False
    base = Path(file_path).name.lower()
    target = constraint_file.lower()
    return base == target or file_path.lower().endswith(target)


def _apply_year_pin(estimate: DateEstimate, c: Constraint) -> None:
    if c.year is None:
        return
    estimate.year = c.year
    if c.month is not None:
        estimate.month = c.month
    if c.type == ConstraintType.HARD:
        estimate.confidence = max(estimate.confidence, 0.95)
        estimate.review_needed = False
        estimate.notes = (
            f"{estimate.notes}; hard pin {c.source} {c.year}".strip("; ")
        )
    else:
        estimate.confidence = max(estimate.confidence, 0.7)
        estimate.notes = (
            f"{estimate.notes}; soft pin {c.source} {c.year}".strip("; ")
        )


def _apply_after(estimate: DateEstimate, c: Constraint) -> None:
    if not c.reference_date:
        return
    try:
        ref = date.fromisoformat(c.reference_date)
    except ValueError:
        return
    if estimate.year is None:
        estimate.year = ref.year
        estimate.month = ref.month
        estimate.confidence = max(estimate.confidence, 0.4)
        return
    if estimate.year < ref.year:
        if c.type == ConstraintType.HARD:
            estimate.year = ref.year
            estimate.month = ref.month
            estimate.notes = (
                f"{estimate.notes}; clamped to after {c.reference_date}".strip("; ")
            )
            estimate.review_needed = True


def _apply_before(estimate: DateEstimate, c: Constraint) -> None:
    if not c.reference_date:
        return
    try:
        ref = date.fromisoformat(c.reference_date)
    except ValueError:
        return
    if estimate.year is None:
        estimate.year = ref.year
        estimate.month = ref.month
        estimate.confidence = max(estimate.confidence, 0.4)
        return
    if estimate.year > ref.year:
        if c.type == ConstraintType.HARD:
            estimate.year = ref.year
            estimate.month = ref.month
            estimate.notes = (
                f"{estimate.notes}; clamped to before {c.reference_date}".strip("; ")
            )
            estimate.review_needed = True


def apply_constraints(
    estimates: list[tuple[int, str, DateEstimate]],
    constraint_set: ConstraintSet,
) -> list[tuple[int, str, DateEstimate]]:
    """Apply constraints to each estimate in-place.

    Args:
        estimates: list of (photo_id, file_path, estimate) tuples.
        constraint_set: parsed ConstraintSet from anchor layer.

    Returns the same list (modified in-place) for chaining.
    """
    if not constraint_set.constraints:
        return estimates

    by_kind: dict[str, list[Constraint]] = {
        "photo_year": [],
        "photo_after": [],
        "photo_before": [],
    }
    for c in constraint_set.constraints:
        by_kind.setdefault(c.kind, []).append(c)

    hard_then_soft = sorted(
        constraint_set.constraints,
        key=lambda c: 0 if c.type == ConstraintType.HARD else 1,
    )

    for photo_id, file_path, estimate in estimates:
        for c in hard_then_soft:
            if not _match_file(file_path, c.file):
                continue
            if c.kind == "photo_year":
                _apply_year_pin(estimate, c)
            elif c.kind == "photo_after":
                _apply_after(estimate, c)
            elif c.kind == "photo_before":
                _apply_before(estimate, c)
            else:
                logger.debug("Ignoring unknown constraint kind: {}", c.kind)

    return estimates
