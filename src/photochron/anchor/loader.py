"""
Load and parse anchors.yaml into a ConstraintSet.
"""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from .models import (
    AnchorEvent,
    AnchorKnownDate,
    AnchorPerson,
    Constraint,
    ConstraintSet,
    ConstraintType,
)


def load_anchors(path: Path) -> ConstraintSet:
    """Load anchors.yaml from disk and return a validated ConstraintSet.

    Returns an empty ConstraintSet if the file does not exist.
    """
    if not path.exists():
        logger.info("No anchors file at {}; proceeding with empty constraint set", path)
        return ConstraintSet()

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}

    return parse_anchors(data)


def parse_anchors(data: dict[str, Any]) -> ConstraintSet:
    """Parse a raw anchors.yaml mapping into a ConstraintSet."""
    persons = _parse_persons(data.get("persons") or [])
    events = _parse_events(data.get("events") or [])
    known_dates = _parse_known_dates(data.get("known_dates") or [])

    constraints: list[Constraint] = []
    constraints.extend(_constraints_from_events(events))
    constraints.extend(_constraints_from_known_dates(known_dates))

    cs = ConstraintSet(persons=persons, constraints=constraints)
    _validate_constraints(cs)
    return cs


def _parse_persons(raw: list[dict[str, Any]]) -> list[AnchorPerson]:
    return [AnchorPerson.model_validate(item) for item in raw]


def _parse_events(raw: list[dict[str, Any]]) -> list[AnchorEvent]:
    return [AnchorEvent.model_validate(item) for item in raw]


def _parse_known_dates(raw: list[dict[str, Any]]) -> list[AnchorKnownDate]:
    return [AnchorKnownDate.model_validate(item) for item in raw]


def _constraints_from_events(events: list[AnchorEvent]) -> list[Constraint]:
    constraints: list[Constraint] = []
    for event in events:
        for photo in event.photos_after:
            constraints.append(
                Constraint(
                    kind="photo_after",
                    file=photo,
                    reference_date=event.date,
                    type=event.type,
                    source=f"event:{event.name}",
                )
            )
        for photo in event.photos_before:
            constraints.append(
                Constraint(
                    kind="photo_before",
                    file=photo,
                    reference_date=event.date,
                    type=event.type,
                    source=f"event:{event.name}",
                )
            )
    return constraints


def _constraints_from_known_dates(entries: list[AnchorKnownDate]) -> list[Constraint]:
    constraints: list[Constraint] = []
    for entry in entries:
        constraints.append(
            Constraint(
                kind="photo_year",
                file=entry.file,
                year=entry.year,
                month=entry.month,
                day=entry.day,
                type=entry.type,
                source="known_date",
            )
        )
    return constraints


def _validate_constraints(cs: ConstraintSet) -> None:
    """Detect obvious contradictions among hard constraints.

    Currently checks for hard photo_after vs photo_before where the 'after'
    date is not strictly earlier than the 'before' date on the same file.
    """
    from datetime import date

    by_file: dict[str, dict[str, str | None]] = {}
    for c in cs.constraints:
        if c.type != ConstraintType.HARD:
            continue
        bucket = by_file.setdefault(c.file, {"after": None, "before": None})
        if c.kind == "photo_after" and c.reference_date:
            existing = bucket["after"]
            if existing is None or c.reference_date > existing:
                bucket["after"] = c.reference_date
        elif c.kind == "photo_before" and c.reference_date:
            existing = bucket["before"]
            if existing is None or c.reference_date < existing:
                bucket["before"] = c.reference_date

    for file, bounds in by_file.items():
        a, b = bounds["after"], bounds["before"]
        if a and b and date.fromisoformat(a) >= date.fromisoformat(b):
            raise ValueError(
                f"Contradicting hard constraints for {file}: after={a} not before before={b}"
            )
