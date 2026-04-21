"""
Anchor data parsing and constraint modeling.

Loads anchors.yaml and produces an in-memory ConstraintSet consumed by the
ranking engine. Persons are synced into the persons table; events and known
dates are serialized to the anchor_constraints table for the current run.
"""

from .loader import load_anchors, parse_anchors
from .models import (
    AnchorEvent,
    AnchorKnownDate,
    AnchorPerson,
    Constraint,
    ConstraintSet,
    ConstraintType,
)

__all__ = [
    "AnchorPerson",
    "AnchorEvent",
    "AnchorKnownDate",
    "Constraint",
    "ConstraintSet",
    "ConstraintType",
    "load_anchors",
    "parse_anchors",
]
