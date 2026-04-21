"""
Pydantic models for anchor data and constraints.
"""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConstraintType(StrEnum):
    """Constraint strength - hard constraints are inviolable, soft are hints."""

    HARD = "hard"
    SOFT = "soft"


class AnchorPerson(BaseModel):
    """Person entry from anchors.yaml."""

    id: str = Field(..., description="Stable person identifier (slug).")
    name: str = Field(..., description="Display name.")
    birthday: str | None = Field(None, description="ISO date string (YYYY-MM-DD). Optional.")

    @field_validator("birthday")
    @classmethod
    def _validate_birthday(cls, v: str | None) -> str | None:
        if v is None:
            return v
        from datetime import date

        try:
            date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"birthday must be ISO date YYYY-MM-DD, got '{v}'") from exc
        return v


class AnchorEvent(BaseModel):
    """Event entry from anchors.yaml producing one or more constraints."""

    name: str
    date: str = Field(..., description="ISO date of the event.")
    type: ConstraintType = ConstraintType.SOFT
    photos_after: list[str] = Field(default_factory=list)
    photos_before: list[str] = Field(default_factory=list)

    @field_validator("date")
    @classmethod
    def _validate_date(cls, v: str) -> str:
        from datetime import date

        try:
            date.fromisoformat(v)
        except ValueError as exc:
            raise ValueError(f"event date must be ISO date, got '{v}'") from exc
        return v


class AnchorKnownDate(BaseModel):
    """Known partial date for a specific file."""

    file: str = Field(..., description="Filename (basename) of the photo.")
    year: int | None = Field(None, ge=1800, le=2100)
    month: int | None = Field(None, ge=1, le=12)
    day: int | None = Field(None, ge=1, le=31)
    type: ConstraintType = ConstraintType.SOFT


class Constraint(BaseModel):
    """
    A single ranking constraint derived from anchors.yaml.

    Kinds:
      - 'photo_year': pin a photo to a year (with month/day if known)
      - 'photo_after': photo must be chronologically after a reference date
      - 'photo_before': photo must be chronologically before a reference date
    """

    kind: str
    file: str
    reference_date: str | None = None
    year: int | None = None
    month: int | None = None
    day: int | None = None
    type: ConstraintType = ConstraintType.SOFT
    source: str = Field("", description="Human-readable source (event name / known_dates entry).")


class ConstraintSet(BaseModel):
    """Full parsed constraint set for one pipeline run."""

    persons: list[AnchorPerson] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    def person_by_id(self, person_id: str) -> AnchorPerson | None:
        for p in self.persons:
            if p.id == person_id:
                return p
        return None
