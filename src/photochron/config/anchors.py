"""
Parsing and validation for anchors.yaml file.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class PersonAnchor(BaseModel):
    """Person anchor with birthday."""

    id: str = Field(..., description="Unique identifier for person")
    name: str = Field(..., description="Display name")
    birthday: str = Field(..., description="Birthday in YYYY-MM-DD format")

    @field_validator("birthday")
    @classmethod
    def validate_birthday(cls, v: str) -> str:
        """Validate birthday format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid birthday format: {v}. Expected YYYY-MM-DD")
        return v


class EventAnchor(BaseModel):
    """Event anchor with date and constraints."""

    name: str = Field(..., description="Event name")
    date: str = Field(..., description="Event date in YYYY-MM-DD format")
    type: Literal["hard", "soft"] = Field("soft", description="Constraint type")
    photos_after: list[str] | None = Field(None, description="Photos that must be AFTER this date")
    photos_before: list[str] | None = Field(None, description="Photos that must be BEFORE this date")

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        """Validate date format."""
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Invalid date format: {v}. Expected YYYY-MM-DD")
        return v

    @field_validator("photos_after", "photos_before")
    @classmethod
    def validate_photo_lists(cls, v: list[str] | None) -> list[str] | None:
        """Validate photo lists are not empty."""
        if v is not None and len(v) == 0:
            raise ValueError("Photo list cannot be empty")
        return v


class KnownDateAnchor(BaseModel):
    """Partially known date for a photo."""

    file: str = Field(..., description="Photo file name (without path)")
    month: int = Field(..., ge=1, le=12, description="Month (1-12)")
    year: int | None = Field(None, ge=1900, le=2100, description="Year (optional)")
    type: Literal["hard", "soft"] = Field("soft", description="Constraint type")


class Anchors(BaseModel):
    """Root anchors model."""

    persons: list[PersonAnchor] | None = Field(None, description="List of persons")
    events: list[EventAnchor] | None = Field(None, description="List of events")
    known_dates: list[KnownDateAnchor] | None = Field(None, description="List of known dates")

    model_config = {"extra": "forbid"}


def load_anchors(anchors_path: Path | None = None) -> Anchors:
    """
    Load and validate anchors from YAML file.

    Args:
        anchors_path: Path to anchors.yaml file. If None, looks for anchors.yaml
                     in current directory and project root.

    Returns:
        Anchors: Validated anchors object.

    Raises:
        FileNotFoundError: If anchors file not found and no default exists
        ValidationError: If anchors fail validation
    """
    # Find anchors file
    if anchors_path is None:
        possible_paths = [
            Path("anchors.yaml"),
            Path(__file__).parent.parent.parent / "anchors.yaml",
        ]
        for path in possible_paths:
            if path.exists():
                anchors_path = path
                break
        else:
            # Return empty anchors if no file exists
            return Anchors()

    # Load YAML
    with open(anchors_path) as f:
        anchors_data = yaml.safe_load(f) or {}

    # Validate with Pydantic
    return Anchors.model_validate(anchors_data)


def validate_anchors(anchors: Anchors) -> list[str]:
    """
    Validate anchor consistency and return warnings.

    Args:
        anchors: Loaded anchors object

    Returns:
        List of warning messages (empty if no issues)
    """
    warnings = []

    # Check for duplicate person IDs
    if anchors.persons:
        person_ids = [p.id for p in anchors.persons]
        duplicates = {id for id in person_ids if person_ids.count(id) > 1}
        if duplicates:
            warnings.append(f"Duplicate person IDs: {', '.join(duplicates)}")

    # Check for duplicate event names
    if anchors.events:
        event_names = [e.name for e in anchors.events]
        duplicates = {name for name in event_names if event_names.count(name) > 1}
        if duplicates:
            warnings.append(f"Duplicate event names: {', '.join(duplicates)}")

    # Check for contradictory hard constraints
    # (This is a simplified check - full constraint validation happens in pipeline)
    if anchors.events:
        hard_events = [e for e in anchors.events if e.type == "hard"]
        # Check for events with same date but different constraints
        event_dates = {}
        for event in hard_events:
            if event.date in event_dates:
                warnings.append(
                    f"Multiple hard events on same date {event.date}: {event_dates[event.date]} and {event.name}"
                )
            else:
                event_dates[event.date] = event.name

    return warnings


def create_anchors_template(template_path: Path) -> None:
    """Create a template anchors.yaml file with examples."""
    template = """# photochron Anchors File
# User-provided anchor data for chronological sorting

# Persons section: Known people with birthdays
persons:
  - id: person_mama
    name: "Mama"
    birthday: "1983-03-15"

  - id: person_papa
    name: "Papa"
    birthday: "1980-07-22"

# Events section: Known events with dates and constraints
events:
  - name: "Move to New City"
    date: "1991-08-01"
    type: hard
    photos_after:
      - "IMG_042.jpg"

  - name: "Family Vacation"
    date: "2005-06-15"
    type: soft
    photos_before:
      - "vacation_01.jpg"

# Known dates section: Photos with partially known dates
known_dates:
  - file: "Christmas_photo.jpg"
    month: 12
    year: 1998
    type: soft
"""

    with open(template_path, "w") as f:
        f.write(template)
