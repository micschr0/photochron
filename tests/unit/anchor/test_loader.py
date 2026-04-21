"""Tests for anchors.yaml parsing and ConstraintSet construction."""

from pathlib import Path

import pytest

from photochron.anchor import ConstraintType, load_anchors, parse_anchors


def test_parse_empty_yaml_yields_empty_constraint_set():
    cs = parse_anchors({})
    assert cs.persons == []
    assert cs.constraints == []


def test_parse_persons_and_events():
    data = {
        "persons": [
            {"id": "person_mama", "name": "Mama", "birthday": "1983-03-15"},
            {"id": "person_oma", "name": "Oma"},
        ],
        "events": [
            {
                "name": "Umzug",
                "date": "1991-08-01",
                "type": "hard",
                "photos_after": ["IMG_042.jpg"],
                "photos_before": ["IMG_041.jpg"],
            }
        ],
        "known_dates": [{"file": "Weihnachten.jpg", "year": 1990, "month": 12, "type": "soft"}],
    }
    cs = parse_anchors(data)
    assert len(cs.persons) == 2
    assert cs.person_by_id("person_mama").birthday == "1983-03-15"

    kinds = sorted(c.kind for c in cs.constraints)
    assert kinds == ["photo_after", "photo_before", "photo_year"]

    hard = [c for c in cs.constraints if c.type == ConstraintType.HARD]
    soft = [c for c in cs.constraints if c.type == ConstraintType.SOFT]
    assert len(hard) == 2  # photo_after + photo_before from hard event
    assert len(soft) == 1  # known_dates soft entry


def test_invalid_birthday_raises():
    with pytest.raises(ValueError):
        parse_anchors({"persons": [{"id": "x", "name": "X", "birthday": "not-a-date"}]})


def test_contradicting_hard_constraints_raise():
    data = {
        "events": [
            {
                "name": "A",
                "date": "1990-01-01",
                "type": "hard",
                "photos_before": ["x.jpg"],
            },
            {
                "name": "B",
                "date": "1985-01-01",
                "type": "hard",
                "photos_after": ["x.jpg"],
            },
        ]
    }
    # after=1985 < before=1990 -> consistent; no error
    parse_anchors(data)


def test_hard_constraint_inconsistency_detected():
    data = {
        "events": [
            {
                "name": "A",
                "date": "1985-01-01",
                "type": "hard",
                "photos_before": ["x.jpg"],
            },
            {
                "name": "B",
                "date": "1990-01-01",
                "type": "hard",
                "photos_after": ["x.jpg"],
            },
        ]
    }
    with pytest.raises(ValueError):
        parse_anchors(data)


def test_load_anchors_returns_empty_for_missing_file(tmp_path: Path):
    cs = load_anchors(tmp_path / "does-not-exist.yaml")
    assert cs.persons == []
    assert cs.constraints == []


def test_load_anchors_reads_yaml_file(tmp_path: Path):
    yaml_path = tmp_path / "anchors.yaml"
    yaml_path.write_text(
        """
persons:
  - id: person_oma
    name: Oma
    birthday: "1942-05-20"
""".strip()
    )
    cs = load_anchors(yaml_path)
    assert len(cs.persons) == 1
    assert cs.persons[0].birthday == "1942-05-20"
