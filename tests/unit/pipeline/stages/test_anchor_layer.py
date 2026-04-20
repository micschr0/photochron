"""Tests for the anchor layer pipeline stage."""

from unittest.mock import patch

import pytest

from photochron.anchor import ConstraintSet, ConstraintType
from photochron.pipeline.stages.anchor_layer import AnchorLayerStage


@pytest.fixture
def setup_store(database_store):
    """Patch get_store across modules that use it in the anchor stage."""
    targets = [
        "photochron.pipeline.stages.anchor_layer.get_store",
        "photochron.pipeline.get_store",
    ]
    with (
        patch(targets[0], return_value=database_store),
        patch(targets[1], return_value=database_store),
    ):
        # Register a pipeline_runs row so mark_complete has a row to update
        with database_store.transaction() as conn:
            conn.execute(
                "INSERT INTO pipeline_runs (run_id, schema_version, config_hash, start_time, status) "
                "VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'running')",
                ("run_test", 1, "hash"),
            )
        yield database_store


def test_anchor_layer_with_missing_yaml_stores_empty_set(setup_store, tmp_path, monkeypatch):
    """When no anchors.yaml exists, the stage still persists an empty ConstraintSet."""
    monkeypatch.chdir(tmp_path)
    stage = AnchorLayerStage()
    stage.run("run_test", "cfg")

    with setup_store.transaction() as conn:
        helper = setup_store.get_query_helper(conn)
        raw = helper.get_anchor_constraints_json("run_test")
    assert raw is not None
    cs = ConstraintSet.model_validate_json(raw)
    assert cs.persons == []
    assert cs.constraints == []


def test_anchor_layer_parses_yaml_and_upserts_persons(setup_store, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    yaml_path = tmp_path / "anchors.yaml"
    yaml_path.write_text(
        """
persons:
  - id: person_mama
    name: Mama
    birthday: "1983-03-15"

events:
  - name: Umzug
    date: "1991-08-01"
    type: hard
    photos_after:
      - "IMG_042.jpg"

known_dates:
  - file: "xmas.jpg"
    year: 1990
    month: 12
    type: soft
""".strip()
    )

    stage = AnchorLayerStage()
    stage.run("run_test", "cfg")

    with setup_store.transaction() as conn:
        helper = setup_store.get_query_helper(conn)
        mama = helper.get_person_by_person_id("person_mama")
        raw = helper.get_anchor_constraints_json("run_test")

    assert mama is not None
    assert mama.name == "Mama"
    assert mama.birthday == "1983-03-15"

    cs = ConstraintSet.model_validate_json(raw)
    kinds = sorted(c.kind for c in cs.constraints)
    assert kinds == ["photo_after", "photo_year"]
    hard_constraints = [c for c in cs.constraints if c.type == ConstraintType.HARD]
    assert any(c.file == "IMG_042.jpg" for c in hard_constraints)
