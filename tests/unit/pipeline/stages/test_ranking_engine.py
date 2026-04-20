"""Tests for the ranking engine pipeline stage."""

from unittest.mock import patch

import pytest

from photochron.anchor import Constraint, ConstraintSet, ConstraintType
from photochron.models import ContextCreate, FaceCreate, PersonCreate, PhotoCreate
from photochron.pipeline.stages.ranking_engine import RankingEngineStage


def _seed_photo(helper, name: str, exif: str | None = None) -> int:
    return helper.insert_photo(
        PhotoCreate(
            content_hash=f"hash_{name}",
            file_path=f"/photos/{name}",
            downsample_path=f"/cache/{name}",
            exif_datetime=exif,
        )
    )


def _seed_context(
    helper, photo_id: int, decade: str | None, medium: str = "print_scan"
) -> None:
    helper.insert_context(
        ContextCreate(
            photo_id=photo_id,
            decade=decade,
            decade_confidence=0.8 if decade else 0.0,
            season=None,
            season_confidence=None,
            event_hint=None,
            event_confidence=None,
            photo_medium=medium,
            photo_medium_confidence=0.6,
            visual_evidence=None,
            alternative_decades=None,
            uncertainty_flag=False,
            hypothesis_notes=None,
            raw_json="{}",
        )
    )


def _seed_person_and_face(helper, photo_id: int, birthday: str, age: float):
    person_id = helper.insert_person(
        PersonCreate(person_id=f"person_{photo_id}", name=f"p{photo_id}", birthday=birthday)
    )
    helper.insert_face(
        FaceCreate(
            photo_id=photo_id,
            person_id=person_id,
            embedding=None,
            age_estimate=age,
            age_std=1.0,
            confidence=0.9,
            bbox_x1=0.0,
            bbox_y1=0.0,
            bbox_x2=10.0,
            bbox_y2=10.0,
        )
    )


def _setup_run(helper, run_id: str, constraints_json: str | None = None):
    helper.conn.execute(
        "INSERT INTO pipeline_runs (run_id, schema_version, config_hash, start_time, status) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP, 'running')",
        (run_id, 1, "hash"),
    )
    if constraints_json is not None:
        helper.upsert_anchor_constraints(run_id, constraints_json, source_path=None)


@pytest.fixture
def patched_store(database_store):
    with (
        patch("photochron.pipeline.stages.ranking_engine.get_store", return_value=database_store),
        patch("photochron.pipeline.get_store", return_value=database_store),
    ):
        yield database_store


def test_ranking_with_exif_uses_exif_year(patched_store, mock_config):
    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        photo_id = _seed_photo(helper, "a.jpg", exif="1995:07:04 10:00:00")
        _seed_context(helper, photo_id, decade="1985-1990")
        _setup_run(helper, "run1", ConstraintSet().model_dump_json())

    with patch(
        "photochron.pipeline.stages.ranking_engine.get_config", return_value=mock_config
    ):
        stage = RankingEngineStage()
        stage.run("run1", "cfg")

    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        ranking = helper.get_ranking_by_photo_id(photo_id)
    assert ranking is not None
    assert ranking.estimated_year == 1995
    assert ranking.confidence == 1.0
    assert ranking.sort_rank == 0


def test_ranking_with_only_llm_decade(patched_store, mock_config):
    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        photo_id = _seed_photo(helper, "a.jpg")
        _seed_context(helper, photo_id, decade="1985-1990")
        _setup_run(helper, "run2", ConstraintSet().model_dump_json())

    with patch(
        "photochron.pipeline.stages.ranking_engine.get_config", return_value=mock_config
    ):
        RankingEngineStage().run("run2", "cfg")

    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        ranking = helper.get_ranking_by_photo_id(photo_id)
    assert ranking is not None
    assert ranking.estimated_year is not None
    assert 1985 <= ranking.estimated_year <= 1990


def test_ranking_uses_face_year_from_birthday(patched_store, mock_config):
    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        photo_id = _seed_photo(helper, "a.jpg")
        _seed_context(helper, photo_id, decade=None, medium="unknown")
        _seed_person_and_face(helper, photo_id, birthday="1983-03-15", age=5.0)
        _setup_run(helper, "run3", ConstraintSet().model_dump_json())

    with patch(
        "photochron.pipeline.stages.ranking_engine.get_config", return_value=mock_config
    ):
        RankingEngineStage().run("run3", "cfg")

    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        ranking = helper.get_ranking_by_photo_id(photo_id)
    assert ranking is not None
    assert ranking.estimated_year == 1988


def test_ranking_applies_hard_year_pin_from_constraints(patched_store, mock_config):
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

    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        photo_id = _seed_photo(helper, "xmas.jpg")
        _seed_context(helper, photo_id, decade="2000-2005")
        _setup_run(helper, "run4", cs.model_dump_json())

    with patch(
        "photochron.pipeline.stages.ranking_engine.get_config", return_value=mock_config
    ):
        RankingEngineStage().run("run4", "cfg")

    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        ranking = helper.get_ranking_by_photo_id(photo_id)
    assert ranking.estimated_year == 1990
    assert ranking.estimated_month == 12


def test_ranking_sorts_by_year(patched_store, mock_config):
    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        older = _seed_photo(helper, "older.jpg")
        newer = _seed_photo(helper, "newer.jpg")
        _seed_context(helper, older, decade="1980-1985")
        _seed_context(helper, newer, decade="2000-2005")
        _setup_run(helper, "run5", ConstraintSet().model_dump_json())

    with patch(
        "photochron.pipeline.stages.ranking_engine.get_config", return_value=mock_config
    ):
        RankingEngineStage().run("run5", "cfg")

    with patched_store.transaction() as conn:
        helper = patched_store.get_query_helper(conn)
        a = helper.get_ranking_by_photo_id(older)
        b = helper.get_ranking_by_photo_id(newer)
    assert a.sort_rank < b.sort_rank
