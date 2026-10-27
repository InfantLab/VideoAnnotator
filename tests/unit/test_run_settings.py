"""What a new run starts from, in FR-012's order (spec 029 R5, R6)."""

from __future__ import annotations

import uuid

import pytest

from tests.fixtures.video_index import JobsDb, make_dataset, make_preset
from videoannotator import run_settings


@pytest.fixture
def db():
    import videoannotator.database.database as db_module

    db_module.Base.metadata.create_all(bind=db_module.engine)
    with db_module.SessionLocal() as session:
        yield session


@pytest.fixture
def user():
    return {"id": str(uuid.uuid4())}


def _name() -> str:
    return f"QS {uuid.uuid4().hex[:6]}"


def test_defaults_when_nothing_is_known(db, user):
    found = run_settings.prefill(db, user=user)

    assert found == {
        "selected_pipelines": [],
        "config": {},
        "source": {"kind": "defaults"},
    }


def test_the_last_run_is_remembered_per_user(db, user):
    run_settings.record_run(
        db,
        user=user,
        dataset_id=None,
        selected_pipelines=["face_analysis"],
        config={"face_analysis": {"min_face_size": 40}},
        run_id="b1",
        run_name="Peekaboo",
    )

    mine = run_settings.prefill(db, user=user)
    theirs = run_settings.prefill(db, user={"id": str(uuid.uuid4())})

    assert mine["selected_pipelines"] == ["face_analysis"]
    assert mine["config"] == {"face_analysis": {"min_face_size": 40}}
    assert mine["source"]["kind"] == "last_used"
    assert mine["source"]["name"] == "Peekaboo"
    assert mine["source"]["changed_at"]
    assert theirs["source"]["kind"] == "defaults"


def test_without_sign_in_the_last_run_is_the_anonymous_one(db):
    run_settings.record_run(
        db,
        user=None,
        dataset_id=None,
        selected_pipelines=["scene_detection"],
        config={},
        run_id="b9",
        run_name=None,
    )

    assert run_settings.prefill(db, user=None)["selected_pipelines"] == [
        "scene_detection"
    ]


def test_a_run_on_a_project_sets_its_settings(db, user):
    project = make_dataset(db, _name())
    run_settings.record_run(
        db,
        user=user,
        dataset_id=str(project.id),
        selected_pipelines=["face_analysis"],
        config={"face_analysis": {"min_face_size": 40}},
        run_id="b2",
        run_name="Faces",
    )
    run_settings.record_run(
        db,
        user=user,
        dataset_id=None,
        selected_pipelines=["scene_detection"],
        config={},
        run_id="b3",
        run_name="Elsewhere",
    )

    found = run_settings.prefill(db, user=user, project_id=str(project.id))

    assert found["selected_pipelines"] == ["face_analysis"]
    assert found["source"]["kind"] == "project"
    assert found["source"]["name"] == project.name
    assert found["source"]["from"] == {"kind": "run", "id": "b2", "name": "Faces"}


def test_an_old_project_starts_from_its_newest_run(db, user, tmp_path):
    jobs = JobsDb(tmp_path)
    project = make_dataset(db, _name())
    video = tmp_path / "v.mp4"
    jobs.make_job(
        video, dataset_id=str(project.id), pipelines={"a": ("completed", None, None)}
    )
    jobs.make_job(
        video,
        dataset_id=str(project.id),
        batch_id="b7",
        batch_name="Newest",
        config={"b": {"x": 1}},
        pipelines={"b": ("completed", None, None)},
    )

    found = run_settings.prefill(
        db, user=user, project_id=str(project.id), storage=jobs.storage
    )

    assert found["selected_pipelines"] == ["b"]
    assert found["config"] == {"b": {"x": 1}}
    assert found["source"]["kind"] == "project"
    assert found["source"]["from"]["name"] == "Newest"


def test_a_project_with_neither_starts_from_the_defaults_and_offers_the_last_run(
    db, user, tmp_path
):
    project = make_dataset(db, _name())
    run_settings.record_run(
        db,
        user=user,
        dataset_id=None,
        selected_pipelines=["scene_detection"],
        config={},
        run_id="b4",
        run_name="Mine",
    )

    found = run_settings.prefill(
        db, user=user, project_id=str(project.id), storage=JobsDb(tmp_path).storage
    )

    assert found["source"]["kind"] == "defaults"
    assert found["selected_pipelines"] == []
    assert found["last_run"]["selected_pipelines"] == ["scene_detection"]
    assert found["last_run"]["name"] == "Mine"


def test_a_chosen_workflow_comes_first(db, user):
    workflow = make_preset(
        db, _name(), pipelines=["speech_recognition"], config={"speech": {"m": "x"}}
    )
    project = make_dataset(
        db,
        _name(),
        current_settings={"selected_pipelines": ["face_analysis"], "config": {}},
    )

    found = run_settings.prefill(
        db, user=user, project_id=str(project.id), workflow_id=str(workflow.id)
    )

    assert found["selected_pipelines"] == ["speech_recognition"]
    assert found["source"]["kind"] == "workflow"
    assert found["source"]["name"] == workflow.name
