"""Ask AI history entries (spec 027 R11)."""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from videoannotator import ask_history
from videoannotator.database.database import Base
from videoannotator.database.models import AskEntry, PromptUse


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    with sessionmaker(bind=engine)() as session:
        yield session


def _try(db, **kwargs):
    fields = {
        "kind": "try",
        "prompt": "Is the adult touching the infant?",
        "model": "qwen3.5:9b",
        "base_url": "http://localhost:11434",
        "endpoint_class": "local",
        "sampling": {"interval": 5.0, "frames": 1},
        "think": False,
        "video_path": "/data/p01_t1.mp4",
        "moments": [12.4],
        "user_id": "u1",
    }
    fields.update(kwargs)
    return ask_history.create_entry(db, **fields)


def _answer(t, label):
    return {
        "t": t,
        "frame_numbers": [round(t * 25)],
        "frame_times": [t],
        "status": "answered",
        "label": label,
        "reasoning": label,
        "raw_response": label,
        "error": None,
        "total_time": 1.5,
    }


def test_create_add_answers_in_order_and_finish(db):
    entry = _try(db, moments=[1.0, 2.0])
    assert entry.status == "running"
    assert entry.answers == []
    ask_history.add_answer(db, entry.id, _answer(1.0, "TOUCH"))
    ask_history.add_answer(db, str(entry.id), _answer(2.0, "NO_TOUCH"))
    ask_history.finish(db, entry.id, "done")

    d = ask_history.to_dict(ask_history.get_entry(db, str(entry.id)))
    assert [a["label"] for a in d["answers"]] == ["TOUCH", "NO_TOUCH"]
    assert d["status"] == "done"
    assert d["finished_at"] is not None
    assert d["prompt"] == "Is the adult touching the infant?"
    assert d["video_name"] == "p01_t1.mp4"
    assert d["user_id"] == "u1"


def test_a_try_records_a_try_use_in_the_prompt_library(db):
    _try(db)
    uses = db.query(PromptUse).all()
    assert [(u.kind, u.model, u.user_id) for u in uses] == [("try", "qwen3.5:9b", "u1")]


def test_a_draft_or_run_adds_no_use(db):
    _try(db, kind="draft", moments=None)
    _try(db, kind="run", moments=None, job_id="job-1")
    assert db.query(PromptUse).count() == 0
    assert {e.kind: e.status for e in db.query(AskEntry)} == {
        "draft": "done",
        "run": "running",
    }


def test_no_field_holds_image_data(db):
    entry = _try(db)
    ask_history.add_answer(db, entry.id, _answer(12.4, "TOUCH"))
    text = json.dumps(ask_history.to_dict(entry))
    assert "jpeg" not in text.lower()
    assert "base64" not in text.lower()
    assert max(len(str(v)) for v in ask_history.to_dict(entry).values()) < 2000


def test_restart_stops_running_tries_keeping_answers(db):
    cut_off = _try(db, moments=[1.0, 2.0])
    ask_history.add_answer(db, cut_off.id, _answer(1.0, "TOUCH"))
    finished = _try(db)
    ask_history.finish(db, finished.id, "done")
    run = _try(db, kind="run", moments=None, job_id="job-1")

    assert ask_history.stop_running_tries(db) == 1
    db.refresh(cut_off)
    db.refresh(run)
    assert cut_off.status == "stopped"
    assert len(cut_off.answers) == 1
    assert run.status == "running"


def test_unknown_or_malformed_id(db):
    assert ask_history.get_entry(db, "not-a-uuid") is None
    assert ask_history.get_entry(db, "00000000-0000-0000-0000-000000000000") is None


def test_server_startup_marks_running_tries_stopped():
    from fastapi.testclient import TestClient

    from videoannotator.api.main import app
    from videoannotator.database import database as db_module

    db_module.Base.metadata.create_all(bind=db_module.engine)
    with db_module.SessionLocal() as session:
        entry = _try(session, moments=[1.0])
        entry_id = entry.id
    with TestClient(app):
        pass
    with db_module.SessionLocal() as session:
        assert session.get(AskEntry, entry_id).status == "stopped"
        session.query(AskEntry).filter(AskEntry.id == entry_id).delete()
        session.commit()
