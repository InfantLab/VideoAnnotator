"""A short clip per sample in Ask AI (spec 033 Story 4)."""

import time

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.fake_openai_server import FakeOpenAIServer
from tests.fixtures.synthetic_video import SyntheticVideoSpec, write_synthetic_video_avi
from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.database import database as db_module
from videoannotator.database.models import AskEntry
from videoannotator.pipelines.vlm_annotation import openai_client

client = TestClient(app)
LAB = "http://127.0.0.1:8000/v1"


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    reset_storage_backend()
    db_module.Base.metadata.create_all(bind=db_module.engine)
    with db_module.SessionLocal() as db:
        db.query(AskEntry).delete()
        db.commit()
    server = FakeOpenAIServer("vllm")
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
    yield server
    reset_storage_backend()


@pytest.fixture
def video(ingest_root):
    path = ingest_root / "p01_t1.avi"
    write_synthetic_video_avi(path, SyntheticVideoSpec(fps=10, num_frames=50))
    return str(path)


def _body(video, fake, **kwargs):
    body = {
        "video_path": video,
        "prompt": "What happens?",
        "model": fake.model,
        "base_url": LAB,
        "backend": "openai_compatible",
        "sampling": {"interval": 2, "clip": {"length": 2, "fps": 2}},
    }
    body.update(kwargs)
    return body


def test_a_try_sends_a_clip_and_records_its_span(video, fake):
    r = client.post("/api/v1/ask/tries", json=_body(video, fake, at=0.0))
    assert r.status_code == 201, r.text
    deadline = time.monotonic() + 10
    while (entry := client.get(f"/api/v1/ask/entries/{r.json()['id']}").json())[
        "status"
    ] == "running":
        assert time.monotonic() < deadline
        time.sleep(0.05)
    (answer,) = entry["answers"]
    assert (answer["clip_start"], answer["clip_end"], answer["clip_fps"]) == (
        0.0,
        1.0,
        2.0,
    )
    assert answer["frame_numbers"] == [0, 5]
    parts = fake.chat_requests[0].body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "video_url"]


def test_a_run_has_clip_settings(video, fake):
    r = client.post("/api/v1/ask/runs", json=_body(video, fake))
    assert r.status_code == 201, r.text
    config = (
        get_storage_backend()
        .load_job_metadata(r.json()["job_id"])
        .config["vlm_annotation"]
    )
    assert (config["sampling_mode"], config["clip_length_sec"], config["clip_fps"]) == (
        "clip",
        2,
        2,
    )


def test_ollama_cant_take_a_clip(video, fake):
    r = client.post(
        "/api/v1/ask/tries",
        json=_body(video, fake, at=0.0, backend="ollama", base_url=None),
    )
    assert (r.status_code, r.json()["error"]["code"]) == (422, "CLIP_NOT_SUPPORTED")


def test_the_frames_a_clip_sends(video):
    r = client.get(
        "/api/v1/ask/frames",
        params={"path": video, "t": 2.0, "clip_length": 2, "clip_fps": 2},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert [f["timestamp_sec"] for f in body["frames"]] == [1.0, 1.5, 2.0, 2.5]
    assert body["requested"] == 4
