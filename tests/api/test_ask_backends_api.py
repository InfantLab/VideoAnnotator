"""Ask AI through an OpenAI-compatible server (spec 033 Story 1)."""

import json
import time

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.fake_openai_server import FakeOpenAIServer
from tests.fixtures.synthetic_video import SyntheticVideoSpec, write_synthetic_video_avi
from videoannotator import secrets_store
from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.batch.job_execution import run_job_pipelines
from videoannotator.batch.result_files import job_folder
from videoannotator.database import database as db_module
from videoannotator.database.models import AskEntry, ModelServer
from videoannotator.pipelines.vlm_annotation import openai_client, vlm_pipeline

client = TestClient(app)

LAB = "http://127.0.0.1:8000/v1"


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    reset_storage_backend()
    db_module.Base.metadata.create_all(bind=db_module.engine)
    with db_module.SessionLocal() as db:
        db.query(AskEntry).delete()
        db.query(ModelServer).delete()
        db.commit()
    secrets_store.model_server_keys_path().unlink(missing_ok=True)
    server = FakeOpenAIServer("vllm")
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
    yield server
    secrets_store.model_server_keys_path().unlink(missing_ok=True)
    reset_storage_backend()


@pytest.fixture
def video(ingest_root):
    path = ingest_root / "p01_t1.avi"
    write_synthetic_video_avi(path, SyntheticVideoSpec(fps=10, num_frames=50))
    return str(path)


def _body(video, fake, **kwargs):
    body = {
        "video_path": video,
        "prompt": "Is the adult touching the infant?",
        "model": fake.model,
        "base_url": LAB,
        "backend": "openai_compatible",
    }
    body.update(kwargs)
    return body


def _until_done(entry_id: str) -> dict:
    deadline = time.monotonic() + 10
    while True:
        entry = client.get(f"/api/v1/ask/entries/{entry_id}").json()
        if entry["status"] != "running":
            return entry
        assert time.monotonic() < deadline, entry
        time.sleep(0.05)


def test_a_try_is_answered_by_the_server(video, fake):
    r = client.post("/api/v1/ask/tries", json=_body(video, fake, at=1.0))
    assert r.status_code == 201, r.text
    assert r.json()["backend"] == "openai_compatible"
    entry = _until_done(r.json()["id"])
    assert entry["status"] == "done"
    (answer,) = entry["answers"]
    assert (answer["status"], answer["label"]) == ("answered", "TOUCH")
    assert len(fake.chat_requests) == 1
    parts = fake.chat_requests[0].body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "image_url"]


def test_the_server_needs_an_address(video, fake):
    r = client.post("/api/v1/ask/tries", json=_body(video, fake, at=1.0, base_url=None))
    assert (r.status_code, r.json()["error"]["code"]) == (422, "MISSING_BASE_URL")


def test_model_not_on_the_server(video, fake):
    r = client.post("/api/v1/ask/tries", json=_body(video, fake, at=1.0, model="x"))
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "MODEL_NOT_FOUND"
    assert fake.model in r.json()["error"]["message"]


def test_server_down(video, fake):
    fake.down = True
    r = client.post("/api/v1/ask/tries", json=_body(video, fake, at=1.0))
    assert (r.status_code, r.json()["error"]["code"]) == (
        503,
        "MODEL_SERVER_UNREACHABLE",
    )


def test_key_refused(video, fake):
    fake.refuse_key = True
    r = client.post("/api/v1/ask/tries", json=_body(video, fake, at=1.0))
    assert (r.status_code, r.json()["error"]["code"]) == (401, "KEY_REFUSED")


def test_a_try_stops_at_a_refused_key(video, fake):
    fake.refuse_key_after = 1
    r = client.post(
        "/api/v1/ask/tries",
        json=_body(video, fake, count=3, sampling={"interval": 1}),
    )
    entry = _until_done(r.json()["id"])
    assert entry["status"] == "failed"
    assert [a["status"] for a in entry["answers"]] == ["answered", "error"]
    assert entry["answers"][-1]["error"] == "The model server refused the key."
    assert len(fake.chat_requests) == 2


def test_a_run_records_the_backend(video, fake):
    r = client.post("/api/v1/ask/runs", json=_body(video, fake))
    assert r.status_code == 201, r.text
    entry = r.json()
    assert entry["backend"] == "openai_compatible"
    storage = get_storage_backend()
    job = storage.load_job_metadata(entry["job_id"])
    config = job.config["vlm_annotation"]
    assert (config["backend"], config["base_url"]) == ("openai_compatible", LAB)

    run_job_pipelines(
        job, storage, {"vlm_annotation": vlm_pipeline.VLMAnnotationPipeline}
    )
    data = json.loads(
        (job_folder(job) / "p01_t1_vlm_annotation.json").read_text(encoding="utf-8")
    )
    assert {a["backend"] for a in data["annotations"]} == {"openai_compatible"}
    record = job.pipeline_results["vlm_annotation"].provenance
    assert record["vlm"]["backend"] == "openai_compatible"
    assert record["models"][0]["source"] == "openai_compatible"
    answers = client.get(f"/api/v1/ask/runs/{entry['job_id']}/answers").json()
    assert [a["label"] for a in answers["answers"]] == ["TOUCH"] * len(
        data["annotations"]
    )


def test_models_of_a_listed_server(fake):
    server = client.post(
        "/api/v1/model-servers",
        json={"name": "lab", "kind": "openai_compatible", "base_url": LAB},
    ).json()
    r = client.get("/api/v1/vlm/models", params={"server": server["id"]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert (body["models"], body["kind"], body["base_url"]) == (
        [fake.model],
        "openai_compatible",
        LAB,
    )
    assert (
        client.get("/api/v1/vlm/models", params={"server": "nope"}).status_code == 404
    )


def test_a_draft_keeps_the_backend(video, fake):
    r = client.post("/api/v1/ask/drafts", json=_body(video, fake))
    assert r.status_code == 201, r.text
    assert r.json()["backend"] == "openai_compatible"
