"""Shaped questions in Ask AI: tries, runs, history, library, exports (spec 033 Story 2)."""

import csv
import io
import time

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.fake_openai_server import FakeOpenAIServer
from tests.fixtures.synthetic_video import SyntheticVideoSpec, write_synthetic_video_avi
from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.batch.job_execution import run_job_pipelines
from videoannotator.database import database as db_module
from videoannotator.database.models import AskEntry, ModelServer, Prompt, PromptUse
from videoannotator.pipelines.vlm_annotation import openai_client, vlm_pipeline

client = TestClient(app)

LAB = "http://127.0.0.1:8000/v1"
SHAPE = [
    {"name": "touch", "kind": "yes_no"},
    {
        "name": "initiator",
        "kind": "choice",
        "values": ["adult", "child", "both", "none"],
    },
]


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    reset_storage_backend()
    db_module.Base.metadata.create_all(bind=db_module.engine)
    with db_module.SessionLocal() as db:
        for table in (AskEntry, PromptUse, Prompt, ModelServer):
            db.query(table).delete()
        db.commit()
    server = FakeOpenAIServer("vllm")
    server.replies = ['{"touch": "yes", "initiator": "adult"}']
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
        "prompt": "Is the adult touching the child? Who started it?",
        "model": fake.model,
        "base_url": LAB,
        "backend": "openai_compatible",
        "answer_shape": SHAPE,
        "enforce_shape": True,
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


def test_a_shaped_try_records_fields(video, fake):
    fake.replies = [
        '{"touch": "yes", "initiator": "adult"}',
        '{"touch": "no", "initiator": "mum"}',
    ]
    r = client.post(
        "/api/v1/ask/tries",
        json=_body(video, fake, count=2, sampling={"interval": 1, "end": 4}),
    )
    assert r.status_code == 201, r.text
    assert r.json()["answer_shape"] == SHAPE
    entry = _until_done(r.json()["id"])
    first, second = entry["answers"]
    assert (first["label"], first["fields"], first["fit"]) == (
        "yes",
        {"touch": "yes", "initiator": "adult"},
        True,
    )
    assert (second["label"], second["fit"]) == ("DIDN'T FIT", False)
    assert (
        second["fit_problem"]
        == "initiator: 'mum' is not one of adult, child, both, none"
    )
    assert "response_format" in fake.chat_requests[0].body


def test_an_invalid_shape_is_refused(video, fake):
    r = client.post(
        "/api/v1/ask/tries",
        json=_body(
            video, fake, at=1.0, answer_shape=[{"name": "label", "kind": "text"}]
        ),
    )
    assert (r.status_code, r.json()["error"]["code"]) == (422, "INVALID_ANSWER_SHAPE")
    assert "reserved" in r.json()["error"]["message"]
    assert not fake.chat_requests


def test_the_library_keeps_the_shape(video, fake):
    _until_done(
        client.post("/api/v1/ask/tries", json=_body(video, fake, at=1.0)).json()["id"]
    )
    with db_module.SessionLocal() as db:
        prompt = db.query(Prompt).one()
        assert prompt.answer_shape == SHAPE
    entries = client.get("/api/v1/ask/entries").json()["entries"]
    assert entries[0]["answer_shape"] == SHAPE


def test_a_library_question_reopens_with_its_shape(video, fake):
    r = client.post("/api/v1/ask/drafts", json=_body(video, fake))
    assert r.status_code == 201
    entry_id = r.json()["id"]
    client.delete(f"/api/v1/ask/entries/{entry_id}")
    library = client.get("/api/v1/ask/entries").json()["library"]
    assert library[0]["answer_shape"] == SHAPE


def test_a_shaped_run_and_its_wide_file(video, fake):
    r = client.post("/api/v1/ask/runs", json=_body(video, fake))
    assert r.status_code == 201, r.text
    entry = r.json()
    storage = get_storage_backend()
    job = storage.load_job_metadata(entry["job_id"])
    assert job.config["vlm_annotation"]["answer_shape"] == SHAPE
    run_job_pipelines(
        job, storage, {"vlm_annotation": vlm_pipeline.VLMAnnotationPipeline}
    )

    answers = client.get(f"/api/v1/ask/runs/{entry['job_id']}/answers").json()[
        "answers"
    ]
    assert answers[0]["fields"] == {"touch": "yes", "initiator": "adult"}
    record = job.pipeline_results["vlm_annotation"].provenance
    assert record["vlm"]["answer_shape_sha256"]
    assert record["vlm"]["structured_output"] == "enforced"

    r = client.get(
        f"/api/v1/ask/entries/{entry['id']}/export/wide", follow_redirects=True
    )
    assert r.status_code == 200, r.text
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert list(rows[0]) == [
        "video", "t", "frame", "touch", "initiator",
        "fit", "fit_problem", "model", "backend", "raw_response",
    ]  # fmt: skip
    assert (rows[0]["touch"], rows[0]["initiator"], rows[0]["fit"]) == (
        "yes",
        "adult",
        "yes",
    )
    assert rows[0]["backend"] == "openai_compatible"


def test_a_tries_wide_file(video, fake):
    entry = _until_done(
        client.post("/api/v1/ask/tries", json=_body(video, fake, at=1.0)).json()["id"]
    )
    r = client.get(f"/api/v1/ask/entries/{entry['id']}/export/wide")
    assert r.status_code == 200, r.text
    assert r.headers["content-disposition"].endswith('_answers_wide.csv"')
    (row,) = list(csv.DictReader(io.StringIO(r.text)))
    assert (row["t"], row["touch"], row["fit"]) == ("1.0", "yes", "yes")


def test_no_wide_file_without_a_shape(video, fake):
    entry = _until_done(
        client.post(
            "/api/v1/ask/tries", json=_body(video, fake, at=1.0, answer_shape=None)
        ).json()["id"]
    )
    r = client.get(f"/api/v1/ask/entries/{entry['id']}/export/wide")
    assert (r.status_code, r.json()["error"]["code"]) == (422, "NOT_SHAPED")


def test_a_shaped_tries_table_has_a_measure_per_field(video, fake):
    entry = _until_done(
        client.post("/api/v1/ask/tries", json=_body(video, fake, at=1.0)).json()["id"]
    )
    r = client.get(f"/api/v1/ask/entries/{entry['id']}/export/table")
    rows = list(csv.DictReader(io.StringIO(r.text)))
    measures = {row["measure"]: row for row in rows}
    assert measures["vlm.touch"]["label"] == "yes"
    assert measures["vlm.initiator"]["label"] == "adult"
    assert measures["vlm_fit"]["label"] == "yes"
    assert "vlm_label" in measures
