"""SC-006: a model server's key appears in no response, file, export, database
row or log line (spec 033 FR-016, FR-017)."""

import json
import logging
import time
from pathlib import Path

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
STORED = "sk-stored-key-0123456789abcdef"
FROM_ENV = "sk-environment-key-fedcba9876543210"
SHAPE = [{"name": "touch", "kind": "yes_no"}]


def _database_bytes() -> bytes:
    """The file the engine writes, whatever an earlier test left in the environment."""
    return Path(db_module.engine.url.database).read_bytes()


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    reset_storage_backend()
    db_module.Base.metadata.create_all(bind=db_module.engine)
    with db_module.SessionLocal() as db:
        db.query(AskEntry).delete()
        db.query(ModelServer).delete()
        db.commit()
    secrets_store.model_server_keys_path().unlink(missing_ok=True)
    secrets_store.install_key_redaction()
    yield
    secrets_store.model_server_keys_path().unlink(missing_ok=True)
    secrets_store.KEY_ENV_NAMES.clear()
    reset_storage_backend()


@pytest.fixture
def video(ingest_root):
    path = ingest_root / "p01_t1.avi"
    write_synthetic_video_avi(path, SyntheticVideoSpec(fps=10, num_frames=30))
    return str(path)


def _until_done(entry_id: str) -> dict:
    deadline = time.monotonic() + 10
    while True:
        entry = client.get(f"/api/v1/ask/entries/{entry_id}").json()
        if entry["status"] != "running":
            return entry
        assert time.monotonic() < deadline
        time.sleep(0.05)


@pytest.mark.parametrize("source", ["stored", "env"])
def test_the_key_goes_nowhere_but_the_header(source, video, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG)
    key = STORED if source == "stored" else FROM_ENV
    server = FakeOpenAIServer("vllm", key=key, replies=['{"touch": "yes"}'])
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
    add = {"name": "lab", "kind": "openai_compatible", "base_url": LAB}
    if source == "stored":
        add["api_key"] = key
    else:
        monkeypatch.setenv("LAB_GPU_KEY", key)
        add["api_key_env"] = "LAB_GPU_KEY"

    responses = [client.post("/api/v1/model-servers", json=add)]
    server_id = responses[0].json()["id"]
    body = {
        "video_path": video,
        "prompt": "Touch?",
        "model": server.model,
        "base_url": LAB,
        "backend": "openai_compatible",
        "answer_shape": SHAPE,
    }
    tried = client.post("/api/v1/ask/tries", json={**body, "at": 1.0})
    responses.append(tried)
    _until_done(tried.json()["id"])
    run = client.post("/api/v1/ask/runs", json=body)
    responses.append(run)
    storage = get_storage_backend()
    job = storage.load_job_metadata(run.json()["job_id"])
    run_job_pipelines(
        job, storage, {"vlm_annotation": vlm_pipeline.VLMAnnotationPipeline}
    )
    storage.save_job_metadata(job)
    # A wrong key afterwards: the refusal says so without repeating anything.
    logging.getLogger("videoannotator.test").info("server key %s in a log line", key)

    responses += [
        client.get("/api/v1/model-servers"),
        client.get(f"/api/v1/ask/entries/{tried.json()['id']}"),
        client.get("/api/v1/ask/entries"),
        client.get(f"/api/v1/ask/runs/{job.job_id}/answers"),
        client.get(f"/api/v1/ask/entries/{tried.json()['id']}/export/table"),
        client.get(f"/api/v1/ask/entries/{tried.json()['id']}/export/wide"),
        client.get(f"/api/v1/jobs/{job.job_id}/export/table"),
        client.get(f"/api/v1/jobs/{job.job_id}/export/wide"),
        client.get(f"/api/v1/jobs/{job.job_id}/methods"),
        client.get(f"/api/v1/jobs/{job.job_id}"),
        client.post(
            "/api/v1/model-servers/check",
            json={"kind": "openai_compatible", "base_url": LAB, "id": server_id},
        ),
    ]
    server.key = "sk-something-else-entirely-000"
    responses.append(client.post("/api/v1/ask/tries", json={**body, "at": 1.0}))
    assert responses[-1].json()["error"]["code"] == "KEY_REFUSED"

    assert all(r.authorized for r in server.requests if r.path.startswith("/v1/"))
    for r in responses:
        assert key not in r.text, r.request.url
    assert json.loads(responses[3 + 3].text)  # the run's answers parsed
    folder = job_folder(job)
    files = [p for p in folder.rglob("*") if p.is_file()]
    assert any(p.name.endswith("_vlm_annotation.json") for p in files)
    for path in files:
        assert key.encode() not in path.read_bytes(), path
    assert key.encode() not in _database_bytes()
    for record in caplog.records:
        assert key not in record.getMessage()
    keys_file = secrets_store.model_server_keys_path()
    if source == "stored":
        assert key in keys_file.read_text()
    else:
        assert not keys_file.exists() or key not in keys_file.read_text()


def test_a_key_in_a_config_is_refused_and_stored_nowhere(video, monkeypatch):
    config = {"vlm_annotation": {"model": "m", "api_key": STORED}}
    with open(video, "rb") as f:
        r = client.post(
            "/api/v1/jobs/",
            files={"video": ("p01_t1.avi", f, "video/x-msvideo")},
            data={"selected_pipelines": "vlm_annotation", "config": json.dumps(config)},
        )
    assert (r.status_code, r.json()["error"]["code"]) == (422, "KEY_IN_CONFIG")
    assert "Settings → Model servers" in r.json()["error"]["message"]
    assert STORED not in r.text
    assert STORED.encode() not in _database_bytes()
