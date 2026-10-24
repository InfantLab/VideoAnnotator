"""SC-005: on a fresh install no frame can reach a remote address (spec 033 R5)."""

import json
import time
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.fake_openai_server import FakeOpenAIServer
from tests.fixtures.synthetic_video import SyntheticVideoSpec, write_synthetic_video_avi
from videoannotator import secrets_store
from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.middleware.auth import require_admin, validate_required_api_key
from videoannotator.batch.job_execution import run_job_pipelines
from videoannotator.batch.types import BatchJob, JobStatus
from videoannotator.database import database as db_module
from videoannotator.database.models import (
    AskEntry,
    EndpointConfirmation,
    ModelServer,
    ServerSetting,
)
from videoannotator.pipelines.vlm_annotation import openai_client, vlm_pipeline

client = TestClient(app)

# A public address written as an IP, so classifying it needs no DNS.
REMOTE = "http://203.0.113.5:8000/v1"


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    reset_storage_backend()
    monkeypatch.delenv("VIDEOANNOTATOR_REMOTE_MODEL_SERVERS", raising=False)
    db_module.Base.metadata.create_all(bind=db_module.engine)
    with db_module.SessionLocal() as db:
        for table in (AskEntry, EndpointConfirmation, ModelServer, ServerSetting):
            db.query(table).delete()
        db.commit()
    secrets_store.model_server_keys_path().unlink(missing_ok=True)
    server = FakeOpenAIServer("vllm")
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
    yield server
    app.dependency_overrides.pop(require_admin, None)
    app.dependency_overrides.pop(validate_required_api_key, None)
    reset_storage_backend()


@pytest.fixture
def video(ingest_root):
    path = ingest_root / "p01_t1.avi"
    write_synthetic_video_avi(path, SyntheticVideoSpec(fps=10, num_frames=30))
    return str(path)


def _ask(fake, video, **kwargs):
    body = {
        "video_path": video,
        "prompt": "Touch?",
        "model": fake.model,
        "base_url": REMOTE,
        "backend": "openai_compatible",
    }
    body.update(kwargs)
    return body


def _code(r):
    return r.status_code, r.json()["error"]["code"]


def _confirm(before_switch: bool) -> None:
    with db_module.SessionLocal() as db:
        db.add(
            EndpointConfirmation(
                base_url=REMOTE,
                confirmed_by="u1",
                confirmed_at=datetime.now(UTC),
                before_switch=before_switch,
            )
        )
        db.commit()


def _as_admin(yes: bool = True) -> None:
    user = {"id": "admin" if yes else "irene", "is_admin": yes}
    if yes:
        app.dependency_overrides[require_admin] = lambda: user
    else:
        app.dependency_overrides[validate_required_api_key] = lambda: user


def test_every_route_refuses_a_remote_address_on_a_fresh_install(fake, video, tmp_path):
    r = client.post(
        "/api/v1/model-servers",
        json={"name": "hosted", "kind": "openai_compatible", "base_url": REMOTE},
    )
    assert _code(r) == (403, "REMOTE_NOT_ALLOWED")
    assert "switched off on this server" in r.json()["error"]["message"]
    assert "Settings → Model servers" in r.json()["error"]["hint"]

    assert _code(client.post("/api/v1/ask/tries", json=_ask(fake, video, at=1.0))) == (
        403,
        "REMOTE_NOT_ALLOWED",
    )
    assert _code(client.post("/api/v1/ask/runs", json=_ask(fake, video))) == (
        403,
        "REMOTE_NOT_ALLOWED",
    )
    preview = client.post(
        "/api/v1/vlm/preview",
        data={
            "video_path": video,
            "timestamp_sec": "1.0",
            "model": fake.model,
            "base_url": REMOTE,
            "backend": "openai_compatible",
        },
    )
    assert _code(preview) == (403, "REMOTE_NOT_ALLOWED")
    suggest = client.post(
        "/api/v1/ask/suggest-shape",
        json={
            "prompt": "What is happening?",
            "model": fake.model,
            "base_url": REMOTE,
            "backend": "openai_compatible",
        },
    )
    assert _code(suggest) == (403, "REMOTE_NOT_ALLOWED")

    config = {
        "vlm_annotation": {
            "backend": "openai_compatible",
            "base_url": REMOTE,
            "model": fake.model,
        }
    }
    with open(video, "rb") as f:
        submitted = client.post(
            "/api/v1/jobs/",
            files={"video": ("p01_t1.avi", f, "video/x-msvideo")},
            data={"selected_pipelines": "vlm_annotation", "config": json.dumps(config)},
        )
    assert _code(submitted) == (403, "REMOTE_NOT_ALLOWED")

    # A run that arrives any other way (the CLI, a stored job) stops in initialize().
    job = BatchJob(
        video_path=video,
        config=config,
        status=JobStatus.PENDING,
        selected_pipelines=["vlm_annotation"],
    )
    job.storage_path = tmp_path / "job"
    storage = get_storage_backend()
    storage.save_job_metadata(job)
    run_job_pipelines(
        job, storage, {"vlm_annotation": vlm_pipeline.VLMAnnotationPipeline}
    )
    result = job.pipeline_results["vlm_annotation"]
    assert result.status == JobStatus.FAILED
    assert "switched off" in result.error_message

    assert fake.chat_requests == []


def test_switched_on_the_confirmation_is_still_needed(fake, video):
    _as_admin()
    r = client.put(
        "/api/v1/model-servers/settings", json={"remote_model_servers": True}
    )
    assert r.status_code == 200, r.text
    assert r.json()["remote_model_servers"] is True
    assert r.json()["warning"]
    r = client.post(
        "/api/v1/model-servers",
        json={"name": "hosted", "kind": "openai_compatible", "base_url": REMOTE},
    )
    assert r.status_code == 201, r.text
    assert r.json()["endpoint"]["sendable"] is False

    assert _code(client.post("/api/v1/ask/tries", json=_ask(fake, video, at=1.0))) == (
        409,
        "REMOTE_NOT_CONFIRMED",
    )
    assert fake.chat_requests == []
    assert (
        client.post(
            "/api/v1/ask/endpoints/confirm", json={"base_url": REMOTE}
        ).status_code
        == 204
    )
    r = client.post("/api/v1/ask/tries", json=_ask(fake, video, at=1.0))
    assert r.status_code == 201, r.text
    deadline = time.monotonic() + 10
    while not fake.chat_requests and time.monotonic() < deadline:
        time.sleep(0.05)
    assert len(fake.chat_requests) == 1


def test_the_environment_fixes_the_switch(monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_REMOTE_MODEL_SERVERS", "off")
    _as_admin()
    r = client.put(
        "/api/v1/model-servers/settings", json={"remote_model_servers": True}
    )
    assert _code(r) == (409, "FIXED_BY_ENVIRONMENT")
    settings = client.get("/api/v1/model-servers/settings").json()
    assert (settings["remote_model_servers"], settings["fixed_by_environment"]) == (
        False,
        True,
    )


def test_a_confirmation_from_before_the_switch_still_works(fake, video):
    _confirm(before_switch=True)
    state = client.get("/api/v1/ask/endpoint", params={"base_url": REMOTE}).json()
    assert (state["sendable"], state["before_switch"]) == (True, True)
    r = client.post("/api/v1/ask/tries", json=_ask(fake, video, at=1.0))
    assert r.status_code == 201, r.text
    r = client.post(
        "/api/v1/model-servers",
        json={"name": "old farm", "kind": "openai_compatible", "base_url": REMOTE},
    )
    assert r.status_code == 201, r.text
    assert r.json()["endpoint"]["before_switch"] is True


def test_a_confirmation_made_after_the_switch_existed_needs_it(fake, video):
    _confirm(before_switch=False)
    assert _code(client.post("/api/v1/ask/tries", json=_ask(fake, video, at=1.0))) == (
        403,
        "REMOTE_NOT_ALLOWED",
    )


def test_only_an_admin_switches_it():
    _as_admin(False)
    r = client.put(
        "/api/v1/model-servers/settings", json={"remote_model_servers": True}
    )
    assert r.status_code == 403
