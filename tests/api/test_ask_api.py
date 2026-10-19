"""Ask AI's API (spec 027) with a fake model server."""

import threading
import time
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.synthetic_video import SyntheticVideoSpec, write_synthetic_video_avi
from videoannotator.api.database import reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.v1 import ask
from videoannotator.database import database as db_module
from videoannotator.database.models import AskEntry, EndpointConfirmation, PromptUse
from videoannotator.pipelines.vlm_annotation.ollama_client import (
    OllamaUnavailableError,
    VLMCallResult,
)

client = TestClient(app)

LOCAL = "http://127.0.0.1:11434"
MODEL = "qwen3.5:9b"


class FakeModelServer:
    """Stands in for OllamaVLMClient; tests change its behaviour per case."""

    models: ClassVar[list[str]] = [MODEL]
    reachable = True
    answer = "TOUCH"
    error: str | None = None
    calls: ClassVar[list[dict[str, Any]]] = []
    hold: threading.Event | None = None

    def __init__(self, base_url: str, timeout: int):
        self.base_url = base_url

    def list_models(self):
        if not FakeModelServer.reachable:
            raise OllamaUnavailableError("Cannot reach ollama server")
        return list(FakeModelServer.models)

    def chat(self, *, model, prompt, images, **kwargs):
        FakeModelServer.calls.append({"model": model, "images": len(images)})
        if FakeModelServer.hold is not None:
            FakeModelServer.hold.wait(10)
        if FakeModelServer.error:
            return VLMCallResult("", "", 0.5, 0.0, 0, 0, 0.0, FakeModelServer.error)
        return VLMCallResult(FakeModelServer.answer, "", 1.25, 0.1, 10, 1, 1.0)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    reset_storage_backend()
    db_module.Base.metadata.create_all(bind=db_module.engine)
    with db_module.SessionLocal() as db:
        db.query(AskEntry).delete()
        db.query(EndpointConfirmation).delete()
        db.query(PromptUse).delete()
        db.commit()
    FakeModelServer.models = [MODEL]
    FakeModelServer.reachable = True
    FakeModelServer.answer = "TOUCH"
    FakeModelServer.error = None
    FakeModelServer.calls = []
    FakeModelServer.hold = None
    monkeypatch.setattr(ask, "OllamaVLMClient", FakeModelServer)
    yield FakeModelServer
    if FakeModelServer.hold is not None:
        FakeModelServer.hold.set()
    reset_storage_backend()


@pytest.fixture
def video(ingest_root):
    path = ingest_root / "p01_t1.avi"
    write_synthetic_video_avi(path, SyntheticVideoSpec(fps=10, num_frames=50))
    return str(path)


def _try(video: str, **kwargs):
    body = {
        "video_path": video,
        "prompt": "Is the adult touching the infant?",
        "model": MODEL,
        "base_url": LOCAL,
        "at": 1.0,
    }
    body.update(kwargs)
    if "count" in kwargs:
        body.pop("at")
    return client.post("/api/v1/ask/tries", json=body)


def _until_done(entry_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        entry = client.get(f"/api/v1/ask/entries/{entry_id}").json()
        if entry["status"] != "running":
            return entry
        assert time.monotonic() < deadline, entry
        time.sleep(0.05)


class TestTries:
    def test_returns_running_before_the_model_answers(self, video, fake_model):
        fake_model.hold = threading.Event()
        r = _try(video)
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["status"] == "running"
        assert body["moments"] == [1.0]
        assert body["endpoint_class"] == "local"
        fake_model.hold.set()
        done = _until_done(body["id"])
        assert done["status"] == "done"
        [answer] = done["answers"]
        assert answer["status"] == "answered"
        assert answer["label"] == "TOUCH"
        assert answer["frame_numbers"] == [10]
        assert answer["frame_times"] == [1.0]
        assert answer["total_time"] == 1.25

    def test_a_burst_sends_its_frames(self, video, fake_model):
        r = _try(video, sampling={"frames": 3, "spacing": 0.5})
        done = _until_done(r.json()["id"])
        [answer] = done["answers"]
        assert answer["frame_numbers"] == [5, 10, 15]
        assert fake_model.calls == [{"model": MODEL, "images": 3}]

    def test_a_model_error_is_an_error_answer(self, video, fake_model):
        fake_model.error = "model crashed"
        done = _until_done(_try(video).json()["id"])
        assert done["status"] == "done"
        [answer] = done["answers"]
        assert answer["status"] == "error"
        assert answer["label"] is None
        assert answer["error"] == "model crashed"

    def test_an_unreadable_frame_is_skipped(self, video, monkeypatch):
        monkeypatch.setattr(ask.vlm_frames, "read_sample", lambda *a, **k: [])
        done = _until_done(_try(video).json()["id"])
        [answer] = done["answers"]
        assert answer["status"] == "skipped"
        assert answer["frame_numbers"] == []

    def test_unreachable_model_server_says_what_to_do(self, video, fake_model):
        fake_model.reachable = False
        r = _try(video)
        assert r.status_code == 503
        error = r.json()["error"]
        assert error["code"] == "OLLAMA_UNREACHABLE"
        assert error["hint"]

    def test_model_not_pulled(self, video, fake_model):
        r = _try(video, model="llava:7b")
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "MODEL_NOT_PULLED"
        assert "ollama pull llava:7b" in r.json()["error"]["hint"]

    def test_missing_llm_extra(self, video, monkeypatch):
        monkeypatch.setattr(ask, "extras_available", lambda extras: False)
        r = _try(video)
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "PIPELINE_UNAVAILABLE"

    def test_records_a_try_use_in_the_prompt_library(self, video):
        _until_done(_try(video).json()["id"])
        with db_module.SessionLocal() as db:
            assert [u.kind for u in db.query(PromptUse)] == ["try"]

    def test_invalid_sampling_names_the_field(self, video):
        r = _try(video, sampling={"interval": 0})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "INVALID_SAMPLING"
        assert r.json()["error"]["details"]["field"] == "interval"
        r = _try(video, sampling={"frames": 12})
        assert r.json()["error"]["details"]["field"] == "frames"

    def test_at_or_count_not_both(self, video):
        r = client.post(
            "/api/v1/ask/tries",
            json={
                "video_path": video,
                "prompt": "x",
                "model": MODEL,
                "base_url": LOCAL,
                "at": 1.0,
                "count": 3,
            },
        )
        assert r.status_code == 422

    def test_video_outside_the_shared_folders(self, video, tmp_path):
        r = _try(str(tmp_path / "elsewhere.avi"))
        assert r.status_code == 403
        assert r.json()["error"]["code"] == "PATH_NOT_ALLOWED"

    def test_unknown_entry(self):
        assert client.get(f"/api/v1/ask/entries/{'0' * 8}").status_code == 404
