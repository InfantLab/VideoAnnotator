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
from videoannotator.database.models import (
    AskEntry,
    EndpointConfirmation,
    Prompt,
    PromptUse,
)
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
        db.query(Prompt).delete()
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


USERS = {
    "irene": {"id": "u-irene", "username": "irene", "is_admin": False},
    "caspar": {"id": "u-caspar", "username": "caspar", "is_admin": False},
    "admin": {"id": "u-admin", "username": "root", "is_admin": True},
}


@pytest.fixture
def as_users():
    from fastapi import Request

    from videoannotator.api.middleware.auth import validate_api_key

    def by_header(request: Request):
        return USERS.get(request.headers.get("X-Test-User", ""))

    app.dependency_overrides[validate_api_key] = by_header
    yield lambda name: {"X-Test-User": name}
    app.dependency_overrides.pop(validate_api_key, None)


def _try_as(video, headers, prompt):
    r = client.post(
        "/api/v1/ask/tries",
        json={
            "video_path": video,
            "prompt": prompt,
            "model": MODEL,
            "base_url": LOCAL,
            "at": 1.0,
        },
        headers=headers,
    )
    assert r.status_code == 201, r.text
    return _until_done(r.json()["id"])


class TestHistory:
    def test_tries_listed_newest_first_with_who(self, video, as_users):
        for i in range(5):
            _try_as(video, as_users("irene"), f"Wording {i}")
        entries = client.get("/api/v1/ask/entries").json()["entries"]
        assert [e["prompt"] for e in entries] == [
            f"Wording {i}" for i in range(4, -1, -1)
        ]
        assert {e["user_id"] for e in entries} == {"u-irene"}

    def test_mine_filters_by_caller(self, video, as_users):
        _try_as(video, as_users("irene"), "Irene's")
        _try_as(video, as_users("caspar"), "Caspar's")
        mine = client.get(
            "/api/v1/ask/entries", params={"mine": True}, headers=as_users("caspar")
        ).json()["entries"]
        assert [e["prompt"] for e in mine] == ["Caspar's"]
        everyone = client.get("/api/v1/ask/entries", headers=as_users("caspar"))
        assert len(everyone.json()["entries"]) == 2

    def test_filter_by_video(self, video, ingest_root):
        other = ingest_root / "other.avi"
        write_synthetic_video_avi(other, SyntheticVideoSpec(fps=10, num_frames=20))
        _until_done(_try(video).json()["id"])
        _until_done(_try(str(other)).json()["id"])
        r = client.get("/api/v1/ask/entries", params={"video_path": str(other)}).json()
        assert [e["video_name"] for e in r["entries"]] == ["other.avi"]
        assert r["library"] == []

    def test_library_prompt_without_entry_appears_until_tried(self, video):
        from videoannotator import prompt_library

        with db_module.SessionLocal() as db:
            prompt_library.record_use(db, "From the old Prompts page", MODEL, "preview")
        library = client.get("/api/v1/ask/entries").json()["library"]
        assert [p["text"] for p in library] == ["From the old Prompts page"]
        assert library[0]["models"] == [MODEL]
        _until_done(_try(video, prompt="From the old Prompts page").json()["id"])
        r = client.get("/api/v1/ask/entries").json()
        assert r["library"] == []
        assert [e["prompt"] for e in r["entries"]] == ["From the old Prompts page"]

    def test_name_and_star_seen_by_another_user(self, video, as_users):
        entry = _try_as(video, as_users("irene"), "Touch?")
        r = client.patch(
            f"/api/v1/ask/entries/{entry['id']}",
            json={"name": "touch v1", "starred": True},
            headers=as_users("irene"),
        )
        assert r.status_code == 200
        seen = client.get(
            f"/api/v1/ask/entries/{entry['id']}", headers=as_users("caspar")
        ).json()
        assert (seen["name"], seen["starred"]) == ("touch v1", True)

    def test_starred_first_then_newest(self, video):
        ids = [
            _until_done(_try(video, prompt=f"q{i}").json()["id"])["id"]
            for i in range(3)
        ]
        client.patch(f"/api/v1/ask/entries/{ids[0]}", json={"starred": True})
        entries = client.get("/api/v1/ask/entries").json()["entries"]
        assert [e["prompt"] for e in entries] == ["q0", "q2", "q1"]

    def test_paging_with_before(self, video):
        for i in range(5):
            _until_done(_try(video, prompt=f"q{i}").json()["id"])
        first = client.get("/api/v1/ask/entries", params={"limit": 2}).json()
        assert [e["prompt"] for e in first["entries"]] == ["q4", "q3"]
        second = client.get(
            "/api/v1/ask/entries", params={"limit": 2, "before": first["next_before"]}
        ).json()
        assert [e["prompt"] for e in second["entries"]] == ["q2", "q1"]

    def test_delete_by_non_author_refused(self, video, as_users):
        entry = _try_as(video, as_users("irene"), "Touch?")
        url = f"/api/v1/ask/entries/{entry['id']}"
        assert client.delete(url, headers=as_users("caspar")).status_code == 403
        assert client.delete(url, headers=as_users("admin")).status_code == 204
        assert client.get(url).status_code == 404

    def test_draft_is_kept_without_answers(self, video):
        r = client.post(
            "/api/v1/ask/drafts",
            json={
                "video_path": video,
                "prompt": "Not tried yet",
                "model": MODEL,
                "base_url": LOCAL,
            },
        )
        assert r.status_code == 201
        assert (r.json()["kind"], r.json()["status"]) == ("draft", "done")
        assert r.json()["answers"] is None


class TestSampling:
    def test_frames_of_a_burst(self, video, fake_model):
        r = client.get(
            "/api/v1/ask/frames",
            params={"path": video, "t": 2.0, "frames": 3, "spacing": 0.5},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["requested"] == 3
        assert [f["timestamp_sec"] for f in body["frames"]] == [1.5, 2.0, 2.5]
        assert [f["frame_number"] for f in body["frames"]] == [15, 20, 25]
        assert all(f["jpeg_base64"] for f in body["frames"])
        assert fake_model.calls == []

    def test_frames_near_the_end_are_fewer(self, video):
        body = client.get(
            "/api/v1/ask/frames",
            params={"path": video, "t": 4.8, "frames": 3, "spacing": 0.5},
        ).json()
        assert body["requested"] == 3
        assert [f["timestamp_sec"] for f in body["frames"]] == [4.3, 4.8]

    def test_count_tries_the_moments_of_the_shared_cases(self, ingest_root):
        import json
        from pathlib import Path

        cases = json.loads(
            (
                Path(__file__).parent.parent
                / "fixtures"
                / "ask_ai"
                / "sampling_cases.json"
            ).read_text()
        )
        case = next(c for c in cases if c["name"] == "count 10 of 40 points")
        long_video = ingest_root / "long.avi"
        write_synthetic_video_avi(
            long_video,
            SyntheticVideoSpec(fps=1, num_frames=int(case["duration"])),
        )
        r = _try(str(long_video), count=10, sampling={"interval": case["interval"]})
        assert r.status_code == 201, r.text
        assert r.json()["moments"] == case["expect"]["chosen"]
        done = _until_done(r.json()["id"])
        assert [a["t"] for a in done["answers"]] == case["expect"]["chosen"]

    def test_listed_moments_are_tried_exactly(self, video):
        r = _try(video, count=10, sampling={"moments": [3.5, 0.5, 2.0]})
        assert r.json()["moments"] == [0.5, 2.0, 3.5]


REMOTE = "https://api.example.com"


@pytest.fixture
def remote_resolves(monkeypatch):
    import socket

    from videoannotator import endpoints

    real = socket.getaddrinfo

    def fake(host, *args, **kwargs):
        if host == "api.example.com":
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))]
        return real(host, *args, **kwargs)

    monkeypatch.setattr(endpoints.socket, "getaddrinfo", fake)


class TestEndpoint:
    def test_remote_unconfirmed_is_refused_until_confirmed(
        self, video, remote_resolves, as_users
    ):
        r = _try(video, base_url=REMOTE)
        assert r.status_code == 409
        assert r.json()["error"]["code"] == "REMOTE_NOT_CONFIRMED"
        assert "api.example.com" in r.json()["error"]["message"]

        state = client.get("/api/v1/ask/endpoint", params={"base_url": REMOTE}).json()
        assert state == {
            "base_url": REMOTE,
            "host": "api.example.com",
            "class": "remote",
            "confirmed": False,
        }

        r = client.post(
            "/api/v1/ask/endpoints/confirm",
            json={"base_url": REMOTE + "/"},
            headers=as_users("irene"),
        )
        assert r.status_code == 204
        r = client.post(
            "/api/v1/ask/tries",
            json={
                "video_path": video,
                "prompt": "Touch?",
                "model": MODEL,
                "base_url": REMOTE,
                "at": 1.0,
            },
            headers=as_users("caspar"),
        )
        assert r.status_code == 201, r.text
        assert r.json()["endpoint_class"] == "remote"
        _until_done(r.json()["id"])
        seen = client.get(
            "/api/v1/ask/endpoint",
            params={"base_url": REMOTE},
            headers=as_users("caspar"),
        ).json()
        assert seen["confirmed"] is True

    def test_local_is_always_confirmed(self):
        state = client.get("/api/v1/ask/endpoint", params={"base_url": LOCAL}).json()
        assert (state["class"], state["confirmed"]) == ("local", True)

    def test_models_list_says_where_the_server_is(self, monkeypatch):
        from videoannotator.api.v1 import vlm

        monkeypatch.setattr(vlm, "OllamaVLMClient", FakeModelServer)
        body = client.get("/api/v1/vlm/models", params={"base_url": LOCAL}).json()
        assert body["endpoint"] == {
            "host": "127.0.0.1",
            "class": "local",
            "confirmed": True,
        }


def _run(video, **kwargs):
    body = {"video_path": video, "prompt": "Touch?", "model": MODEL, "base_url": LOCAL}
    body.update(kwargs)
    return client.post("/api/v1/ask/runs", json=body)


class TestRuns:
    def test_a_run_is_a_job_with_the_mapped_config(self, video):
        from videoannotator.api.database import get_storage_backend

        r = _run(
            video,
            sampling={
                "interval": 2,
                "start": 1,
                "end": 4,
                "frames": 3,
                "spacing": 0.25,
            },
        )
        assert r.status_code == 201, r.text
        entry = r.json()
        assert (entry["kind"], entry["status"]) == ("run", "running")
        assert entry["job_status"] == "pending"
        job = get_storage_backend().load_job_metadata(entry["job_id"])
        assert job.selected_pipelines == ["vlm_annotation"]
        assert str(job.video_path) == video
        config = job.config["vlm_annotation"]
        assert config["frame_interval_sec"] == 2
        assert (config["start_sec"], config["end_sec"]) == (1, 4)
        assert config["sampling_mode"] == "frame_burst"
        assert config["burst_offsets"] == [-1, 0, 1]
        assert config["burst_spacing_sec"] == 0.25
        assert config["base_url"] == LOCAL
        assert config["prompt"] == "Touch?"

    def test_listed_moments_in_the_config(self, video):
        from videoannotator.api.database import get_storage_backend

        entry = _run(video, sampling={"moments": [3, 1, 1]}).json()
        config = get_storage_backend().load_job_metadata(entry["job_id"]).config
        assert config["vlm_annotation"]["moments_sec"] == [1, 3]
        assert config["vlm_annotation"]["sampling_mode"] == "single_frame"

    def test_answers_page_with_after(self, video):
        import json

        from videoannotator.api.database import get_storage_backend
        from videoannotator.batch.result_files import job_folder

        entry = _run(video).json()
        url = f"/api/v1/ask/runs/{entry['job_id']}/answers"
        before = client.get(url).json()
        assert (before["done"], before["answers"], before["planned"]) == (0, [], 2)

        job = get_storage_backend().load_job_metadata(entry["job_id"])
        folder = job_folder(job)
        folder.mkdir(parents=True, exist_ok=True)
        annotations = [
            {"timestamp_sec": 0.0, "context_frame_numbers": [0], "label": "TOUCH",
             "status": "answered", "frame_times": [0.0], "total_time": 1.0},
            {"timestamp_sec": 5.0, "context_frame_numbers": [50],
             "label": "ERROR: timed out"},
        ]  # fmt: skip
        (folder / "p01_t1_vlm_annotation.json").write_text(
            json.dumps(
                {
                    "annotations": annotations,
                    "vlm_progress": {
                        "planned": 4,
                        "done": 3,
                        "finished": False,
                        "stopped_early": False,
                    },
                    "vlm_skipped": [{"t": 2.5, "reason": "no readable frame"}],
                }
            )
        )
        body = client.get(url).json()
        assert (body["planned"], body["done"]) == (4, 3)
        assert [a["status"] for a in body["answers"]] == ["answered", "error"]
        assert body["answers"][1]["error"] == "timed out"
        assert body["answers"][1]["label"] is None
        assert body["skipped"] == [{"t": 2.5, "reason": "no readable frame"}]
        later = client.get(url, params={"after": 1}).json()
        assert [a["t"] for a in later["answers"]] == [5.0]

    def test_a_try_during_a_run_goes_before_the_runs_next_sample(
        self, video, monkeypatch
    ):
        from tests.fixtures.fake_vlm import FakeTryVLM, FakeVLM
        from videoannotator.api.database import get_storage_backend
        from videoannotator.batch.job_execution import run_job_pipelines
        from videoannotator.pipelines.vlm_annotation import vlm_pipeline

        FakeVLM.reset()
        FakeVLM.hold_run_call = 2
        monkeypatch.setattr(vlm_pipeline, "OllamaVLMClient", FakeVLM)
        monkeypatch.setattr(ask, "OllamaVLMClient", FakeTryVLM)

        entry = _run(video, sampling={"interval": 1}).json()
        storage = get_storage_backend()
        job = storage.load_job_metadata(entry["job_id"])
        runner = threading.Thread(
            target=run_job_pipelines,
            args=(job, storage, {"vlm_annotation": vlm_pipeline.VLMAnnotationPipeline}),
        )
        runner.start()
        try:
            assert FakeVLM.held.wait(10)
            tried = _try(video)
            assert tried.status_code == 201, tried.text
            deadline = time.monotonic() + 5
            while ask_threads_waiting() == 0 and time.monotonic() < deadline:
                time.sleep(0.01)
        finally:
            FakeVLM.release.set()
        _until_done(tried.json()["id"])
        runner.join(20)
        assert FakeVLM.log[:3] == ["run-1", "run-2", "try"]


def ask_threads_waiting() -> int:
    from videoannotator.pipelines.vlm_annotation.gate import gate_for

    return gate_for(LOCAL, MODEL)._tries_waiting
