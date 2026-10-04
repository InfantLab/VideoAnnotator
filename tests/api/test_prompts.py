"""The prompt library (spec 020): every prompt that runs, kept once, findable."""

import base64
import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.synthetic_video import synthetic_video_bytes_avi
from videoannotator import prompt_library
from videoannotator.api.database import reset_storage_backend
from videoannotator.api.main import app
from videoannotator.database import database as db_module
from videoannotator.database.models import Prompt, PromptUse
from videoannotator.pipelines.vlm_annotation.ollama_client import VLMCallResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_library():
    reset_storage_backend()
    db_module.Base.metadata.create_all(bind=db_module.engine)
    with db_module.SessionLocal() as db:
        db.query(PromptUse).delete()
        db.query(Prompt).delete()
        db.commit()
    yield
    reset_storage_backend()


def _use(text, model="gemma4:e4b", kind="preview", job_id=None):
    with db_module.SessionLocal() as db:
        prompt_library.record_use(db, text, model, kind, job_id=job_id)


def test_the_same_text_is_kept_once_with_every_use():
    _use("Is the adult touching the infant?")
    _use(
        "Is the adult touching the infant?", model="qwen3.5:9b", kind="job", job_id="j1"
    )
    _use("Is the adult touching the infant? ")  # trailing space: a different prompt

    prompts = client.get("/api/v1/prompts/").json()["prompts"]
    assert len(prompts) == 2
    first = next(p for p in prompts if p["text"] == "Is the adult touching the infant?")
    assert first["sha256"] == prompt_library.prompt_sha256(first["text"])
    assert first["models"] == ["gemma4:e4b", "qwen3.5:9b"]
    assert first["job_ids"] == ["j1"]
    assert first["use_count"] == 2


def test_search_name_star_tag_and_hide():
    _use("Describe the scene.")
    _use("Count the faces.")
    sha = prompt_library.prompt_sha256("Count the faces.")
    client.put(
        f"/api/v1/prompts/{sha}",
        json={"name": "faces", "starred": True, "tags": ["count", " "]},
    )

    prompts = client.get("/api/v1/prompts/").json()["prompts"]
    assert prompts[0]["sha256"] == sha  # starred first
    assert prompts[0]["tags"] == ["count"]
    assert [
        p["text"]
        for p in client.get("/api/v1/prompts/", params={"q": "FACES"}).json()["prompts"]
    ] == ["Count the faces."]
    assert client.get("/api/v1/prompts/", params={"tag": "count"}).json()["total"] == 1
    assert client.get("/api/v1/prompts/", params={"model": "nope"}).json()["total"] == 0

    client.put(f"/api/v1/prompts/{sha}", json={"hidden": True})
    assert client.get("/api/v1/prompts/").json()["total"] == 1
    assert (
        client.get("/api/v1/prompts/", params={"include_hidden": True}).json()["total"]
        == 2
    )
    assert client.get(f"/api/v1/prompts/{sha[:10]}").json()["name"] == "faces"


def test_a_prompt_jobs_used_cannot_be_deleted():
    _use("Used by a job.", kind="job", job_id="j1")
    _use("Only previewed.")
    used = prompt_library.prompt_sha256("Used by a job.")
    response = client.delete(f"/api/v1/prompts/{used}")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "PROMPT_USED_BY_JOBS"
    assert (
        client.delete(
            f"/api/v1/prompts/{prompt_library.prompt_sha256('Only previewed.')}"
        ).status_code
        == 204
    )


def test_a_preview_is_recorded_and_returns_the_frames_it_used():
    upload = client.post(
        "/api/v1/jobs/",
        files={
            "video": ("t.avi", io.BytesIO(synthetic_video_bytes_avi()), "video/avi")
        },
    ).json()
    fake = VLMCallResult(
        raw_text="LABEL: TOUCH", thinking="", total_time=1, load_time=0,
        prompt_tokens=1, resp_tokens=1, tokens_per_sec=1,
    )  # fmt: skip
    with (
        patch("videoannotator.api.v1.vlm.OllamaVLMClient.__init__", return_value=None),
        patch("videoannotator.api.v1.vlm.OllamaVLMClient.chat", return_value=fake),
    ):
        response = client.post(
            "/api/v1/vlm/preview",
            data={
                "video_path": upload["video_path"],
                "timestamp_sec": "0.5",
                "prompt": "Touch?",
                "model": "gemma4:e4b",
                "sampling_mode": "frame_burst",
                "burst_offsets": "[-1,0,1]",
                "frame_interval_sec": "1.0",
            },
        )
    assert response.status_code == 200, response.text
    frames = response.json()["frames"]
    # A burst near the start is clamped to the frames that exist.
    assert len(frames) >= 2
    assert len({f["frame_number"] for f in frames}) == len(frames)
    assert base64.b64decode(frames[0]["jpeg_base64"])[:2] == b"\xff\xd8"  # a JPEG
    assert frames[1]["timestamp_sec"] is not None

    prompts = client.get("/api/v1/prompts/").json()["prompts"]
    assert [(p["text"], p["models"]) for p in prompts] == [("Touch?", ["gemma4:e4b"])]
