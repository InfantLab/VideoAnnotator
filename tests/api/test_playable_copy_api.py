"""A copy of a video every browser can play (viewer design D7, UAT #54)."""

import shutil
import time

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.synthetic_video import synthetic_video_bytes_avi
from videoannotator import playable
from videoannotator.api.database import reset_storage_backend
from videoannotator.api.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def storage(tmp_path, monkeypatch):
    reset_storage_backend()
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    yield tmp_path / "storage"
    reset_storage_backend()


@pytest.fixture
def uploaded():
    r = client.post(
        "/api/v1/videos",
        files={"video": ("clip.avi", synthetic_video_bytes_avi(), "video/x-msvideo")},
    )
    assert r.status_code == 201, r.text
    return r.json()["path"]


def _state(path: str) -> dict:
    r = client.get("/api/v1/videos/playable-copy", params={"path": path})
    assert r.status_code == 200, r.text
    return r.json()


def _token(path: str) -> dict:
    return client.post("/api/v1/videos/stream-token", json={"path": path}).json()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_makes_a_copy_and_streams_it(uploaded):
    assert _state(uploaded)["state"] == "none"
    assert _token(uploaded)["playable_copy"] == "none"

    r = client.post("/api/v1/videos/playable-copy", json={"path": uploaded})
    assert r.json()["state"] in ("making", "ready")
    for _ in range(120):
        if _state(uploaded)["state"] == "ready":
            break
        time.sleep(0.25)
    assert _state(uploaded)["state"] == "ready"

    token = _token(uploaded)
    assert token["playable_copy"] == "ready"
    r = client.get(
        "/api/v1/videos/stream",
        params={"path": uploaded, "token": token["token"], "copy": "true"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"
    assert len(r.content) > 0


def test_no_copy_to_stream_yet(uploaded):
    token = _token(uploaded)
    r = client.get(
        "/api/v1/videos/stream",
        params={"path": uploaded, "token": token["token"], "copy": "true"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NO_PLAYABLE_COPY"


def test_without_ffmpeg_says_so(uploaded, monkeypatch):
    monkeypatch.setattr(playable, "ffmpeg", lambda: None)
    r = client.post("/api/v1/videos/playable-copy", json={"path": uploaded})
    assert r.json()["state"] == "unavailable"
    assert "ffmpeg" in r.json()["error"]


def test_a_failed_copy_says_why(uploaded, monkeypatch):
    monkeypatch.setattr(
        playable, "command", lambda source, target: ["false", str(source), str(target)]
    )
    client.post("/api/v1/videos/playable-copy", json={"path": uploaded})
    for _ in range(40):
        if _state(uploaded)["state"] == "failed":
            break
        time.sleep(0.1)
    found = _state(uploaded)
    assert found["state"] == "failed"
    assert found["error"].startswith("The copy couldn't be made")


def test_the_copy_keeps_every_frame_at_its_time(tmp_path):
    cmd = playable.command(tmp_path / "in.mp4", tmp_path / "out.mp4")
    assert cmd[cmd.index("-fps_mode") + 1] == "passthrough"
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
