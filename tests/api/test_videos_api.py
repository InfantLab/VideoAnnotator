"""Videos for Ask AI: upload once, play with a short-lived token (spec 027 R12)."""

import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.synthetic_video import SyntheticVideoSpec, synthetic_video_bytes_avi
from videoannotator.api.database import reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.v1 import videos

client = TestClient(app)


@pytest.fixture(autouse=True)
def storage(tmp_path, monkeypatch):
    reset_storage_backend()
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    yield tmp_path / "storage"
    reset_storage_backend()


def _upload(content: bytes, name: str = "p01_t1.avi"):
    return client.post(
        "/api/v1/videos", files={"video": (name, content, "video/x-msvideo")}
    )


def _token(path: str):
    return client.post("/api/v1/videos/stream-token", json={"path": path})


def test_upload_stores_identical_content_once(storage):
    content = synthetic_video_bytes_avi()
    first = _upload(content)
    assert first.status_code == 201, first.text
    second = _upload(content, name="renamed.avi")
    body = first.json()
    assert second.json()["path"] == body["path"]
    assert body["name"] == "p01_t1.avi"
    assert body["path"].startswith(str((storage / "videos").resolve()))
    assert Path(body["path"]).parent.name == body["sha256"][:16]
    assert body["duration_sec"] == pytest.approx(2.0, abs=0.2)
    assert len([p for p in (storage / "videos").rglob("*") if p.is_file()]) == 1


def test_range_request_returns_the_right_bytes():
    content = synthetic_video_bytes_avi()
    path = _upload(content).json()["path"]
    token = _token(path).json()["token"]
    r = client.get(
        "/api/v1/videos/stream",
        params={"path": path, "token": token},
        headers={"Range": "bytes=10-109"},
    )
    assert r.status_code == 206
    assert r.content == content[10:110]


def test_token_for_one_path_is_refused_for_another():
    a = _upload(synthetic_video_bytes_avi()).json()["path"]
    b = _upload(synthetic_video_bytes_avi() + b"\0", name="b.avi").json()["path"]
    token = _token(a).json()["token"]
    r = client.get("/api/v1/videos/stream", params={"path": b, "token": token})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "TOKEN_INVALID"


def test_expired_token_is_refused():
    path = Path(_upload(synthetic_video_bytes_avi()).json()["path"])
    token, _ = videos.make_stream_token(path, now=time.time() - 2 * 3600)
    r = client.get("/api/v1/videos/stream", params={"path": str(path), "token": token})
    assert r.status_code == 401


def test_stream_without_token_is_refused():
    path = _upload(synthetic_video_bytes_avi()).json()["path"]
    assert client.get("/api/v1/videos/stream", params={"path": path}).status_code == 422
    r = client.get("/api/v1/videos/stream", params={"path": path, "token": "x"})
    assert r.status_code == 401


def test_path_outside_the_allowed_places_is_refused(tmp_path, ingest_root):
    outside = tmp_path / "elsewhere" / "v.avi"
    outside.parent.mkdir()
    outside.write_bytes(synthetic_video_bytes_avi())
    r = _token(str(outside))
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "PATH_NOT_ALLOWED"


def test_a_shared_folder_video_is_readable(ingest_root):
    video = ingest_root / "clip.avi"
    video.write_bytes(synthetic_video_bytes_avi())
    r = _token(str(video))
    assert r.status_code == 200
    assert r.json()["name"] == "clip.avi"
    assert r.json()["duration_sec"] == pytest.approx(2.0, abs=0.2)
    assert r.json()["fps"] == pytest.approx(SyntheticVideoSpec().fps)


def test_missing_video_in_a_shared_folder_is_not_found(ingest_root):
    r = _token(str(ingest_root / "gone.avi"))
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "VIDEO_NOT_FOUND"
