"""/videos/notes: researchers' notes on a video (spec 028 contracts/api.md)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.middleware.auth import validate_api_key
from videoannotator.batch.types import BatchJob, JobStatus

ALICE = {"id": "u1", "username": "alice", "is_admin": False}
BOB = {"id": "u2", "username": "bob", "is_admin": False}

client = TestClient(app)


@pytest.fixture(autouse=True)
def storage(tmp_path, monkeypatch):
    import videoannotator.database.database as db_module

    reset_storage_backend()
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    db_module.Base.metadata.create_all(bind=db_module.engine)
    app.dependency_overrides[validate_api_key] = lambda: ALICE
    yield
    app.dependency_overrides.pop(validate_api_key, None)
    reset_storage_backend()


@pytest.fixture
def video(tmp_path) -> str:
    path = tmp_path / "elsewhere" / "p01_t1.mp4"
    path.parent.mkdir()
    path.write_bytes(b"not decodable")
    get_storage_backend().save_job_metadata(
        BatchJob(
            video_path=path, output_dir=tmp_path / "out", status=JobStatus.COMPLETED
        )
    )
    return str(path)


def _create(video, **body):
    return client.post(
        "/api/v1/videos/notes",
        json={"path": video, "start_sec": 1.0, "text": "infant looks away"} | body,
    )


def test_create_list_update_delete(video):
    created = _create(video)
    assert created.status_code == 201, created.text
    note = created.json()
    assert note["author_name"] == "alice"
    assert note["end_sec"] is None

    listed = client.get("/api/v1/videos/notes", params={"path": video}).json()
    assert [n["id"] for n in listed["notes"]] == [note["id"]]

    app.dependency_overrides[validate_api_key] = lambda: BOB
    patched = client.patch(
        f"/api/v1/videos/notes/{note['id']}", json={"text": "looks back", "end_sec": 3}
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["text"] == "looks back"
    assert patched.json()["end_sec"] == 3
    assert patched.json()["author_id"] == "u1"
    assert patched.json()["updated_by"] == "u2"

    assert client.delete(f"/api/v1/videos/notes/{note['id']}").status_code == 204
    assert client.get("/api/v1/videos/notes", params={"path": video}).json() == {
        "notes": []
    }


@pytest.mark.parametrize(
    ("body", "code"),
    [
        ({"text": "  "}, "NOTE_TEXT_REQUIRED"),
        ({"text": "x" * 2001}, "NOTE_TEXT_TOO_LONG"),
        ({"start_sec": -2}, "NOTE_TIME_INVALID"),
        ({"start_sec": 4, "end_sec": 4}, "NOTE_TIME_INVALID"),
    ],
)
def test_notes_that_cant_be_saved(video, body, code):
    r = _create(video, **body)
    assert r.status_code == 422
    assert r.json()["error"]["code"] == code


def test_a_note_that_does_not_exist(video):
    missing = "00000000-0000-0000-0000-000000000000"
    assert (
        client.patch(f"/api/v1/videos/notes/{missing}", json={"text": "x"}).json()[
            "error"
        ]["code"]
        == "NOTE_NOT_FOUND"
    )
    assert client.delete(f"/api/v1/videos/notes/{missing}").status_code == 404


def test_a_path_no_job_read_outside_the_shared_folders(tmp_path):
    private = tmp_path / "private.mp4"
    private.write_bytes(b"v")
    assert _create(str(private)).status_code == 403
    r = client.get("/api/v1/videos/notes", params={"path": str(private)})
    assert r.json()["error"]["code"] == "PATH_NOT_ALLOWED"


def test_needs_the_api_key(video, monkeypatch):
    app.dependency_overrides.pop(validate_api_key, None)
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    assert client.get("/api/v1/videos/notes", params={"path": video}).status_code == 401
