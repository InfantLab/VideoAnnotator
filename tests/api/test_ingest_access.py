"""`GET /api/v1/ingest/access`: what this caller may do with local videos (spec 022)."""

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.main import app
from videoannotator.api.middleware.auth import validate_api_key
from videoannotator.api.v1 import ingest as ingest_module
from videoannotator.results_folder import results_root

ADMIN = {"id": "admin-1", "username": "root", "is_admin": True}
RESEARCHER = {"id": "user-1", "username": "irene", "is_admin": False}

local = TestClient(app)
remote = TestClient(app, client=("192.168.1.20", 50000))


@pytest.fixture(autouse=True)
def as_admin(monkeypatch):
    app.dependency_overrides[validate_api_key] = lambda: ADMIN
    monkeypatch.setattr(ingest_module, "in_container", lambda: False)
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def videos(ingest_root):
    (ingest_root / "child01.mp4").write_bytes(b"v")
    return ingest_root


def _access(client=local):
    response = client.get("/api/v1/ingest/access")
    assert response.status_code == 200, response.text
    return response.json()


def test_an_admin_on_this_machine_may_read_in_place(videos):
    body = _access()
    assert body["same_machine"] is True
    assert body["can_read_in_place"] is True
    assert body["reason"] is None
    assert body["allowed_folders"] == [
        {"path": str(videos.resolve()), "display_path": str(videos.resolve())}
    ]
    assert body["results_root"]["path"] == str(results_root())


def test_another_machine_is_not_offered_in_place_reading(videos):
    body = _access(remote)
    assert body["same_machine"] is False
    assert body["can_read_in_place"] is False
    assert "another computer" in body["reason"]
    assert body["can_open_folders"] is False


def test_a_non_admin_is_told_why(videos):
    app.dependency_overrides[validate_api_key] = lambda: RESEARCHER
    body = _access()
    assert body["same_machine"] is True
    assert body["can_read_in_place"] is False
    assert "administrator" in body["reason"]


def test_published_locally_makes_every_caller_local(videos, monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_PUBLISHED_LOCALLY", "1")
    body = _access(remote)
    assert body["same_machine"] is True
    assert body["can_read_in_place"] is True


def test_no_usable_video_folder(ingest_root):
    body = _access()
    assert body["can_read_in_place"] is False
    assert "VIDEOANNOTATOR_INGEST_ROOTS" in body["reason"]


def test_no_video_folder_under_docker_says_how_to_set_one(ingest_root, monkeypatch):
    monkeypatch.setattr(ingest_module, "in_container", lambda: True)
    body = _access()
    assert body["can_read_in_place"] is False
    assert "VIDEOS_DIR" in body["reason"]


def test_docker_shows_host_paths_and_cannot_open_folders(videos, monkeypatch):
    monkeypatch.setattr(ingest_module, "in_container", lambda: True)
    monkeypatch.setattr(ingest_module, "folder_opener", lambda: None)
    monkeypatch.setenv("VIDEOANNOTATOR_RESULTS_DIR", "/results")
    monkeypatch.setenv(
        "VIDEOANNOTATOR_HOST_PATHS",
        f"{videos.resolve()}=/home/ada/Studies;/results=/home/ada/VideoAnnotator",
    )
    body = _access()
    assert body["allowed_folders"][0]["display_path"] == "/home/ada/Studies"
    assert body["results_root"]["display_path"] == "/home/ada/VideoAnnotator"
    assert body["can_open_folders"] is False


def test_can_open_folders_needs_a_desktop(videos, monkeypatch):
    monkeypatch.setattr(ingest_module, "folder_opener", lambda: ["xdg-open"])
    assert _access()["can_open_folders"] is True
    monkeypatch.setattr(ingest_module, "folder_opener", lambda: None)
    assert _access()["can_open_folders"] is False
