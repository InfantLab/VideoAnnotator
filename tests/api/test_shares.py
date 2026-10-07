"""Shared folders in Settings, and Stop sharing (spec 024, research R10).

The server can't change what is shared: it leaves a request in the launcher's
requests folder, which can only ever make sharing narrower.
"""

import os

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.main import app
from videoannotator.api.middleware.auth import (
    validate_api_key,
    validate_required_api_key,
)
from videoannotator.api.v1 import ingest as ingest_module

ADMIN = {"id": "admin-1", "username": "root", "is_admin": True}
RESEARCHER = {"id": "user-1", "username": "irene", "is_admin": False}

local = TestClient(app)
remote = TestClient(app, client=("192.168.1.20", 50000))


@pytest.fixture(autouse=True)
def launcher(tmp_path, monkeypatch):
    """Two folders shared by the launcher, at Windows-style host paths, and one
    missing at this start."""
    studies, second = tmp_path / "c" / "Studies", tmp_path / "c" / "Second"
    studies.mkdir(parents=True)
    second.mkdir()
    requests = tmp_path / "requests"
    requests.mkdir()
    monkeypatch.setattr(ingest_module, "in_container", lambda: True)
    monkeypatch.setattr(ingest_module, "INGEST_ROOTS", f"{studies}{os.pathsep}{second}")
    monkeypatch.setattr(ingest_module, "LAUNCHER_REQUESTS_DIR", requests)
    monkeypatch.setenv("VIDEOANNOTATOR_LAUNCHER", "1")
    monkeypatch.setenv("VIDEOANNOTATOR_MISSING_SHARES", "E:\\Data")
    monkeypatch.setenv(
        "VIDEOANNOTATOR_HOST_PATHS",
        f"{studies}=C:\\Users\\ada\\Studies;{second}=C:\\Users\\ada\\Second",
    )
    monkeypatch.delenv("VIDEOANNOTATOR_RESULTS_OWNER", raising=False)
    app.dependency_overrides[validate_api_key] = lambda: ADMIN
    app.dependency_overrides[validate_required_api_key] = lambda: ADMIN
    yield {"studies": studies, "second": second, "requests": requests}
    app.dependency_overrides.clear()


def _shares():
    response = local.get("/api/v1/ingest/access")
    assert response.status_code == 200, response.text
    return response.json()["shares"]


def _stop(path, client=local):
    return client.post("/api/v1/ingest/shares/stop", json={"path": path})


def test_access_lists_every_share_present_or_not(launcher):
    assert _shares() == [
        {
            "path": str(launcher["studies"]),
            "display_path": "C:\\Users\\ada\\Studies",
            "present": True,
            "stop_requested": False,
        },
        {
            "path": str(launcher["second"]),
            "display_path": "C:\\Users\\ada\\Second",
            "present": True,
            "stop_requested": False,
        },
        {
            "path": "E:\\Data",
            "display_path": "E:\\Data",
            "present": False,
            "stop_requested": False,
        },
    ]


def test_stop_sharing_leaves_a_request_for_the_next_start(launcher):
    response = _stop("C:\\Users\\ada\\Studies")
    assert response.status_code == 200, response.text
    assert response.json()["stop_requested"] is True
    assert response.json()["display_path"] == "C:\\Users\\ada\\Studies"
    requests = launcher["requests"] / "stop-sharing.txt"
    assert requests.read_text(encoding="utf-8") == "C:\\Users\\ada\\Studies\n"
    assert [s["stop_requested"] for s in _shares()] == [True, False, False]


def test_a_share_can_be_named_by_its_container_path(launcher):
    assert _stop(str(launcher["second"])).status_code == 200
    text = (launcher["requests"] / "stop-sharing.txt").read_text(encoding="utf-8")
    assert text == "C:\\Users\\ada\\Second\n"


def test_a_missing_share_can_be_stopped_too(launcher):
    assert _stop("E:\\Data").status_code == 200


def test_asking_twice_writes_it_once(launcher):
    _stop("C:\\Users\\ada\\Studies")
    _stop("C:\\Users\\ada\\Studies")
    text = (launcher["requests"] / "stop-sharing.txt").read_text(encoding="utf-8")
    assert text.splitlines() == ["C:\\Users\\ada\\Studies"]


def test_only_a_current_share(launcher):
    response = _stop("C:\\Windows")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "SHARE_NOT_FOUND"
    assert not (launcher["requests"] / "stop-sharing.txt").exists()


def test_only_with_the_launcher(launcher, monkeypatch):
    monkeypatch.delenv("VIDEOANNOTATOR_LAUNCHER")
    response = _stop("C:\\Users\\ada\\Studies")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "NOT_MANAGED_BY_LAUNCHER"


def test_only_an_administrator(launcher):
    app.dependency_overrides[validate_required_api_key] = lambda: RESEARCHER
    assert _stop("C:\\Users\\ada\\Studies").status_code == 403


def test_only_on_this_machine(launcher):
    assert _stop("C:\\Users\\ada\\Studies", client=remote).status_code == 403


def test_the_request_belongs_to_the_researcher(launcher, monkeypatch):
    calls = []
    monkeypatch.setenv("VIDEOANNOTATOR_RESULTS_OWNER", "1000:1000")
    monkeypatch.setattr(
        "videoannotator.results_folder.os.chown",
        lambda path, uid, gid: calls.append((str(path), uid, gid)),
        raising=False,
    )
    _stop("C:\\Users\\ada\\Studies")
    assert (str(launcher["requests"] / "stop-sharing.txt"), 1000, 1000) in calls


def test_no_shares_outside_the_launcher_or_with_the_home_default(monkeypatch):
    monkeypatch.setattr(ingest_module, "in_container", lambda: False)
    monkeypatch.setattr(ingest_module, "INGEST_ROOTS", "")
    monkeypatch.delenv("VIDEOANNOTATOR_MISSING_SHARES")
    assert _shares() == []
