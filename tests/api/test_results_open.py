"""`POST /api/v1/results/open`: show a results folder on this computer (spec 022)."""

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.main import app
from videoannotator.api.v1 import results as results_module

local = TestClient(app)
remote = TestClient(app, client=("192.168.1.20", 50000))


@pytest.fixture
def run_folder(results_root):
    folder = results_root / "Wave 2 (2026-10-06)"
    folder.mkdir(parents=True)
    return folder


@pytest.fixture
def opened(monkeypatch):
    calls = []
    monkeypatch.setattr(results_module, "open_folder", calls.append)
    return calls


def _open(path, client=local):
    return client.post("/api/v1/results/open", json={"path": str(path)})


def test_opens_a_folder_in_the_results_root(run_folder, opened):
    assert _open(run_folder).status_code == 204
    assert opened == [run_folder.resolve()]


def test_the_results_root_itself_can_be_opened(run_folder, results_root, opened):
    assert _open(run_folder / "..").status_code == 204
    assert opened == [results_root.resolve()]


def test_only_from_this_machine(run_folder, opened):
    response = _open(run_folder, remote)
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "NOT_SAME_MACHINE"
    assert opened == []


@pytest.mark.parametrize("escape", ["../..", "../../..", "/etc"])
def test_nothing_outside_the_results_root(run_folder, opened, escape):
    target = escape if escape.startswith("/") else run_folder / escape
    response = _open(target)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "PATH_OUTSIDE_RESULTS"
    assert opened == []


def test_a_symlink_out_of_the_results_root_is_refused(run_folder, tmp_path, opened):
    outside = tmp_path / "private"
    outside.mkdir()
    (run_folder / "sneaky").symlink_to(outside)
    response = _open(run_folder / "sneaky")
    assert response.status_code == 422
    assert opened == []


def test_no_desktop_says_so(run_folder, monkeypatch):
    def fail(path):
        raise OSError("no desktop to open folders on")

    monkeypatch.setattr(results_module, "open_folder", fail)
    response = _open(run_folder)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "OPEN_FOLDER_UNSUPPORTED"


def test_a_moved_folder_is_named(results_root, opened):
    response = _open(results_root / "Renamed (2026-10-06)")
    assert response.status_code == 404
    assert "Renamed (2026-10-06)" in response.json()["error"]["message"]
