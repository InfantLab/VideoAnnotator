"""API tests for server boot identity and self-restart (specs/011-pipeline-readiness).

The restart itself (graceful uvicorn shutdown, then re-launch from the CLI) needs
a real process and is covered by tests/manual/pipeline_readiness_e2e.md. Here a
fake uvicorn server stands in: `POST /system/restart` succeeding means it asked
that server to exit and flagged the re-launch.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from videoannotator.api import extras_install, restart
from videoannotator.api.main import app
from videoannotator.api.middleware.auth import validate_required_api_key
from videoannotator.api.v1 import system as system_module

client = TestClient(app)

ADMIN_USER = {"id": "admin-1", "username": "root", "is_admin": True}
NON_ADMIN_USER = {"id": "user-1", "username": "irene", "is_admin": False}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    app.dependency_overrides[validate_required_api_key] = lambda: ADMIN_USER
    monkeypatch.setattr(restart, "_server", None)
    monkeypatch.setattr(restart, "_restart_requested", False)
    monkeypatch.delenv(restart.RESTART_MODE_ENV, raising=False)
    extras_install._in_flight.clear()
    yield
    app.dependency_overrides.clear()
    extras_install._in_flight.clear()


@pytest.fixture
def restartable(monkeypatch):
    """As if started by `videoannotator server` (single worker, no reload)."""
    fake_server = SimpleNamespace(should_exit=False)
    monkeypatch.setattr(restart, "_server", fake_server)
    return fake_server


def _running_jobs(job_ids):
    backend = MagicMock()
    backend.list_jobs.return_value = job_ids
    return patch.object(system_module, "get_storage_backend", return_value=backend)


class TestBootIdentity:
    @pytest.mark.parametrize(
        "path", ["/health", "/api/v1/health", "/api/v1/system/health"]
    )
    def test_every_health_endpoint_reports_it(self, path):
        body = client.get(path).json()
        assert body["boot_id"] == restart.BOOT_ID
        assert body["started_at"] == restart.STARTED_AT
        assert body["restart_mode"] == "unsupported"

    def test_restart_mode_reflects_how_the_server_was_started(self, restartable):
        assert client.get("/health").json()["restart_mode"] == "execv"

    def test_supervisor_mode_is_opt_in(self, monkeypatch):
        monkeypatch.setenv(restart.RESTART_MODE_ENV, "exit_for_supervisor")
        assert client.get("/health").json()["restart_mode"] == "exit_for_supervisor"


class TestRestartEndpoint:
    def test_refused_when_the_server_cannot_restart_itself(self):
        resp = client.post("/api/v1/system/restart")
        assert resp.status_code == 409
        error = resp.json()["error"]
        assert error["code"] == "RESTART_UNSUPPORTED"
        assert "videoannotator server" in error["hint"]

    def test_non_admin_is_forbidden(self, restartable):
        app.dependency_overrides[validate_required_api_key] = lambda: NON_ADMIN_USER
        assert client.post("/api/v1/system/restart").status_code == 403
        assert restartable.should_exit is False

    def test_refused_while_an_install_runs_even_with_force(self, restartable):
        extras_install.try_begin_install("face", "install-1")
        with _running_jobs([]):
            resp = client.post("/api/v1/system/restart?force=true")
        assert resp.status_code == 409
        error = resp.json()["error"]
        assert error["code"] == "INSTALL_IN_PROGRESS"
        assert error["details"]["install_job_ids"] == ["install-1"]
        assert restartable.should_exit is False

    def test_refused_while_jobs_run_unless_forced(self, restartable):
        with _running_jobs(["job-a", "job-b"]):
            resp = client.post("/api/v1/system/restart")
        assert resp.status_code == 409
        error = resp.json()["error"]
        assert error["code"] == "JOBS_RUNNING"
        assert error["details"]["job_ids"] == ["job-a", "job-b"]
        assert restartable.should_exit is False

    def test_force_restarts_despite_running_jobs(self, restartable):
        with _running_jobs(["job-a"]):
            resp = client.post("/api/v1/system/restart?force=true")
        assert resp.status_code == 202
        assert restartable.should_exit is True

    def test_accepted_restart_stops_the_server_and_flags_relaunch(self, restartable):
        with _running_jobs([]):
            resp = client.post("/api/v1/system/restart")
        assert resp.status_code == 202
        assert resp.json() == {"restarting": True, "boot_id": restart.BOOT_ID}
        assert restartable.should_exit is True
        assert restart.restart_requested() is True


class TestRelaunchCommand:
    def test_reruns_the_same_cli_arguments_via_the_module(self, monkeypatch):
        monkeypatch.setattr(
            restart.sys,
            "argv",
            ["/venv/bin/videoannotator", "server", "--port", "18099"],
        )
        assert restart.relaunch_command() == [
            restart.sys.executable,
            "-m",
            "videoannotator.cli",
            "server",
            "--port",
            "18099",
        ]
