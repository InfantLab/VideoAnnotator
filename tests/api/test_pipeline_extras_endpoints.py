"""API tests for the pipeline extras install endpoints (specs/005-pipeline-extras-install).

Covers User Story 1 (trigger + track an install), User Story 2 (restart-required
signal), and User Story 3 (auth/validation guardrails). No test here invokes a real
`pip install`/`uv sync` -- `extras_install.start_install`/`run_install` is always
mocked or driven directly against the DB row.
"""

import threading
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from videoannotator.api import extras_install
from videoannotator.api.main import create_app
from videoannotator.api.middleware.auth import validate_required_api_key
from videoannotator.database.models import ExtrasInstallJob, ExtrasInstallJobStatus

ADMIN_USER = {"id": "admin-1", "username": "admin", "is_admin": True}
NON_ADMIN_USER = {"id": "user-1", "username": "alice", "is_admin": False}


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Point the app's SQLAlchemy engine at a fresh, empty temp SQLite file."""
    import videoannotator.database.database as db_module

    db_path = tmp_path / "test_extras_install.db"
    new_engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    monkeypatch.setattr(db_module, "engine", new_engine)
    monkeypatch.setattr(
        db_module,
        "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=new_engine),
    )
    db_module.Base.metadata.create_all(bind=new_engine)
    yield new_engine


@pytest.fixture(autouse=True)
def _reset_extras_install_state():
    extras_install._in_flight.clear()
    extras_install._restart_required = False
    yield
    extras_install._in_flight.clear()
    extras_install._restart_required = False


@pytest.fixture
def client(temp_db):
    app = create_app()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def admin_client(client):
    client.app.dependency_overrides[validate_required_api_key] = lambda: ADMIN_USER
    return client


@pytest.fixture
def non_admin_client(client):
    client.app.dependency_overrides[validate_required_api_key] = lambda: NON_ADMIN_USER
    return client


@pytest.fixture
def db_session(temp_db):
    import videoannotator.database.database as db_module

    session = db_module.SessionLocal()
    yield session
    session.close()


class TestTriggerInstall:
    def test_returns_202_with_documented_shape(self, admin_client, db_session):
        with (
            patch.object(extras_install, "start_install") as mock_start,
            patch(
                "videoannotator.api.v1.pipelines.extras_available", return_value=False
            ),
        ):
            resp = admin_client.post("/api/v1/pipelines/extras/scene/install")

        assert resp.status_code == 202
        body = resp.json()
        assert body["extra_name"] == "scene"
        assert body["status"] == "pending"
        assert "job_id" in body
        mock_start.assert_called_once()

        job = (
            db_session.query(ExtrasInstallJob)
            .filter(ExtrasInstallJob.id == body["job_id"])
            .first()
        )
        assert job is not None
        assert job.status == ExtrasInstallJobStatus.PENDING

    def test_already_satisfied_fast_path(self, admin_client, db_session):
        with patch(
            "videoannotator.api.v1.pipelines.extras_available", return_value=True
        ):
            resp = admin_client.post("/api/v1/pipelines/extras/scene/install")

        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "completed"

        job = (
            db_session.query(ExtrasInstallJob)
            .filter(ExtrasInstallJob.id == body["job_id"])
            .first()
        )
        assert job.status == ExtrasInstallJobStatus.COMPLETED
        assert "already installed" in job.command_output

    def test_dedup_returns_same_job_for_in_flight_request(
        self, admin_client, db_session
    ):
        with (
            patch.object(extras_install, "start_install"),  # no-op: stays "in flight"
            patch(
                "videoannotator.api.v1.pipelines.extras_available", return_value=False
            ),
        ):
            first = admin_client.post("/api/v1/pipelines/extras/face/install")
            second = admin_client.post("/api/v1/pipelines/extras/face/install")

        assert first.status_code == 202
        assert second.status_code == 202
        assert first.json()["job_id"] == second.json()["job_id"]

    def test_unknown_extras_group_returns_422(self, admin_client):
        resp = admin_client.post("/api/v1/pipelines/extras/gpu-magic/install")
        assert resp.status_code == 422


class TestInstallJobStatus:
    def test_unknown_job_id_returns_404(self, admin_client):
        resp = admin_client.get("/api/v1/pipelines/extras/install-jobs/does-not-exist")
        assert resp.status_code == 404

    def test_polls_through_pending_running_completed(self, admin_client, db_session):
        job = ExtrasInstallJob(
            extra_name="scene", status=ExtrasInstallJobStatus.PENDING
        )
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)
        job_id = str(job.id)

        resp = admin_client.get(f"/api/v1/pipelines/extras/install-jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

        job.status = ExtrasInstallJobStatus.RUNNING
        db_session.commit()
        resp = admin_client.get(f"/api/v1/pipelines/extras/install-jobs/{job_id}")
        assert resp.json()["status"] == "running"

        job.status = ExtrasInstallJobStatus.COMPLETED
        db_session.commit()
        resp = admin_client.get(f"/api/v1/pipelines/extras/install-jobs/{job_id}")
        assert resp.json()["status"] == "completed"

    def test_failed_job_has_non_empty_command_output(self, admin_client, db_session):
        job = ExtrasInstallJob(
            extra_name="scene",
            status=ExtrasInstallJobStatus.FAILED,
            command_output="ERROR: something went wrong",
        )
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)

        resp = admin_client.get(f"/api/v1/pipelines/extras/install-jobs/{job.id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        assert body["command_output"] == "ERROR: something went wrong"


class TestRestartRequiredSignal:
    """Semantics per contracts/restart-required-signal.md."""

    def test_false_on_fresh_process(self, admin_client, db_session):
        job = ExtrasInstallJob(
            extra_name="scene", status=ExtrasInstallJobStatus.PENDING
        )
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)

        list_resp = admin_client.get("/api/v1/pipelines")
        assert list_resp.json()["restart_required"] is False

        job_resp = admin_client.get(f"/api/v1/pipelines/extras/install-jobs/{job.id}")
        assert job_resp.json()["restart_required"] is False

    def test_true_on_both_endpoints_after_a_completed_install(
        self, admin_client, db_session
    ):
        job = ExtrasInstallJob(
            extra_name="scene", status=ExtrasInstallJobStatus.PENDING
        )
        db_session.add(job)
        db_session.commit()
        db_session.refresh(job)

        extras_install._mark_restart_required()

        list_resp = admin_client.get("/api/v1/pipelines")
        assert list_resp.json()["restart_required"] is True

        job_resp = admin_client.get(f"/api/v1/pipelines/extras/install-jobs/{job.id}")
        assert job_resp.json()["restart_required"] is True

    def test_stays_true_across_a_subsequent_failed_job_for_a_different_extra(
        self, admin_client, db_session
    ):
        extras_install._mark_restart_required()

        failed_job = ExtrasInstallJob(
            extra_name="audio",
            status=ExtrasInstallJobStatus.FAILED,
            command_output="ERROR: network unreachable",
        )
        db_session.add(failed_job)
        db_session.commit()

        list_resp = admin_client.get("/api/v1/pipelines")
        assert list_resp.json()["restart_required"] is True

    def test_a_failed_only_job_never_sets_it_true(self, admin_client, db_session):
        with (
            patch.object(extras_install, "start_install") as mock_start,
            patch(
                "videoannotator.api.v1.pipelines.extras_available", return_value=False
            ),
        ):
            resp = admin_client.post("/api/v1/pipelines/extras/audio/install")
        job_id = resp.json()["job_id"]
        mock_start.assert_called_once()

        # Simulate run_install's failure branch directly (never calls
        # _mark_restart_required on the failed path).
        job = (
            db_session.query(ExtrasInstallJob)
            .filter(ExtrasInstallJob.id == job_id)
            .first()
        )
        job.status = ExtrasInstallJobStatus.FAILED
        job.command_output = "ERROR: something went wrong"
        db_session.commit()

        list_resp = admin_client.get("/api/v1/pipelines")
        assert list_resp.json()["restart_required"] is False


class TestInstallSafety:
    """User Story 3: rejected before any subprocess/job creation."""

    def test_unauthenticated_returns_401_and_creates_no_job(self, client, db_session):
        resp = client.post("/api/v1/pipelines/extras/scene/install")
        assert resp.status_code == 401
        assert db_session.query(ExtrasInstallJob).count() == 0

    def test_non_admin_returns_403_and_creates_no_job(
        self, non_admin_client, db_session
    ):
        resp = non_admin_client.post("/api/v1/pipelines/extras/scene/install")
        assert resp.status_code == 403
        assert db_session.query(ExtrasInstallJob).count() == 0

    def test_admin_with_unknown_extras_group_returns_422_and_never_starts_install(
        self, admin_client, db_session
    ):
        with patch.object(extras_install, "start_install") as mock_start:
            resp = admin_client.post("/api/v1/pipelines/extras/gpu-magic/install")
        assert resp.status_code == 422
        assert db_session.query(ExtrasInstallJob).count() == 0
        mock_start.assert_not_called()

    def test_unauthenticated_with_unknown_extras_group_still_returns_401_not_422(
        self, client
    ):
        # Auth is checked before extras-name validation (FR-002/FR-003 ordering).
        resp = client.post("/api/v1/pipelines/extras/gpu-magic/install")
        assert resp.status_code == 401

    def test_concurrent_requests_for_same_extra_create_only_one_job(
        self, admin_client, db_session
    ):
        results: list[int] = []

        def _fire():
            resp = admin_client.post("/api/v1/pipelines/extras/audio/install")
            results.append(resp.status_code)

        with (
            patch.object(extras_install, "start_install"),
            patch(
                "videoannotator.api.v1.pipelines.extras_available", return_value=False
            ),
        ):
            threads = [threading.Thread(target=_fire) for _ in range(5)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert all(code == 202 for code in results)
        assert (
            db_session.query(ExtrasInstallJob)
            .filter(ExtrasInstallJob.extra_name == "audio")
            .count()
            == 1
        )
