"""Installed pipelines survive the container being recreated (spec 024, R7).

A completed install is remembered in the database; at server start a group
whose packages are gone (a new container) is installed again, the pipeline
card says "restoring", and jobs that need it wait rather than fail. No test
runs a real install: `start_install` is always replaced.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from videoannotator.api import extras_install, readiness
from videoannotator.api.background_tasks import BackgroundJobManager
from videoannotator.batch.types import BatchJob, JobStatus
from videoannotator.database.models import ExtrasInstallJob, ExtrasInstallJobStatus
from videoannotator.registry.pipeline_registry import get_registry


@pytest.fixture
def db(monkeypatch, tmp_path):
    import videoannotator.database.database as db_module

    engine = create_engine(
        f"sqlite:///{tmp_path / 'restore.db'}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(db_module, "engine", engine)
    monkeypatch.setattr(db_module, "SessionLocal", factory)
    db_module.Base.metadata.create_all(bind=engine)
    session = factory()
    yield session
    session.close()


@pytest.fixture(autouse=True)
def clean_state():
    def reset():
        extras_install._in_flight.clear()
        extras_install._restoring.clear()
        extras_install._restore_failed.clear()

    reset()
    yield
    reset()


def _row(db, extra, status):
    db.add(ExtrasInstallJob(extra_name=extra, status=status))
    db.commit()


@pytest.fixture
def started(monkeypatch):
    calls = []
    monkeypatch.setattr(
        extras_install, "start_install", lambda job_id, extra: calls.append(extra)
    )
    return calls


def test_only_completed_installs_are_remembered(db):
    _row(db, "scene", ExtrasInstallJobStatus.COMPLETED)
    _row(db, "scene", ExtrasInstallJobStatus.COMPLETED)
    _row(db, "face", ExtrasInstallJobStatus.FAILED)
    _row(db, "audio", ExtrasInstallJobStatus.PENDING)
    assert extras_install.remembered_groups() == ["scene"]


def test_a_missing_group_is_installed_again_once(db, started, monkeypatch):
    _row(db, "scene", ExtrasInstallJobStatus.COMPLETED)
    _row(db, "person", ExtrasInstallJobStatus.COMPLETED)
    monkeypatch.setattr(
        extras_install, "group_importable", lambda extra: extra == "person"
    )

    assert extras_install.restore_missing_groups() == ["scene"]

    assert started == ["scene"]
    assert extras_install.restoring("scene")
    assert not extras_install.restoring("person")
    rows = db.query(ExtrasInstallJob).filter_by(extra_name="scene").all()
    assert len(rows) == 2
    restore = next(r for r in rows if r.status == ExtrasInstallJobStatus.PENDING)
    assert "Restoring" in restore.command_output


def test_nothing_to_restore(db, started, monkeypatch):
    _row(db, "scene", ExtrasInstallJobStatus.COMPLETED)
    monkeypatch.setattr(extras_install, "group_importable", lambda extra: True)
    assert extras_install.restore_missing_groups() == []
    assert started == []


def test_restoring_all_covers_every_group():
    extras_install._restoring.add("all")
    assert extras_install.restoring("scene")


def test_a_finished_restore_is_no_longer_restoring(db, monkeypatch):
    _row(db, "scene", ExtrasInstallJobStatus.COMPLETED)
    monkeypatch.setattr(extras_install, "group_importable", lambda extra: False)
    with patch.object(extras_install, "start_install"):
        extras_install.restore_missing_groups()
    job = (
        db.query(ExtrasInstallJob)
        .filter_by(status=ExtrasInstallJobStatus.PENDING)
        .one()
    )
    failed = MagicMock(returncode=1, stdout="", stderr="No space left on device")
    with (
        patch.object(
            extras_install, "resolve_install_command", return_value=(["x"], None)
        ),
        patch.object(extras_install.subprocess, "run", return_value=failed),
        patch.object(extras_install, "_installed_distributions", return_value=({}, {})),
    ):
        extras_install.run_install(str(job.id), "scene")

    assert not extras_install.restoring("scene")
    assert "No space left on device" in extras_install.restore_failure("scene")


def test_the_card_says_restoring(monkeypatch):
    meta = get_registry().get("scene_detection")
    extras_install._restoring.add("scene")
    monkeypatch.setattr(readiness, "extras_available", lambda extras: False)
    body = readiness.pipeline_readiness(meta)
    assert body["state"] == "restoring"
    assert body["next_action"] == "wait"


def _manager(jobs: dict[str, BatchJob]):
    storage = MagicMock()
    storage.list_jobs.side_effect = lambda status_filter=None: [
        job_id for job_id, job in jobs.items() if job.status.value == status_filter
    ]
    storage.load_job_metadata.side_effect = lambda job_id: jobs[job_id]
    manager = BackgroundJobManager(storage_backend=storage, max_concurrent_jobs=4)
    started = []

    async def process(job):
        started.append(job.job_id)

    manager._process_job_async = process
    return manager, storage, started


def _cycle(manager):
    async def run():
        await manager._process_cycle()
        await asyncio.sleep(0)

    asyncio.run(run())


def test_a_job_needing_a_restoring_group_waits_then_runs():
    scene = BatchJob(selected_pipelines=["scene_detection"], status=JobStatus.PENDING)
    other = BatchJob(selected_pipelines=["face_analysis"], status=JobStatus.PENDING)
    manager, _, started = _manager({scene.job_id: scene, other.job_id: other})
    extras_install._restoring.add("scene")

    _cycle(manager)
    assert started == [other.job_id]
    assert scene.status == JobStatus.PENDING

    extras_install._restoring.discard("scene")
    _cycle(manager)
    assert scene.job_id in started


def test_a_failed_restore_fails_the_waiting_job_with_its_reason():
    scene = BatchJob(selected_pipelines=["scene_detection"], status=JobStatus.PENDING)
    manager, storage, started = _manager({scene.job_id: scene})
    extras_install._restore_failed["scene"] = "No space left on device"

    _cycle(manager)

    assert started == []
    assert scene.status == JobStatus.FAILED
    assert "No space left on device" in scene.error_message
    assert "scene" in scene.error_message
    storage.save_job_metadata.assert_called_with(scene)
