"""Researchers' notes in the table and ELAN exports (spec 028 R10)."""

from __future__ import annotations

import csv
import io
import json
import uuid
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.unit.exporters.conftest import FIXTURES, make_run
from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.middleware.auth import validate_api_key
from videoannotator.batch.types import BatchJob, JobStatus

ALICE = {"id": "u1", "username": "alice", "is_admin": False}


@pytest.fixture
def client():
    import videoannotator.database.database as db_module

    reset_storage_backend()
    db_module.Base.metadata.create_all(bind=db_module.engine)
    app.dependency_overrides[validate_api_key] = lambda: ALICE
    yield TestClient(app)
    app.dependency_overrides.clear()
    reset_storage_backend()


@pytest.fixture
def run(tmp_path):
    """A finished run of one video at a path no other test uses (notes persist)."""
    video = f"/data/notes-{uuid.uuid4().hex[:8]}/demo_clip.mp4"
    folder = make_run(tmp_path, {"demo_clip": FIXTURES})
    batch_id = str(uuid.uuid4())
    job = BatchJob(
        video_path=Path(video),
        output_dir=folder / "demo_clip",
        status=JobStatus.COMPLETED,
        batch_id=batch_id,
    )
    get_storage_backend().save_job_metadata(job)
    return video, batch_id, job


def _note(client, video, start, text, end=None):
    r = client.post(
        "/api/v1/videos/notes",
        json={"path": video, "start_sec": start, "end_sec": end, "text": text},
    )
    assert r.status_code == 201, r.text


def _rows(response):
    return list(csv.DictReader(io.StringIO(response.text)))


def _summary(response) -> dict:
    return json.loads(response.headers["x-videoannotator-export-summary"])


@pytest.mark.parametrize("scope", ["job", "batch"])
def test_tables_include_the_videos_notes(client, run, scope):
    video, batch_id, job = run
    _note(client, video, 1.5, "infant looks away")
    _note(client, video, 4.0, "peekaboo", end=6.0)
    url = (
        f"/api/v1/jobs/{job.job_id}/export/table"
        if scope == "job"
        else f"/api/v1/batches/{batch_id}/export/table"
    )
    response = client.get(url)
    assert response.status_code == 200, response.text
    notes = [r for r in _rows(response) if r["pipeline"] == "notes"]
    assert [(n["label"], n["author"], n["run"]) for n in notes] == [
        ("infant looks away", "alice", ""),
        ("peekaboo", "alice", ""),
    ]
    summary = _summary(response)
    assert summary["table_format_version"] == 2
    assert summary["researcher_notes"] == 2


def test_job_elan_has_a_notes_tier(client, run):
    video, _, job = run
    _note(client, video, 1.5, "infant looks away")
    response = client.get(f"/api/v1/jobs/{job.job_id}/export/elan")
    assert response.status_code == 200, response.text
    assert 'TIER_ID="Notes (researcher)"' in response.text
    assert 'ANNOTATOR="alice"' in response.text
    assert "infant looks away" in response.text


def test_batch_elan_has_a_notes_tier(client, run):
    video, batch_id, _ = run
    _note(client, video, 2.0, "looks back")
    response = client.get(f"/api/v1/batches/{batch_id}/export/elan")
    assert response.status_code == 200, response.text
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        [name] = [n for n in archive.namelist() if n.endswith(".eaf")]
        assert "Notes (researcher)" in archive.read(name).decode()


def test_no_notes_no_tier_and_no_rows(client, run):
    _, batch_id, job = run
    table = client.get(f"/api/v1/batches/{batch_id}/export/table")
    assert not [r for r in _rows(table) if r["pipeline"] == "notes"]
    assert _summary(table)["researcher_notes"] == 0
    eaf = client.get(f"/api/v1/jobs/{job.job_id}/export/elan")
    assert "Notes (researcher" not in eaf.text
