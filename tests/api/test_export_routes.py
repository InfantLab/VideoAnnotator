"""Results out through the API: the same exports as the CLI (spec 025)."""

from __future__ import annotations

import csv
import io
import json
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.unit.exporters.conftest import FIXTURES, RUN_NAME, copy_outputs, make_run
from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.middleware.auth import validate_api_key
from videoannotator.batch.types import BatchJob, JobStatus
from videoannotator.exporters.results_set import from_folder
from videoannotator.exporters.table import parquet_available, write_csv

OWNER = {"id": "owner-1", "username": "alice", "is_admin": False}


@pytest.fixture
def client():
    import videoannotator.database.database as db_module

    reset_storage_backend()
    db_module.Base.metadata.create_all(bind=db_module.engine)
    app.dependency_overrides[validate_api_key] = lambda: OWNER
    yield TestClient(app)
    app.dependency_overrides.clear()
    reset_storage_backend()


@pytest.fixture
def run_jobs(tmp_path):
    """A finished two-video run in a spec 022 results folder."""
    run = make_run(tmp_path, {"demo_clip": FIXTURES, "second": FIXTURES})
    # The second video's files are named after it, as a real run's would be.
    for path in (run / "second").iterdir():
        path.rename(path.with_name(path.name.replace("demo_clip", "second")))
    batch_id = str(uuid.uuid4())
    storage = get_storage_backend()
    jobs = []
    for video in ("demo_clip", "second"):
        job = BatchJob(
            video_path=Path(f"/data/videos/{video}.mp4"),
            output_dir=run / video,
            status=JobStatus.COMPLETED,
            batch_id=batch_id,
            batch_name="Demo run",
        )
        storage.save_job_metadata(job)
        jobs.append(job)
    return run, batch_id, jobs


def _summary(response) -> dict:
    return json.loads(response.headers["x-videoannotator-export-summary"])


def test_job_table_is_the_cli_table_for_its_folder(client, run_jobs):
    run, _, jobs = run_jobs
    response = client.get(f"/api/v1/jobs/{jobs[0].job_id}/export/table")
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="demo_clip.csv"' in response.headers["content-disposition"]
    out = io.StringIO(newline="")
    write_csv(from_folder(run / "demo_clip"), out)
    assert response.text == out.getvalue()
    assert _summary(response)["videos"] == 1


def test_batch_table_covers_every_video(client, run_jobs):
    run, batch_id, _ = run_jobs
    response = client.get(f"/api/v1/batches/{batch_id}/export/table")
    assert response.status_code == 200, response.text
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert {r["video"] for r in rows} == {"demo_clip", "second"}
    assert {r["run"] for r in rows} == {RUN_NAME}
    assert RUN_NAME in response.headers["content-disposition"]
    out = io.StringIO(newline="")
    write_csv(from_folder(run), out)
    assert response.text == out.getvalue()  # FR-010


def test_dataset_table_includes_every_run(client, tmp_path):
    created = client.post(
        "/api/v1/datasets/",
        json={
            "name": f"Pilot {uuid.uuid4().hex[:6]}",
            "video_manifest": [{"filename": "demo_clip.mp4", "size_bytes": 1}],
        },
    )
    assert created.status_code == 201, created.text
    dataset_id = created.json()["id"]
    storage = get_storage_backend()
    for name in ("First (2026-10-01)", "Second (2026-10-02)"):
        run = make_run(tmp_path, {"demo_clip": FIXTURES}, name=name)
        storage.save_job_metadata(
            BatchJob(
                video_path=Path("/data/videos/demo_clip.mp4"),
                output_dir=run / "demo_clip",
                status=JobStatus.COMPLETED,
                dataset_id=dataset_id,
            )
        )
    response = client.get(f"/api/v1/datasets/{dataset_id}/export/table")
    assert response.status_code == 200, response.text
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert {r["run"] for r in rows} == {"First (2026-10-01)", "Second (2026-10-02)"}


def test_dataset_never_run(client):
    created = client.post(
        "/api/v1/datasets/",
        json={"name": f"Empty {uuid.uuid4().hex[:6]}", "video_manifest": []},
    )
    response = client.get(f"/api/v1/datasets/{created.json()['id']}/export/table")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "NOTHING_TO_EXPORT"


def test_failed_job_has_nothing_to_export(client, tmp_path):
    folder = tmp_path / "job"
    folder.mkdir()
    job = BatchJob(
        video_path=folder / "clip.mp4",
        storage_path=folder,
        status=JobStatus.FAILED,
        error_message="No audio could be extracted",
    )
    get_storage_backend().save_job_metadata(job)
    response = client.get(f"/api/v1/jobs/{job.job_id}/export/table")
    assert response.status_code == 409
    assert "No audio could be extracted" in response.json()["error"]["message"]


def test_failed_video_in_a_run_is_in_the_summary(client, run_jobs, tmp_path):
    _, batch_id, _ = run_jobs
    empty = tmp_path / "failed-job"
    empty.mkdir()
    get_storage_backend().save_job_metadata(
        BatchJob(
            video_path=empty / "third.mp4",
            storage_path=empty,
            status=JobStatus.FAILED,
            error_message="boom",
            batch_id=batch_id,
        )
    )
    response = client.get(f"/api/v1/batches/{batch_id}/export/table")
    assert response.status_code == 200
    assert _summary(response)["missing"] == [
        {"run": "", "video": "third", "reason": "failed: boom"}
    ]


def test_unknown_ids(client):
    assert client.get("/api/v1/jobs/nope/export/table").json()["error"]["code"] == (
        "JOB_NOT_FOUND"
    )
    missing = client.get(f"/api/v1/batches/{uuid.uuid4()}/export/table")
    assert missing.json()["error"]["code"] == "BATCH_NOT_FOUND"
    assert client.get("/api/v1/datasets/nope/export/table").status_code == 404


def test_unknown_format(client, run_jobs):
    _, _, jobs = run_jobs
    response = client.get(f"/api/v1/jobs/{jobs[0].job_id}/export/table?format=xlsx")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "INVALID_EXPORT_FORMAT"


@pytest.mark.skipif(not parquet_available(), reason="pyarrow not installed")
def test_parquet(client, run_jobs):
    import pyarrow.parquet as pq

    _, batch_id, _ = run_jobs
    response = client.get(f"/api/v1/batches/{batch_id}/export/table?format=parquet")
    assert response.status_code == 200, response.text
    table = pq.read_table(io.BytesIO(response.content))
    assert table.num_rows == _summary(response)["rows"]


def test_parquet_not_installed(client, run_jobs, monkeypatch):
    import videoannotator.api.v1.exports as exports

    monkeypatch.setattr(exports, "parquet_available", lambda: False)
    _, _, jobs = run_jobs
    response = client.get(f"/api/v1/jobs/{jobs[0].job_id}/export/table?format=parquet")
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "PARQUET_NOT_INSTALLED"
    assert error["hint"] == "pip install videoannotator[parquet]"
    assert error["details"] == {"extra": "parquet"}


def test_legacy_job_folder_exports(client, tmp_path):
    folder = copy_outputs(FIXTURES / "legacy", tmp_path / "old-job")
    job = BatchJob(
        video_path=folder / "demo_clip.mp4",
        storage_path=folder,
        status=JobStatus.COMPLETED,
    )
    get_storage_backend().save_job_metadata(job)
    response = client.get(f"/api/v1/jobs/{job.job_id}/export/table")
    assert response.status_code == 200
    rows = list(csv.DictReader(io.StringIO(response.text)))
    assert {r["videoannotator_version"] for r in rows} == {"not recorded"}


def test_job_methods(client, run_jobs):
    _, _, jobs = run_jobs
    response = client.get(f"/api/v1/jobs/{jobs[0].job_id}/methods")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["prose"].startswith("The video was annotated automatically")
    assert body["bibtex"].startswith("@software{addyman2026videoannotator,")
    assert body["summary"]["videos"] == 1


def test_batch_methods_describe_the_run_once(client, run_jobs):
    _, batch_id, _ = run_jobs
    body = client.get(f"/api/v1/batches/{batch_id}/methods").json()
    assert body["prose"].startswith("All 2 videos were annotated")


def test_methods_as_markdown(client, run_jobs):
    _, batch_id, _ = run_jobs
    response = client.get(f"/api/v1/batches/{batch_id}/methods?format=markdown")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/markdown")
    assert response.text.startswith("# Methods")


def test_methods_errors(client):
    assert client.get("/api/v1/jobs/nope/methods").status_code == 404


def test_job_elan(client, run_jobs):
    _, _, jobs = run_jobs
    response = client.get(f"/api/v1/jobs/{jobs[0].job_id}/export/elan")
    assert response.status_code == 200, response.text
    assert 'filename="demo_clip.eaf"' in response.headers["content-disposition"]
    assert b"<ANNOTATION_DOCUMENT" in response.content
    assert _summary(response)["left_out"]


def test_batch_elan_is_a_zip_of_one_file_per_video(client, run_jobs):
    import zipfile

    _, batch_id, _ = run_jobs
    response = client.get(f"/api/v1/batches/{batch_id}/export/elan")
    assert response.status_code == 200, response.text
    names = zipfile.ZipFile(io.BytesIO(response.content)).namelist()
    assert sorted(names) == [f"{RUN_NAME}/demo_clip.eaf", f"{RUN_NAME}/second.eaf"]
