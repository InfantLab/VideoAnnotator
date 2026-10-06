"""A run's results as one download, and deleting a whole run (spec 022, US7)."""

import io
import uuid
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.middleware.auth import validate_required_api_key
from videoannotator.batch.types import BatchJob, JobStatus

client = TestClient(app)


@pytest.fixture(autouse=True)
def as_admin():
    reset_storage_backend()
    app.dependency_overrides[validate_required_api_key] = lambda: {"is_admin": True}
    yield
    app.dependency_overrides.clear()
    reset_storage_backend()


@pytest.fixture
def run(ingest_root):
    """A finished 2-video run read in place, with results on disk."""
    for name in ("child01", "child02"):
        (ingest_root / f"{name}.mp4").write_bytes(b"video")
    body = client.post(
        "/api/v1/ingest",
        json={
            "path": str(ingest_root),
            "selected_pipelines": ["scene_detection"],
            "batch_name": "Wave 2",
        },
    ).json()
    storage = get_storage_backend()
    for job_id in body["created"]:
        job = storage.load_job_metadata(job_id)
        stem = job.video_path.stem
        (job.output_dir / f"{stem}_scene_detection.json").write_text("{}")
        (job.output_dir / "stray_copy.mp4").write_bytes(b"never zipped")
        job.status = JobStatus.COMPLETED
        storage.save_job_metadata(job)
    return body


def _zip(response) -> zipfile.ZipFile:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/zip"
    return zipfile.ZipFile(io.BytesIO(response.content))


class TestRunZip:
    def test_same_layout_as_on_disk_without_videos(self, run):
        top = Path(run["results_folder"]["path"]).name
        names = _zip(
            client.get(f"/api/v1/batches/{run['batch_id']}/results.zip")
        ).namelist()
        assert sorted(names) == [
            f"{top}/child01/child01_scene_detection.json",
            f"{top}/child02/child02_scene_detection.json",
            f"{top}/run.json",
        ]

    def test_unknown_batch(self):
        response = client.get(f"/api/v1/batches/{uuid.uuid4()}/results.zip")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "BATCH_NOT_FOUND"

    def test_runs_from_before_results_folders(self, tmp_path):
        storage = get_storage_backend()
        batch_id = str(uuid.uuid4())
        for name in ("a", "a"):
            folder = tmp_path / str(uuid.uuid4())
            folder.mkdir()
            (folder / f"{name}.mp4").write_bytes(b"video")
            (folder / f"{name}_scene_detection.json").write_text("{}")
            storage.save_job_metadata(
                BatchJob(
                    video_path=folder / f"{name}.mp4",
                    storage_path=folder,
                    status=JobStatus.COMPLETED,
                    batch_id=batch_id,
                    batch_name="Old run",
                )
            )
        names = _zip(client.get(f"/api/v1/batches/{batch_id}/results.zip")).namelist()
        assert sorted(names) == [
            "Old run/a 2/a_scene_detection.json",
            "Old run/a/a_scene_detection.json",
        ]


class TestJobZip:
    def test_leaves_the_video_out_by_default(self, run):
        job_id = run["created"][0]
        response = client.get(f"/api/v1/jobs/{job_id}/artifacts")
        names = _zip(response).namelist()
        assert names == ["child01_scene_detection.json"]
        assert "include_video=true" in response.headers["x-videoannotator-notice"]

    def test_includes_the_video_on_request(self, run, ingest_root):
        job_id = run["created"][0]
        response = client.get(f"/api/v1/jobs/{job_id}/artifacts?include_video=true")
        assert sorted(_zip(response).namelist()) == [
            "child01.mp4",
            "child01_scene_detection.json",
        ]
        assert "x-videoannotator-notice" not in response.headers


class TestDeleteRun:
    def test_deletes_jobs_and_results_never_videos(self, run, ingest_root):
        run_folder = Path(run["results_folder"]["path"])
        assert client.delete(f"/api/v1/batches/{run['batch_id']}").status_code == 204
        storage = get_storage_backend()
        assert storage.list_jobs_by_batch(run["batch_id"]) == []
        assert not run_folder.exists()
        assert sorted(p.name for p in ingest_root.iterdir()) == [
            "child01.mp4",
            "child02.mp4",
        ]

    def test_a_running_job_is_cancelled_first(self, run, monkeypatch):
        from videoannotator.api.v1 import batches

        storage = get_storage_backend()
        job = storage.load_job_metadata(run["created"][0])
        job.status = JobStatus.RUNNING
        storage.save_job_metadata(job)
        cancelled = []
        real = batches.cancel_job_by_id
        monkeypatch.setattr(
            batches,
            "cancel_job_by_id",
            lambda job_id, s: cancelled.append(job_id) or real(job_id, s),
        )
        assert client.delete(f"/api/v1/batches/{run['batch_id']}").status_code == 204
        assert run["created"][0] in cancelled

    def test_unknown_batch(self):
        response = client.delete(f"/api/v1/batches/{uuid.uuid4()}")
        assert response.status_code == 404
