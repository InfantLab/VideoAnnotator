"""GET /jobs/{id}/results advertises download URLs that work.

Until v1.6.0 the download route served the storage backend's `output_file`,
which for the database backend is `database:/annotations/...`, so every
advertised `download_url` returned OUTPUT_FILE_MISSING. Files are now served
from the job folder, where pipelines write `<video stem>_<suffix>`.
"""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.batch.types import JobStatus, PipelineResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_db():
    reset_storage_backend()
    yield
    reset_storage_backend()


@pytest.fixture
def finished_job():
    response = client.post(
        "/api/v1/jobs/",
        files={"video": ("clip.mp4", io.BytesIO(b"fake video"), "video/mp4")},
        data={"selected_pipelines": "person_tracking,speech_recognition"},
    )
    assert response.status_code == 201, response.text
    job_id = response.json()["id"]

    storage = get_storage_backend()
    job = storage.load_job_metadata(job_id)
    folder = Path(job.storage_path)
    stem = Path(job.video_path).stem
    (folder / f"{stem}_person_tracking.json").write_text('{"coco": true}')
    (folder / f"{stem}_person_tracks.json").write_text('{"tracks": true}')
    job.status = JobStatus.COMPLETED
    job.pipeline_results = {
        "person_tracking": PipelineResult(
            pipeline_name="person_tracking",
            status=JobStatus.COMPLETED,
            output_file=Path(f"database:/annotations/{job_id}/person_tracking"),
        ),
        "speech_recognition": PipelineResult(
            pipeline_name="speech_recognition",
            status=JobStatus.FAILED,
            error_message="No audio could be extracted",
        ),
    }
    storage.save_job_metadata(job)
    return job_id, stem


def test_results_list_each_file_a_pipeline_wrote(finished_job):
    job_id, stem = finished_job
    results = client.get(f"/api/v1/jobs/{job_id}/results").json()["pipeline_results"]

    tracking = results["person_tracking"]
    base = f"/api/v1/jobs/{job_id}/results/files/person_tracking"
    assert tracking["download_url"] == base
    assert [f["name"] for f in tracking["files"]] == [
        f"{stem}_person_tracking.json",
        f"{stem}_person_tracks.json",
    ]
    assert results["speech_recognition"]["download_url"] is None
    assert results["speech_recognition"]["files"] == []


def test_every_advertised_url_downloads(finished_job):
    job_id, _ = finished_job
    tracking = client.get(f"/api/v1/jobs/{job_id}/results").json()["pipeline_results"][
        "person_tracking"
    ]

    main = client.get(tracking["download_url"])
    assert main.status_code == 200
    assert main.content == b'{"coco": true}'
    contents = [client.get(f["download_url"]).content for f in tracking["files"]]
    assert contents == [b'{"coco": true}', b'{"tracks": true}']


@pytest.mark.parametrize(
    "path",
    [
        "speech_recognition",
        "person_tracking?name=other.json",
        "person_tracking?name=../../etc/passwd",
    ],
)
def test_missing_file_is_a_404_with_a_hint(finished_job, path):
    job_id, _ = finished_job
    response = client.get(f"/api/v1/jobs/{job_id}/results/files/{path}")
    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "OUTPUT_FILE_MISSING"
    assert "/artifacts" in error["hint"]
