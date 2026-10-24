"""GET /api/v1/jobs/{id}/export/wide (spec 033 R10)."""

import json

from fastapi.testclient import TestClient

from videoannotator.api.database import get_storage_backend
from videoannotator.api.main import app
from videoannotator.batch.result_files import job_folder
from videoannotator.batch.types import BatchJob, JobStatus

client = TestClient(app)


def _job(tmp_path, data):
    tmp_path.mkdir(parents=True, exist_ok=True)
    video = tmp_path / "p01.mp4"
    video.write_bytes(b"")
    job = BatchJob(video_path=video, config={}, status=JobStatus.COMPLETED)
    job.storage_path = tmp_path / "job"
    storage = get_storage_backend()
    storage.save_job_metadata(job)
    folder = job_folder(job)
    folder.mkdir(parents=True, exist_ok=True)
    if data is not None:
        (folder / "p01_vlm_annotation.json").write_text(json.dumps(data))
    return job


def test_a_shaped_runs_wide_file(tmp_path):
    shape = [{"name": "touch", "kind": "yes_no"}]
    job = _job(
        tmp_path,
        {
            "vlm_answer_shape": shape,
            "annotations": [
                {
                    "timestamp_sec": 0.0,
                    "frame_number": 0,
                    "status": "answered",
                    "fields": {"touch": "no"},
                    "fit": True,
                    "model": "m",
                    "backend": "ollama",
                },
            ],
        },
    )
    r = client.get(f"/api/v1/jobs/{job.job_id}/export/wide")
    assert r.status_code == 200, r.text
    assert r.text.splitlines()[1].startswith("p01.mp4,0.0,0,no,yes,")


def test_unshaped_and_missing(tmp_path):
    job = _job(tmp_path, {"annotations": []})
    r = client.get(f"/api/v1/jobs/{job.job_id}/export/wide")
    assert (r.status_code, r.json()["error"]["code"]) == (422, "NOT_SHAPED")
    empty = _job(tmp_path / "b", None)
    assert client.get(f"/api/v1/jobs/{empty.job_id}/export/wide").status_code == 404
