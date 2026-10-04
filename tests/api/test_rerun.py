"""Run it again (spec 019): a new, linked job; the original untouched."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.batch.types import BatchJob, JobStatus

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_db():
    reset_storage_backend()
    yield
    reset_storage_backend()


def _finished_job(status=JobStatus.COMPLETED, batch_id=None, **form) -> BatchJob:
    data = {
        "selected_pipelines": "scene_detection",
        "config": '{"scene_detection": {"threshold": 20}}',
        **form,
    }
    if batch_id:
        data.update(batch_id=batch_id, batch_name="Session 1")
    response = client.post(
        "/api/v1/jobs/",
        files={"video": ("clip.mp4", io.BytesIO(b"video bytes"), "video/mp4")},
        data=data,
    )
    assert response.status_code == 201, response.text
    storage = get_storage_backend()
    job = storage.load_job_metadata(response.json()["id"])
    job.status = status
    storage.save_job_metadata(job)
    return job


def _snapshot(job_id: str) -> tuple[dict, dict[str, bytes]]:
    job = get_storage_backend().load_job_metadata(job_id)
    files = {p.name: p.read_bytes() for p in Path(job.storage_path).iterdir()}
    return job.to_dict(), files


def test_rerun_is_a_new_linked_job_and_the_original_is_untouched():
    original = _finished_job()
    before = _snapshot(original.job_id)

    response = client.post(f"/api/v1/jobs/{original.job_id}/rerun")
    assert response.status_code == 201, response.text
    rerun = response.json()
    assert rerun["id"] != original.job_id
    assert rerun["rerun_of"] == original.job_id
    assert rerun["status"] == "pending"
    assert rerun["selected_pipelines"] == ["scene_detection"]
    assert rerun["config"] == {"scene_detection": {"threshold": 20}}

    assert _snapshot(original.job_id) == before
    assert client.get(f"/api/v1/jobs/{original.job_id}").json()["reruns"] == [
        rerun["id"]
    ]


def test_edited_settings_replace_the_originals():
    original = _finished_job()
    rerun = client.post(
        f"/api/v1/jobs/{original.job_id}/rerun",
        json={"config": {"scene_detection": {"threshold": 35}}},
    ).json()
    assert rerun["config"] == {"scene_detection": {"threshold": 35}}
    assert rerun["selected_pipelines"] == ["scene_detection"]


def test_uploaded_video_survives_deleting_the_original():
    original = _finished_job()
    rerun = client.post(f"/api/v1/jobs/{original.job_id}/rerun").json()
    assert Path(rerun["video_path"]).parent == Path(rerun["storage_path"])
    assert client.delete(f"/api/v1/jobs/{original.job_id}").status_code in (200, 204)
    assert Path(rerun["video_path"]).read_bytes() == b"video bytes"


def test_a_server_folder_video_is_used_in_place(tmp_path):
    video = tmp_path / "corpus" / "dyad.mp4"
    video.parent.mkdir()
    video.write_bytes(b"x")
    job = BatchJob(
        video_path=video,
        status=JobStatus.FAILED,
        selected_pipelines=["scene_detection"],
    )
    get_storage_backend().save_job_metadata(job)
    rerun = client.post(f"/api/v1/jobs/{job.job_id}/rerun").json()
    assert Path(rerun["video_path"]) == video


def test_a_running_job_or_a_missing_video_cannot_be_rerun(tmp_path):
    running = _finished_job(status=JobStatus.RUNNING)
    response = client.post(f"/api/v1/jobs/{running.job_id}/rerun")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "JOB_NOT_FINISHED"

    gone = BatchJob(video_path=tmp_path / "gone.mp4", status=JobStatus.COMPLETED)
    get_storage_backend().save_job_metadata(gone)
    response = client.post(f"/api/v1/jobs/{gone.job_id}/rerun")
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "RERUN_VIDEO_MISSING"


def test_unknown_pipeline_in_edited_settings_is_rejected():
    original = _finished_job()
    response = client.post(
        f"/api/v1/jobs/{original.job_id}/rerun", json={"selected_pipelines": ["nope"]}
    )
    assert response.status_code >= 400
    assert get_storage_backend().list_reruns(original.job_id) == []


def test_batch_rerun_makes_a_new_batch_and_reports_skips():
    a = _finished_job(batch_id="b-1")
    b = _finished_job(status=JobStatus.RUNNING, batch_id="b-1")
    response = client.post("/api/v1/batches/b-1/rerun")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["rerun_of_batch"] == "b-1" and body["batch_id"] != "b-1"
    assert len(body["created"]) == 1
    assert [s["job_id"] for s in body["skipped"]] == [b.job_id]
    new = get_storage_backend().load_job_metadata(body["created"][0])
    assert (new.rerun_of, new.batch_id, new.batch_name) == (
        a.job_id,
        body["batch_id"],
        "Session 1 (rerun)",
    )
