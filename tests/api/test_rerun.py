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


def test_a_jobs_video_streams_with_range_requests():
    job = _finished_job()
    full = client.get(f"/api/v1/jobs/{job.job_id}/video")
    assert full.status_code == 200
    assert full.content == b"video bytes"
    part = client.get(
        f"/api/v1/jobs/{job.job_id}/video", headers={"Range": "bytes=0-4"}
    )
    assert part.status_code == 206
    assert part.content == b"video"
    Path(job.video_path).unlink()
    gone = client.get(f"/api/v1/jobs/{job.job_id}/video")
    assert gone.status_code == 404
    assert gone.json()["error"]["code"] == "VIDEO_NOT_STORED"


# --- Spec 022: missing videos, said before anything starts --------------------


@pytest.fixture
def moved_run(ingest_root):
    """A finished 3-video run read in place, then one video moved into site_b/."""
    from videoannotator.api.middleware.auth import validate_required_api_key

    site_a = ingest_root / "site_a"
    site_a.mkdir()
    for name in ("child01", "child02", "child03"):
        (site_a / f"{name}.mp4").write_bytes(b"video " + name.encode())
    app.dependency_overrides[validate_required_api_key] = lambda: {"is_admin": True}
    body = client.post(
        "/api/v1/ingest",
        json={"path": str(site_a), "selected_pipelines": ["scene_detection"]},
    ).json()
    storage = get_storage_backend()
    for job_id in body["created"]:
        job = storage.load_job_metadata(job_id)
        job.status = JobStatus.COMPLETED
        storage.save_job_metadata(job)
    site_b = ingest_root / "site_b"
    site_b.mkdir()
    (site_a / "child02.mp4").rename(site_b / "child02.mp4")
    from videoannotator.api.middleware.auth import validate_api_key

    app.dependency_overrides[validate_api_key] = lambda: {"is_admin": True}
    yield body, site_a, site_b
    app.dependency_overrides.clear()


def _job_for(body, name):
    storage = get_storage_backend()
    return next(
        job_id
        for job_id in body["created"]
        if storage.load_job_metadata(job_id).video_path.name == name
    )


def test_check_lists_missing_videos_and_creates_nothing(moved_run):
    body, site_a, _ = moved_run
    before = len(get_storage_backend().list_jobs())
    response = client.post(f"/api/v1/batches/{body['batch_id']}/rerun?check=true")
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["created"] == []
    assert len(get_storage_backend().list_jobs()) == before
    [skipped] = result["skipped"]
    assert skipped["job_id"] == _job_for(body, "child02.mp4")
    assert str(site_a / "child02.mp4") in skipped["reason"]


def test_without_check_the_rest_still_run(moved_run):
    body, _, _ = moved_run
    result = client.post(f"/api/v1/batches/{body['batch_id']}/rerun").json()
    assert len(result["created"]) == 2
    assert len(result["skipped"]) == 1


def test_relocate_finds_a_moved_video_by_name_and_size(moved_run):
    body, site_a, site_b = moved_run
    moved = _job_for(body, "child02.mp4")
    url = f"/api/v1/batches/{body['batch_id']}/rerun"
    params = {"relocate_folder": str(site_b)}

    checked = client.post(url, params={**params, "check": "true"}).json()
    assert checked["created"] == []
    assert checked["skipped"] == []
    assert checked["relocated"] == [
        {
            "job_id": moved,
            "from": str(site_a / "child02.mp4"),
            "to": str(site_b / "child02.mp4"),
        }
    ]

    result = client.post(url, params=params).json()
    assert len(result["created"]) == 3
    storage = get_storage_backend()
    paths = {storage.load_job_metadata(j).video_path for j in result["created"]}
    assert site_b / "child02.mp4" in paths


def test_relocate_ignores_a_same_named_file_of_another_size(moved_run):
    body, _, site_b = moved_run
    (site_b / "child02.mp4").write_bytes(b"a different take, longer")
    result = client.post(
        f"/api/v1/batches/{body['batch_id']}/rerun",
        params={"relocate_folder": str(site_b), "check": "true"},
    ).json()
    assert result["relocated"] == []
    assert len(result["skipped"]) == 1


def test_relocate_folder_must_be_allowed(moved_run, tmp_path):
    body, _, _ = moved_run
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    response = client.post(
        f"/api/v1/batches/{body['batch_id']}/rerun",
        params={"relocate_folder": str(outside), "check": "true"},
    )
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "INGEST_PATH_NOT_ALLOWED"


def test_a_moved_videos_results_stay_viewable(moved_run):
    body, _, _ = moved_run
    job_id = _job_for(body, "child02.mp4")
    job = client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["video_available"] is False
    assert client.get(f"/api/v1/jobs/{job_id}/results").status_code == 200
    other = client.get(f"/api/v1/jobs/{_job_for(body, 'child01.mp4')}").json()
    assert other["video_available"] is True
