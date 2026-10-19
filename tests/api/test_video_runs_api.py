"""GET /videos/runs: what the Video page needs to draw its lanes (spec 028 R2)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.synthetic_video import synthetic_video_bytes_avi
from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.batch.types import BatchJob, JobStatus, PipelineResult

client = TestClient(app)


@pytest.fixture(autouse=True)
def storage(tmp_path, monkeypatch):
    import videoannotator.database.database as db_module

    reset_storage_backend()
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    db_module.Base.metadata.create_all(bind=db_module.engine)
    yield tmp_path / "storage"
    reset_storage_backend()


def _runs(path: str):
    return client.get("/api/v1/videos/runs", params={"path": path})


def _job_on(video: Path, out: Path, **results) -> BatchJob:
    out.mkdir(parents=True, exist_ok=True)
    job = BatchJob(
        video_path=video,
        output_dir=out,
        status=JobStatus.COMPLETED,
        pipeline_results=results,
    )
    get_storage_backend().save_job_metadata(job)
    return job


def test_lists_the_runs_on_a_video(tmp_path):
    video = tmp_path / "elsewhere" / "p01_t1.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    job = _job_on(
        video,
        tmp_path / "out",
        scene_detection=PipelineResult("scene_detection", JobStatus.COMPLETED),
        speech_recognition=PipelineResult(
            "speech_recognition", JobStatus.FAILED, error_message="No audio"
        ),
    )
    (tmp_path / "out" / "p01_t1_scene_detection.json").write_text("{}")

    r = _runs(str(video))

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["path"] == str(video.resolve())
    assert body["name"] == "p01_t1.mp4"
    assert body["available"] is True
    assert body["unavailable_reason"] is None
    [run] = body["runs"]
    assert run["job_id"] == job.job_id
    assert run["status"] == "completed"
    pipelines = {p["name"]: p for p in run["pipelines"]}
    assert pipelines["speech_recognition"]["error"] == "No audio"
    [scene_file] = pipelines["scene_detection"]["files"]
    assert scene_file["kind"] == "result"
    downloaded = client.get(scene_file["download_url"])
    assert downloaded.status_code == 200
    assert body["projects"] == []


def test_a_readable_video_with_no_runs(storage):
    upload = client.post(
        "/api/v1/videos",
        files={"video": ("fresh.avi", synthetic_video_bytes_avi(), "video/x-msvideo")},
    ).json()

    r = _runs(upload["path"])

    assert r.status_code == 200, r.text
    assert r.json()["runs"] == []
    assert r.json()["duration_sec"] == pytest.approx(2.0, abs=0.2)


def test_a_moved_video_still_lists_its_runs(tmp_path):
    video = tmp_path / "gone" / "p01_t1.mp4"
    _job_on(video, tmp_path / "out")

    r = _runs(str(video))

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["available"] is False
    assert body["unavailable_reason"]
    assert body["duration_sec"] is None
    assert len(body["runs"]) == 1


def test_a_path_no_job_read_outside_the_shared_folders(tmp_path):
    video = tmp_path / "private.mp4"
    video.write_bytes(b"video")

    r = _runs(str(video))

    assert r.status_code == 403
    assert r.json()["error"]["code"] == "PATH_NOT_ALLOWED"


def test_an_empty_path():
    r = _runs("  ")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "VIDEO_PATH_REQUIRED"


def test_needs_the_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTH_REQUIRED", "true")
    assert _runs(str(tmp_path / "x.mp4")).status_code == 401
