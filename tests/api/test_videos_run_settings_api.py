"""GET /videos/run-settings, and every route that starts a run keeping the
project's and the caller's settings (spec 029 R5, R6)."""

from __future__ import annotations

import io

from tests.fixtures.synthetic_video import synthetic_video_bytes_avi
from tests.fixtures.video_index import write_video
from videoannotator.api.database import get_storage_backend
from videoannotator.batch.types import JobStatus


def _settings(api, **params):
    r = api.client.get("/api/v1/videos/run-settings", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def _upload(api, **data):
    r = api.client.post(
        "/api/v1/jobs/",
        files={
            "video": ("clip.avi", io.BytesIO(synthetic_video_bytes_avi()), "video/avi")
        },
        data={"selected_pipelines": "scene_detection"} | data,
    )
    assert r.status_code == 201, r.text
    return r.json()


def _finish(job_id):
    storage = get_storage_backend()
    job = storage.load_job_metadata(job_id)
    job.status = JobStatus.COMPLETED
    storage.save_job_metadata(job)


def test_defaults_on_a_fresh_server(api):
    assert _settings(api) == {
        "selected_pipelines": [],
        "config": {},
        "source": {
            "kind": "defaults",
            "id": None,
            "name": None,
            "changed_at": None,
            "from": None,
        },
    }


def test_an_upload_sets_the_last_run_and_its_project(api):
    project = api.project("Uploads")

    _upload(api, batch_id="b1", batch_name="Home videos", dataset_id=project["id"])

    assert _settings(api)["source"]["kind"] == "last_used"
    assert _settings(api)["source"]["name"] == "Home videos"
    found = _settings(api, project=project["id"])
    assert found["selected_pipelines"] == ["scene_detection"]
    assert found["source"]["kind"] == "project"


def test_running_a_job_again_with_changes(api):
    job = _upload(api)
    _finish(job["id"])

    r = api.client.post(
        f"/api/v1/jobs/{job['id']}/rerun",
        json={"config": {"scene_detection": {"threshold": 0.5}}},
    )

    assert r.status_code == 201, r.text
    assert _settings(api)["config"] == {"scene_detection": {"threshold": 0.5}}


def test_running_a_batch_again(api):
    job = _upload(api, batch_id="b2", batch_name="First")
    _finish(job["id"])

    r = api.client.post(
        "/api/v1/batches/b2/rerun",
        json={"config": {"scene_detection": {"threshold": 0.6}}},
    )

    assert r.status_code == 201, r.text
    found = _settings(api)
    assert found["config"] == {"scene_detection": {"threshold": 0.6}}
    assert found["source"]["name"] == "First (rerun)"


def test_running_a_dataset_from_its_stored_copies(api):
    job = _upload(api)
    project = api.project(
        "Stored",
        video_manifest=[
            {"filename": "clip.avi", "size_bytes": len(synthetic_video_bytes_avi())}
        ],
    )
    _finish(job["id"])

    r = api.client.post(
        f"/api/v1/datasets/{project['id']}/run",
        json={
            "selected_pipelines": ["scene_detection"],
            "config": {"scene_detection": {"threshold": 0.7}},
        },
    )

    assert r.status_code == 201, r.text
    found = _settings(api, project=project["id"])
    assert found["config"] == {"scene_detection": {"threshold": 0.7}}
    assert found["source"]["from"]["kind"] == "run"


def test_editing_a_projects_settings_without_running(api):
    project = api.project("Edited")

    r = api.client.put(
        f"/api/v1/datasets/{project['id']}",
        json={
            "current_settings": {
                "selected_pipelines": ["face_analysis"],
                "config": {"face_analysis": {"min_face_size": 50}},
            }
        },
    )

    assert r.status_code == 200, r.text
    stored = r.json()["current_settings"]
    assert stored["source"] == {"kind": "edit"}
    assert stored["updated_by"] == api.user["id"]
    found = _settings(api, project=project["id"])
    assert found["config"] == {"face_analysis": {"min_face_size": 50}}
    assert found["source"]["from"] == {"kind": "edit"}


def test_a_folder_run(api):
    video = write_video(api.shared / "pk" / "p01.mp4")
    r = api.client.post(
        "/api/v1/videos/run",
        json={"paths": [str(video)], "selected_pipelines": ["scene_detection"]},
    )
    assert r.status_code == 201, r.text

    assert _settings(api)["source"]["kind"] == "last_used"


def test_a_folder_ingest(api, monkeypatch):
    from videoannotator.api.main import app
    from videoannotator.api.middleware.auth import require_admin
    from videoannotator.api.v1 import ingest

    write_video(api.shared / "pk" / "p01.mp4")
    project = api.project("Ingested", server_folder=str(api.shared / "pk"))
    app.dependency_overrides[require_admin] = lambda: api.user | {"is_admin": True}
    monkeypatch.setattr(ingest, "require_local_caller", lambda request: None)

    r = api.client.post(
        "/api/v1/ingest",
        json={
            "path": str(api.shared / "pk"),
            "selected_pipelines": ["scene_detection"],
            "config": {"scene_detection": {"threshold": 0.3}},
            "dataset_id": project["id"],
        },
    )

    assert r.status_code == 201, r.text
    found = _settings(api, project=project["id"])
    assert found["config"] == {"scene_detection": {"threshold": 0.3}}
