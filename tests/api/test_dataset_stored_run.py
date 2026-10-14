"""Running a saved dataset from the copies its uploaded jobs keep: no folder to
pick and nothing to upload again, while the server still has the videos."""

import io
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.middleware.auth import validate_api_key

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


def _dataset(client, *videos: tuple[str, bytes]) -> str:
    response = client.post(
        "/api/v1/datasets/",
        json={
            "name": f"3 JOKES {uuid.uuid4().hex[:6]}",
            "video_manifest": [
                {"filename": name, "size_bytes": len(data)} for name, data in videos
            ],
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


def _upload(client, name: str, data: bytes, dataset_id: str | None = None) -> str:
    response = client.post(
        "/api/v1/jobs/",
        files={"video": (name, io.BytesIO(data), "video/mp4")},
        data={"selected_pipelines": "scene_detection"}
        | ({"dataset_id": dataset_id} if dataset_id else {}),
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


@pytest.fixture
def videos():
    """Names unique to the test: the storage, and so its jobs, outlive one."""
    tag = uuid.uuid4().hex[:6]
    return (f"a_{tag}.mp4", b"video a"), (f"b_{tag}.mp4", b"video bb")


def test_stored_videos_found_by_name_and_size(client, videos):
    A, B = videos
    dataset_id = _dataset(client, A, B)
    job_a = _upload(client, *A, dataset_id=dataset_id)
    _upload(client, B[0], b"a different b")  # same name, other size

    body = client.get(f"/api/v1/datasets/{dataset_id}/stored-videos").json()

    assert [(v["filename"], v["job_id"]) for v in body["videos"]] == [
        (A[0], job_a),
        (B[0], None),
    ]
    assert (body["stored"], body["missing"]) == (1, 1)


def test_run_creates_a_batch_from_stored_copies(client, videos):
    A, B = videos
    dataset_id = _dataset(client, A, B)
    _upload(client, *A, dataset_id=dataset_id)
    _upload(client, *B, dataset_id=dataset_id)

    response = client.post(
        f"/api/v1/datasets/{dataset_id}/run",
        json={"selected_pipelines": ["scene_detection"]},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert len(body["created"]) == 2
    assert body["skipped"] == []
    assert body["batch_name"].startswith("3 JOKES")
    storage = get_storage_backend()
    for job_id in body["created"]:
        job = storage.load_job_metadata(job_id)
        assert job.batch_id == body["batch_id"]
        assert job.dataset_id == dataset_id
        assert job.rerun_of is None
        # Its own copy, so deleting the job it came from can't break it.
        assert Path(job.video_path).parent == Path(job.storage_path)
        assert Path(job.video_path).read_bytes() in (A[1], B[1])


def test_run_skips_videos_no_longer_stored(client, videos):
    A, B = videos
    dataset_id = _dataset(client, A, B)
    _upload(client, *A, dataset_id=dataset_id)

    body = client.post(
        f"/api/v1/datasets/{dataset_id}/run",
        json={"selected_pipelines": ["scene_detection"]},
    ).json()

    assert len(body["created"]) == 1
    assert body["skipped"] == [
        {"filename": B[0], "reason": "no longer stored on the server"}
    ]


def test_run_with_nothing_stored_says_so(client, videos):
    A, B = videos
    dataset_id = _dataset(client, A)

    response = client.post(
        f"/api/v1/datasets/{dataset_id}/run",
        json={"selected_pipelines": ["scene_detection"]},
    )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "DATASET_NOT_STORED"


def test_deleted_job_video_is_not_offered(client, videos):
    A, B = videos
    dataset_id = _dataset(client, A)
    job_id = _upload(client, *A, dataset_id=dataset_id)
    assert client.delete(f"/api/v1/jobs/{job_id}").status_code in (200, 204)

    body = client.get(f"/api/v1/datasets/{dataset_id}/stored-videos").json()

    assert body["videos"][0]["job_id"] is None


def test_a_video_whose_folder_isnt_shared_any_more_says_so(
    client, videos, ingest_root, tmp_path
):
    from videoannotator.batch.types import BatchJob, JobStatus

    A, B = videos
    dataset_id = _dataset(client, A, B)
    _upload(client, *A, dataset_id=dataset_id)
    unshared = tmp_path / "Old study" / B[0]
    get_storage_backend().save_job_metadata(
        BatchJob(video_path=unshared, status=JobStatus.COMPLETED)
    )

    body = client.post(
        f"/api/v1/datasets/{dataset_id}/run",
        json={"selected_pipelines": ["scene_detection"]},
    ).json()

    assert body["skipped"] == [
        {
            "filename": B[0],
            "reason": f"{unshared.parent} isn't shared with VideoAnnotator any more",
        }
    ]
