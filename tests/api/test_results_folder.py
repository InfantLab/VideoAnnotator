"""Every run's results in one visible folder, by run then video (spec 022, US6/US8)."""

import io
import json
import time
import uuid
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.middleware.auth import (
    validate_api_key,
    validate_required_api_key,
)
from videoannotator.batch.types import JobStatus

ADMIN = {"id": "admin-1", "username": "root", "is_admin": True}
TODAY = date.today().isoformat()

client = TestClient(app)


@pytest.fixture(autouse=True)
def as_admin():
    import videoannotator.database.database as db_module

    reset_storage_backend()
    db_module.Base.metadata.create_all(bind=db_module.engine)
    app.dependency_overrides[validate_required_api_key] = lambda: ADMIN
    app.dependency_overrides[validate_api_key] = lambda: ADMIN
    yield
    app.dependency_overrides.clear()
    reset_storage_backend()


@pytest.fixture
def study(ingest_root):
    folder = ingest_root / "BabyJokes"
    folder.mkdir()
    for name in ("child01", "child02", "child03"):
        (folder / f"{name}.mp4").write_bytes(b"video " + name.encode())
    return folder


def _ingest(path, **body):
    response = client.post(
        "/api/v1/ingest",
        json={"path": str(path), "selected_pipelines": ["scene_detection"], **body},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _job(job_id):
    return get_storage_backend().load_job_metadata(job_id)


def _upload(name="clip.mp4", **data):
    response = client.post(
        "/api/v1/jobs/",
        files={"video": (name, io.BytesIO(b"uploaded bytes"), "video/mp4")},
        data={"selected_pipelines": "scene_detection", **data},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _finish(job_id):
    storage = get_storage_backend()
    job = storage.load_job_metadata(job_id)
    job.status = JobStatus.COMPLETED
    storage.save_job_metadata(job)


class TestFolderIngest:
    def test_each_video_gets_a_folder_in_a_run_named_after_the_run(
        self, study, results_root
    ):
        body = _ingest(study, batch_name="BabyJokes wave 2")
        run = results_root.resolve() / f"BabyJokes wave 2 ({TODAY})"
        assert body["results_folder"] == {"path": str(run), "display_path": str(run)}
        for job_id in body["created"]:
            job = _job(job_id)
            assert job.output_dir == run / job.video_path.stem
            assert job.output_dir.is_dir()
        assert sorted(p.name for p in run.iterdir()) == [
            "child01",
            "child02",
            "child03",
            "run.json",
        ]

    def test_a_second_run_with_the_same_name_is_numbered(self, study, results_root):
        _ingest(study, batch_name="Same")
        second = _ingest(study, batch_name="Same")
        assert second["results_folder"]["path"].endswith(f"Same ({TODAY} 2)")

    def test_run_record_lists_every_video_and_its_source(self, study, results_root):
        body = _ingest(study, batch_name="R", config={"scene_detection": {"t": 1}})
        record = json.loads(
            (Path(body["results_folder"]["path"]) / "run.json").read_text()
        )
        assert record["format"] == "videoannotator-run"
        assert record["run"]["batch_id"] == body["batch_id"]
        assert record["pipelines"] == ["scene_detection"]
        assert record["config"] == {"scene_detection": {"t": 1}}
        sources = {v["source"]["path"] for v in record["videos"]}
        assert sources == {str(p) for p in study.glob("*.mp4")}
        assert all(v["source"]["kind"] == "in_place" for v in record["videos"])

    def test_no_video_is_copied(self, study, results_root):
        body = _ingest(study)
        assert not list(Path(body["results_folder"]["path"]).rglob("*.mp4"))
        for job_id in body["created"]:
            assert not list(Path(_job(job_id).storage_path).glob("*.mp4"))

    def test_unwritable_results_folder_refuses_the_run(
        self, study, tmp_path, monkeypatch
    ):
        blocker = tmp_path / "not-a-folder"
        blocker.write_text("x")
        monkeypatch.setenv("VIDEOANNOTATOR_RESULTS_DIR", str(blocker / "results"))
        before = len(get_storage_backend().list_jobs())
        response = client.post(
            "/api/v1/ingest",
            json={"path": str(study), "selected_pipelines": ["scene_detection"]},
        )
        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "RESULTS_DIR_UNWRITABLE"
        assert str(blocker / "results") in error["message"]
        assert len(get_storage_backend().list_jobs()) == before

    def test_the_results_folder_is_never_scanned_for_videos(
        self, ingest_root, monkeypatch
    ):
        monkeypatch.setenv("VIDEOANNOTATOR_RESULTS_DIR", str(ingest_root / "results"))
        (ingest_root / "results").mkdir()
        (ingest_root / "results" / "stray.mp4").write_bytes(b"x")
        (ingest_root / "real.mp4").write_bytes(b"x")
        body = _ingest(ingest_root, recursive=True)
        assert body["total"] == 1

    def test_a_hundred_videos_in_under_five_seconds(self, ingest_root, results_root):
        for i in range(100):
            (ingest_root / f"v{i:03d}.mp4").write_bytes(b"v")
        start = time.monotonic()
        body = _ingest(ingest_root, files=[f"v{i:03d}.mp4" for i in range(100)])
        assert time.monotonic() - start < 5
        run = Path(body["results_folder"]["path"])
        assert len([p for p in run.iterdir() if p.is_dir()]) == 100


class TestUpload:
    def test_results_go_to_the_results_folder_and_the_copy_stays_internal(
        self, results_root
    ):
        body = _upload("session.mp4")
        job = _job(body["id"])
        assert body["results_folder"]["path"] == str(job.output_dir)
        assert job.output_dir.parent.name == f"session ({TODAY})"
        assert job.video_path.parent == job.storage_path
        assert not list(results_root.rglob("*.mp4"))
        record = json.loads((job.output_dir.parent / "run.json").read_text())
        assert record["videos"][0]["source"] == {
            "kind": "uploaded",
            "original_filename": "session.mp4",
            "size_bytes": len(b"uploaded bytes"),
        }

    def test_one_batch_is_one_run_folder(self, results_root):
        batch_id = str(uuid.uuid4())
        jobs = [
            _upload(f"v{i}.mp4", batch_id=batch_id, batch_name="Wave 3")
            for i in range(3)
        ]
        folders = {Path(j["results_folder"]["path"]).parent for j in jobs}
        assert len(folders) == 1
        assert folders.pop().name == f"Wave 3 ({TODAY})"


class TestRerunAndDataset:
    def test_rerun_gets_its_own_run_folder(self, results_root):
        original = _upload("clip.mp4", batch_id=str(uuid.uuid4()), batch_name="S1")
        _finish(original["id"])
        response = client.post(f"/api/v1/batches/{original['batch_id']}/rerun")
        assert response.status_code == 201, response.text
        rerun = _job(response.json()["created"][0])
        assert rerun.output_dir.parent.name == f"S1 (rerun) ({TODAY})"
        assert rerun.output_dir != _job(original["id"]).output_dir

    def test_single_job_rerun_is_named_as_a_rerun(self, results_root):
        original = _upload("clip.mp4")
        _finish(original["id"])
        response = client.post(f"/api/v1/jobs/{original['id']}/rerun")
        assert response.status_code == 201, response.text
        assert Path(response.json()["results_folder"]["path"]).parent.name == (
            f"clip (rerun) ({TODAY})"
        )

    def test_dataset_run_is_named_after_the_dataset(self, results_root):
        name = f"vid_{uuid.uuid4().hex[:6]}.mp4"
        _upload(name)
        dataset = client.post(
            "/api/v1/datasets/",
            json={
                "name": "Pilot",
                "video_manifest": [
                    {"filename": name, "size_bytes": len(b"uploaded bytes")}
                ],
            },
        ).json()
        response = client.post(
            f"/api/v1/datasets/{dataset['id']}/run",
            json={"selected_pipelines": ["scene_detection"]},
        )
        assert response.status_code == 201, response.text
        job = _job(response.json()["created"][0])
        assert job.output_dir.parent.name == f"Pilot ({TODAY})"


class TestResponsesAndDeletion:
    def test_job_and_batch_responses_carry_the_folder(self, study, results_root):
        body = _ingest(study, batch_name="Shown")
        job = client.get(f"/api/v1/jobs/{body['created'][0]}").json()
        assert job["results_folder"]["path"].startswith(body["results_folder"]["path"])
        assert job["video_available"] is True
        batch = client.get(f"/api/v1/batches/{body['batch_id']}").json()
        assert batch["results_folder"] == body["results_folder"]

    def test_deleting_jobs_removes_results_never_videos(self, study, results_root):
        body = _ingest(study, batch_name="Gone")
        run = Path(body["results_folder"]["path"])
        storage = get_storage_backend()
        first = body["created"][0]
        first_folder = _job(first).output_dir
        storage.delete_job(first)
        assert not first_folder.exists()
        assert run.is_dir()
        for job_id in body["created"][1:]:
            storage.delete_job(job_id)
        assert not run.exists()
        assert len(list(study.glob("*.mp4"))) == 3

    def test_retry_clears_its_own_partial_output(self, study, results_root):
        body = _ingest(study)
        storage = get_storage_backend()
        job = storage.load_job_metadata(body["created"][0])
        (job.output_dir / "partial.json").write_text("{}")
        job.status = JobStatus.FAILED
        storage.save_job_metadata(job)
        assert client.post(f"/api/v1/jobs/{job.job_id}/retry").status_code == 200
        assert list(job.output_dir.iterdir()) == []


class TestChangingTheResultsFolder:
    def test_new_runs_go_to_the_new_folder_and_old_ones_stay(
        self, study, tmp_path, monkeypatch
    ):
        first = _ingest(study, batch_name="Before")
        elsewhere = tmp_path / "elsewhere"
        monkeypatch.setenv("VIDEOANNOTATOR_RESULTS_DIR", str(elsewhere))
        second = _ingest(study, batch_name="After")
        assert Path(second["results_folder"]["path"]).parent == elsewhere.resolve()
        old_job = client.get(f"/api/v1/jobs/{first['created'][0]}").json()
        assert old_job["results_folder"]["path"].startswith(
            first["results_folder"]["path"]
        )
        assert Path(old_job["results_folder"]["path"]).is_dir()

    def test_dotenv_sets_the_results_folder(self, tmp_path, monkeypatch):
        from videoannotator.config_env import load_env_file, results_dir

        (tmp_path / ".env").write_text(
            f"VIDEOANNOTATOR_RESULTS_DIR={tmp_path / 'from-dotenv'}\n"
        )
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VIDEOANNOTATOR_RESULTS_DIR")
        load_env_file()
        assert results_dir() == (tmp_path / "from-dotenv").resolve()
