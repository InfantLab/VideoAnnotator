"""A run from Ask AI: answers as they come, Stop keeps them (spec 027 US5)."""

import csv
import io
import json
import threading

from fastapi.testclient import TestClient

from tests.fixtures.fake_vlm import FakeTryVLM, FakeVLM
from tests.fixtures.synthetic_video import SyntheticVideoSpec, write_synthetic_video_avi
from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.v1 import ask
from videoannotator.batch.job_execution import run_job_pipelines
from videoannotator.batch.result_files import job_folder
from videoannotator.batch.types import JobStatus
from videoannotator.database import database as db_module
from videoannotator.pipelines.vlm_annotation import vlm_pipeline

client = TestClient(app)
LOCAL = "http://127.0.0.1:11434"


def test_run_answers_as_they_come_then_stop_keeps_them(ingest_root, monkeypatch):
    reset_storage_backend()
    db_module.Base.metadata.create_all(bind=db_module.engine)
    FakeVLM.reset()
    FakeVLM.hold_run_call = 4
    monkeypatch.setattr(vlm_pipeline, "OllamaVLMClient", FakeVLM)
    monkeypatch.setattr(ask, "OllamaVLMClient", FakeTryVLM)
    video = ingest_root / "p01_t1.avi"
    write_synthetic_video_avi(video, SyntheticVideoSpec(fps=10, num_frames=100))

    r = client.post(
        "/api/v1/ask/runs",
        json={
            "video_path": str(video),
            "prompt": "Touch?",
            "model": "qwen3.5:9b",
            "base_url": LOCAL,
            "sampling": {"interval": 1},
        },
    )
    assert r.status_code == 201, r.text
    entry = r.json()
    job_id = entry["job_id"]
    storage = get_storage_backend()
    job = storage.load_job_metadata(job_id)
    runner = threading.Thread(
        target=run_job_pipelines,
        args=(job, storage, {"vlm_annotation": vlm_pipeline.VLMAnnotationPipeline}),
    )
    runner.start()
    try:
        assert FakeVLM.held.wait(10)
        so_far = client.get(f"/api/v1/ask/runs/{job_id}/answers").json()
        assert so_far["status"] == "running"
        assert so_far["planned"] == 11
        assert len(so_far["answers"]) >= 1

        assert client.post(f"/api/v1/jobs/{job_id}/cancel").status_code == 200
    finally:
        FakeVLM.release.set()
    runner.join(20)
    assert not runner.is_alive()

    job = storage.load_job_metadata(job_id)
    assert job.status == JobStatus.CANCELLED
    result = job.pipeline_results["vlm_annotation"]
    assert result.status == JobStatus.COMPLETED
    assert result.provenance["vlm"]["stopped_early"] is True
    assert result.provenance["vlm"]["samples_answered"] == 4
    assert result.provenance["vlm"]["samples_planned"] == 11

    data = json.loads((job_folder(job) / "p01_t1_vlm_annotation.json").read_text())
    assert len(data["annotations"]) == 4
    assert data["vlm_progress"]["stopped_early"] is True
    assert data["provenance"]["vlm"]["stopped_early"] is True

    final = client.get(f"/api/v1/ask/runs/{job_id}/answers").json()
    assert final["status"] == "stopped"
    assert final["stopped_early"] is True
    assert len(final["answers"]) == 4

    shown = client.get(f"/api/v1/ask/entries/{entry['id']}").json()
    assert shown["status"] == "stopped"

    table = client.get(f"/api/v1/jobs/{job_id}/export/table")
    assert table.status_code == 200, table.text
    rows = list(csv.DictReader(io.StringIO(table.text)))
    assert len([r for r in rows if r.get("measure") == "vlm_label"]) == 4
