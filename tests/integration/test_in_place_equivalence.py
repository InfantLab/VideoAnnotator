"""A video read in place gives the same results as an uploaded copy (spec 022, FR-007).

The two routes differ only in where the job's video is: the researcher's own
folder, or a copy in internal storage. This runs the demo clip both ways
through the real scene_detection pipeline and compares the annotations.

Needs the scene extras and model weights, so it is marked real_models
(deselected in CI), like test_output_baseline.py.
"""

import io
import json
import shutil

import pytest
from fastapi.testclient import TestClient

from tests.integration.test_output_baseline import DEMO_VIDEO, _without_run_details
from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.middleware.auth import validate_required_api_key
from videoannotator.batch.job_execution import run_job_pipelines
from videoannotator.registry.pipeline_loader import get_pipeline_loader

PIPELINE = "scene_detection"


@pytest.mark.integration
@pytest.mark.real_models
@pytest.mark.slow
def test_in_place_and_uploaded_give_the_same_annotations(ingest_root):
    classes = get_pipeline_loader().load_all_pipelines()
    if PIPELINE not in classes:
        pytest.skip(f"{PIPELINE} is not installed")

    reset_storage_backend()
    app.dependency_overrides[validate_required_api_key] = lambda: {"is_admin": True}
    client = TestClient(app)
    try:
        shutil.copy2(DEMO_VIDEO, ingest_root / DEMO_VIDEO.name)
        in_place = client.post(
            "/api/v1/ingest",
            json={"path": str(ingest_root), "selected_pipelines": [PIPELINE]},
        ).json()["created"][0]
        uploaded = client.post(
            "/api/v1/jobs/",
            files={
                "video": (
                    DEMO_VIDEO.name,
                    io.BytesIO(DEMO_VIDEO.read_bytes()),
                    "video/mp4",
                )
            },
            data={"selected_pipelines": PIPELINE},
        ).json()["id"]
    finally:
        app.dependency_overrides.clear()

    storage = get_storage_backend()
    outputs = []
    for job_id in (in_place, uploaded):
        job = run_job_pipelines(
            storage.load_job_metadata(job_id), storage, {PIPELINE: classes[PIPELINE]}
        )
        assert job.status.value == "completed", job.error_message
        output = job.output_dir / f"{DEMO_VIDEO.stem}_{PIPELINE}.json"
        outputs.append(_without_run_details(".json", output.read_text()))

    assert outputs[0] == outputs[1]
    assert json.dumps(outputs[0])  # something was actually compared
