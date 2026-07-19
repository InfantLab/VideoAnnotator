"""Contract test: unavailable-pipeline error shape on every user-facing
surface that can name a pipeline.

Contract: specs/004-extras-based-install/contracts/unavailable-pipeline-error.md
- API: 422 (not 500), JSON body with `detail`, `install_hint`, `pipeline`.
- CLI: non-zero exit, stderr message, no Python traceback.
"""

import io
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from videoannotator.api.database import reset_storage_backend, set_database_path
from videoannotator.api.main import create_app
from videoannotator.cli import app as cli_app


@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)
    set_database_path(db_path)
    reset_storage_backend()

    app = create_app()
    yield TestClient(app)

    reset_storage_backend()
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def sample_video_file():
    from tests.fixtures.synthetic_video import synthetic_video_bytes_avi

    return io.BytesIO(synthetic_video_bytes_avi())


class TestAPIUnavailablePipelineErrorShape:
    def test_submit_job_with_unavailable_pipeline_returns_422(
        self, client, sample_video_file, monkeypatch
    ):
        import videoannotator.api.v1.jobs as jobs_module

        monkeypatch.setattr(jobs_module, "extras_available", lambda extras: False)

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {"selected_pipelines": "face_analysis"}

        response = client.post("/api/v1/jobs/", files=files, data=data)

        assert response.status_code == 422
        body = response.json()
        assert "face_analysis" in body["detail"]
        assert body["pipeline"] == "face_analysis"
        assert body["install_hint"] == "pip install videoannotator[face]"

    def test_submit_job_with_demoted_laion_pipeline_gets_migration_message(
        self, client, sample_video_file, monkeypatch
    ):
        import videoannotator.api.v1.jobs as jobs_module

        monkeypatch.setattr(jobs_module, "extras_available", lambda extras: False)

        files = {"video": ("test.avi", sample_video_file, "video/avi")}
        data = {"selected_pipelines": "face_laion_clip"}

        response = client.post("/api/v1/jobs/", files=files, data=data)

        assert response.status_code == 422
        body = response.json()
        assert "no longer installed by default" in body["detail"]
        assert body["install_hint"] == "pip install videoannotator[face-laion]"


class TestCLIUnavailablePipelineErrorShape:
    def test_job_submit_unavailable_pipeline_exits_nonzero_no_traceback(self, tmp_path):
        video_path = tmp_path / "sample.mp4"
        video_path.write_bytes(b"not a real video, just needs to exist")

        fake_response = MagicMock()
        fake_response.status_code = 422
        fake_response.json.return_value = {
            "detail": "Pipeline 'face_laion_clip' is not available in this install. "
            "As of v1.5.0, pipelines requiring the 'face-laion' extras group "
            "are no longer installed by default.",
            "install_hint": "pip install videoannotator[face-laion]",
            "pipeline": "face_laion_clip",
        }

        runner = CliRunner()
        with patch("requests.post", return_value=fake_response):
            result = runner.invoke(
                cli_app,
                [
                    "job",
                    "submit",
                    str(video_path),
                    "--pipelines",
                    "face_laion_clip",
                ],
            )

        assert result.exit_code != 0
        assert "Traceback" not in result.output
        assert "Error:" in result.output
        assert "pip install videoannotator[face-laion]" in result.output
