"""Integration tests for StorageProvider with API and Worker."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import reset_storage_backend, set_database_path
from videoannotator.api.dependencies import get_current_user
from videoannotator.api.main import create_app


@pytest.fixture
def temp_storage_root():
    """Create a temporary storage root."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def client(temp_storage_root):
    """Create a test client with a temporary database and storage."""
    prev_storage_root = os.environ.get("STORAGE_ROOT")
    prev_db_path = os.environ.get("VIDEOANNOTATOR_DB_PATH")
    prev_database_url = os.environ.get("DATABASE_URL")

    # Setup DB
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    set_database_path(Path(db_path))
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"

    # Setup Storage Provider
    os.environ["STORAGE_ROOT"] = str(temp_storage_root)

    reset_storage_backend()
    try:
        from videoannotator.storage.manager import get_storage_provider

        get_storage_provider.cache_clear()
    except Exception:
        pass

    app = create_app()

    # Override auth dependency
    app.dependency_overrides[get_current_user] = lambda: {
        "id": "test-user",
        "username": "test",
    }

    with TestClient(app) as client:
        yield client

    # Cleanup DB
    reset_storage_backend()
    if os.path.exists(db_path):
        os.unlink(db_path)

    # Restore env and clear caches for other tests.
    if prev_storage_root is None:
        os.environ.pop("STORAGE_ROOT", None)
    else:
        os.environ["STORAGE_ROOT"] = prev_storage_root

    if prev_db_path is None:
        os.environ.pop("VIDEOANNOTATOR_DB_PATH", None)
    else:
        os.environ["VIDEOANNOTATOR_DB_PATH"] = prev_db_path

    if prev_database_url is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = prev_database_url

    reset_storage_backend()
    try:
        from videoannotator.storage.manager import get_storage_provider

        get_storage_provider.cache_clear()
    except Exception:
        pass


def test_job_submission_creates_files_in_storage(client, temp_storage_root):
    """Test that submitting a job creates files in the configured storage root."""
    # Create a dummy video file
    video_content = b"fake video content"
    files = {"video": ("test.mp4", video_content, "video/mp4")}

    response = client.post("/api/v1/jobs/", files=files)
    assert response.status_code == 201
    data = response.json()
    job_id = data["id"]

    # Check if job directory exists in temp_storage_root
    job_dir = temp_storage_root / job_id
    assert job_dir.exists()
    assert job_dir.is_dir()

    # Check if video file exists
    video_path = job_dir / "test.mp4"
    assert video_path.exists()
    assert video_path.read_bytes() == video_content

    # Check response paths
    assert data["storage_path"] == str(job_dir)
    assert data["video_path"] == str(video_path)


def test_download_artifacts(client, temp_storage_root):
    """Test downloading artifacts via API."""
    # 1. Submit job
    video_content = b"fake video content"
    files = {"video": ("test.mp4", video_content, "video/mp4")}
    response = client.post("/api/v1/jobs/", files=files)
    job_id = response.json()["id"]

    # 2. Manually add some artifacts (simulating pipeline output)
    storage_path = Path(response.json()["storage_path"])
    assert storage_path.exists()

    (storage_path / "results.json").write_text('{"foo": "bar"}')
    (storage_path / "annotations.vtt").write_text("WEBVTT")

    # 3. Download zip
    response = client.get(f"/api/v1/jobs/{job_id}/artifacts")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    # Verify zip content
    import io
    import zipfile

    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        assert "results.json" in zf.namelist()
        assert "annotations.vtt" in zf.namelist()
        assert zf.read("results.json") == b'{"foo": "bar"}'
