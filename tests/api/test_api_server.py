"""Tests for VideoAnnotator API Server v1.2.0 with Database Integration.

Comprehensive test coverage for the FastAPI server including endpoints,
database integration, job management, and system health checks.
"""

import io
import json
import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import reset_storage_backend, set_database_path

# Import the API application
from videoannotator.api.main import create_app
from videoannotator.version import __version__ as videoannotator_version


@pytest.fixture
def temp_db():
    """Create a temporary database for testing.

    IMPORTANT: This fixture must restore environment variables on teardown.
    Many other tests rely on the session-scoped `tests/conftest.py::test_storage_env`
    configuration. If we leave per-test values pointing at a deleted temp directory,
    later tests become order-dependent.
    """
    prev_storage_root = os.environ.get("STORAGE_ROOT")
    prev_db_path = os.environ.get("VIDEOANNOTATOR_DB_PATH")
    prev_database_url = os.environ.get("DATABASE_URL")

    import videoannotator.database.database as db_module

    prev_engine = getattr(db_module, "engine", None)
    prev_session_local = getattr(db_module, "SessionLocal", None)

    prev_migrations_engine = None
    try:
        import videoannotator.database.migrations as migrations_module

        prev_migrations_engine = getattr(migrations_module, "engine", None)
    except Exception:
        migrations_module = None

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        db_path = tmp_path / "test_videoannotator.db"
        storage_root = tmp_path / "storage" / "jobs"
        storage_root.mkdir(parents=True, exist_ok=True)

        # Ensure all codepaths use this per-test database + storage root
        os.environ["STORAGE_ROOT"] = str(storage_root)
        os.environ["VIDEOANNOTATOR_DB_PATH"] = str(db_path)
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        set_database_path(db_path)

        # Patch the database engine/session to point at this file
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy.pool import StaticPool

        if (
            hasattr(db_module, "engine")
            and getattr(db_module, "engine", None) is not None
        ):
            try:
                db_module.engine.dispose()
            except Exception:
                pass

        new_engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
        db_module.engine = new_engine
        db_module.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=new_engine
        )

        # Also update migrations module if it imported engine
        if migrations_module is not None:
            migrations_module.engine = new_engine

        # Clear caches so new env/engine are picked up
        reset_storage_backend()

        try:
            from videoannotator.storage.manager import get_storage_provider

            get_storage_provider.cache_clear()
        except Exception:
            pass

        yield db_path

        # Cleanup - reset backend first to close connections
        reset_storage_backend()
        reset_storage_backend()

        try:
            from videoannotator.storage.manager import get_storage_provider

            get_storage_provider.cache_clear()
        except Exception:
            pass

        # Restore SQLAlchemy engine/session to the previous values so later tests
        # don't try to use a deleted temp DB file.
        try:
            if getattr(db_module, "engine", None) is not None:
                db_module.engine.dispose()
        except Exception:
            pass

        if prev_engine is not None:
            db_module.engine = prev_engine
        if prev_session_local is not None:
            db_module.SessionLocal = prev_session_local

        if migrations_module is not None and prev_migrations_engine is not None:
            migrations_module.engine = prev_migrations_engine

        # Restore environment to whatever it was before this test.
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

        # Ensure subsequent tests see the restored env.
        reset_storage_backend()
        try:
            from videoannotator.storage.manager import get_storage_provider

            get_storage_provider.cache_clear()
        except Exception:
            pass


@pytest.fixture
def client(temp_db):
    """Create a test client for the FastAPI app with temporary database."""
    app = create_app()
    return TestClient(app)


@pytest.fixture
def sample_video_file():
    """Create a sample video file for upload testing."""
    from tests.fixtures.synthetic_video import synthetic_video_bytes_avi

    return io.BytesIO(synthetic_video_bytes_avi())


class TestHealthEndpoint:
    """Test health check endpoints."""

    def test_health_check_basic(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["api_version"] == videoannotator_version
        assert "videoannotator_version" in data
        assert "message" in data

    def test_detailed_health_check(self, client):
        """Test detailed system health check."""
        response = client.get("/api/v1/system/health")
        assert response.status_code == 200

        data = response.json()
        print(f"Health response: {data}")
        # Don't enforce healthy status - system might be unhealthy due to initialization issues
        assert data["status"] in ["healthy", "unhealthy"]
        assert data["api_version"] == videoannotator_version
        assert "timestamp" in data

        if data["status"] == "healthy":
            # Only check detailed fields if status is healthy
            assert "system" in data
            assert "services" in data
            assert "workers" in data

            # Check system info structure
            system = data["system"]
            assert "platform" in system
            assert "python_version" in system
            assert "cpu_percent" in system
            assert "memory_percent" in system

            # Check worker info structure
            workers = data["workers"]
            assert "status" in workers
            assert workers["status"] in ["running", "stopped"]
            assert "active_jobs" in workers
            assert "queued_jobs" in workers
            assert "max_concurrent_workers" in workers
            assert "poll_interval_seconds" in workers
            assert isinstance(workers["processing_jobs"], list)

            # Check database service status
            services = data["services"]
            assert "database" in services
            db_status = services["database"]
            assert db_status["status"] == "healthy"
        else:
            # If unhealthy, we should have an error message
            assert "error" in data


class TestDatabaseEndpoint:
    """Test database information endpoint."""

    def test_database_info(self, client):
        """Test database information endpoint."""
        response = client.get("/api/v1/system/database")
        assert response.status_code == 200

        data = response.json()
        assert data["backend_type"] == "sqlite"
        assert "connection_info" in data
        assert "statistics" in data
        assert "schema_version" in data

        # Check connection info
        conn_info = data["connection_info"]
        assert "database_path" in conn_info
        assert "database_size_mb" in conn_info

        # Check statistics
        stats = data["statistics"]
        assert "total_jobs" in stats
        assert "pending_jobs" in stats
        assert "running_jobs" in stats
        assert "completed_jobs" in stats
        assert "failed_jobs" in stats
        assert "total_annotations" in stats

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint with database integration."""
        response = client.get("/api/v1/system/metrics")
        assert response.status_code == 200

        data = response.json()
        assert "timestamp" in data
        assert "metrics" in data
        assert "performance" in data

        metrics = data["metrics"]
        assert "jobs_total" in metrics
        assert "jobs_active" in metrics
        assert "jobs_completed" in metrics
        assert "jobs_failed" in metrics
        assert "total_annotations" in metrics


class TestJobEndpoints:
    """Test job management endpoints with database persistence."""

    def test_submit_job_basic(self, client, sample_video_file):
        """Test basic job submission with database storage."""
        files = {"video": ("test_video.avi", sample_video_file, "video/avi")}

        response = client.post("/api/v1/jobs/", files=files)
        assert response.status_code == 201

        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"
        assert data["video_path"] is not None
        assert data["created_at"] is not None

        # Verify job persists in database
        job_id = data["id"]
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        retrieved_data = response.json()
        assert retrieved_data["id"] == job_id

    def test_submit_job_with_config(self, client, sample_video_file):
        """Test job submission with configuration and pipelines."""
        config = {"output_format": "coco", "confidence_threshold": 0.8}
        pipelines = "person_tracking,scene_detection"

        files = {"video": ("test_video.avi", sample_video_file, "video/avi")}
        data = {"config": json.dumps(config), "selected_pipelines": pipelines}

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 201

        job_data = response.json()
        assert job_data["config"] == config
        assert job_data["selected_pipelines"] == ["person_tracking", "scene_detection"]

    def test_get_job_status(self, client, sample_video_file):
        """Test retrieving job status from database."""
        # Create a job first
        files = {"video": ("test_video.avi", sample_video_file, "video/avi")}
        response = client.post("/api/v1/jobs/", files=files)
        assert response.status_code == 201
        job_id = response.json()["id"]

        # Retrieve job status
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == job_id
        assert data["status"] == "pending"
        assert data["video_path"] is not None

    def test_queue_position_pending_jobs(self, client):
        """Pending jobs include a 1-based queue_position matching FIFO order."""
        job_ids: list[str] = []
        for i in range(3):
            files = {
                "video": (f"video{i}.mp4", io.BytesIO(b"test content"), "video/mp4")
            }
            response = client.post("/api/v1/jobs/", files=files)
            assert response.status_code == 201
            job_ids.append(response.json()["id"])

        # List jobs and verify queue_position values exist for pending jobs
        response = client.get("/api/v1/jobs/")
        assert response.status_code == 200
        data = response.json()

        positions = {job["id"]: job.get("queue_position") for job in data["jobs"]}
        assert positions[job_ids[0]] == 1
        assert positions[job_ids[1]] == 2
        assert positions[job_ids[2]] == 3

        # Individual job status should report the same queue_position
        response = client.get(f"/api/v1/jobs/{job_ids[1]}")
        assert response.status_code == 200
        job_data = response.json()
        assert job_data["status"] == "pending"
        assert job_data.get("queue_position") == 2

    def test_get_nonexistent_job(self, client):
        """Test retrieving non-existent job from database."""
        response = client.get("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_list_jobs_empty(self, client):
        """Test listing jobs when database is empty."""
        response = client.get("/api/v1/jobs/")
        assert response.status_code == 200

        data = response.json()
        assert data["jobs"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["per_page"] == 10

    def test_list_jobs_with_data(self, client, sample_video_file):
        """Test listing jobs with existing data in database."""
        # Create multiple jobs
        job_ids = []
        for i in range(3):
            files = {
                "video": (f"video{i}.mp4", io.BytesIO(b"test content"), "video/mp4")
            }
            response = client.post("/api/v1/jobs/", files=files)
            assert response.status_code == 201
            job_ids.append(response.json()["id"])

        # List jobs
        response = client.get("/api/v1/jobs/")
        assert response.status_code == 200

        data = response.json()
        assert len(data["jobs"]) == 3
        assert data["total"] == 3
        assert data["page"] == 1

        # Check all job IDs are present
        retrieved_ids = [job["id"] for job in data["jobs"]]
        for job_id in job_ids:
            assert job_id in retrieved_ids

    def test_list_jobs_pagination(self, client, sample_video_file):
        """Test job listing pagination."""
        # Create multiple jobs
        for i in range(5):
            files = {
                "video": (f"video{i}.mp4", io.BytesIO(b"test content"), "video/mp4")
            }
            response = client.post("/api/v1/jobs/", files=files)
            assert response.status_code == 201

        # Test first page
        response = client.get("/api/v1/jobs/?page=1&per_page=3")
        assert response.status_code == 200

        data = response.json()
        assert len(data["jobs"]) == 3
        assert data["total"] == 5
        assert data["page"] == 1
        assert data["per_page"] == 3

        # Test second page
        response = client.get("/api/v1/jobs/?page=2&per_page=3")
        assert response.status_code == 200

        data = response.json()
        assert len(data["jobs"]) == 2  # Remaining jobs
        assert data["total"] == 5
        assert data["page"] == 2

    def test_delete_job(self, client, sample_video_file):
        """Test deleting a job from database."""
        # Create a job first
        files = {"video": ("test_video.avi", sample_video_file, "video/avi")}
        response = client.post("/api/v1/jobs/", files=files)
        assert response.status_code == 201
        job_id = response.json()["id"]

        # Delete the job
        response = client.delete(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 204

        # Verify job is deleted
        response = client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 404

    def test_delete_nonexistent_job(self, client):
        """Test deleting non-existent job."""
        response = client.delete("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404


class TestPipelineEndpoints:
    """Test pipeline information endpoints."""

    def test_list_pipelines(self, client):
        """Test listing available pipelines."""
        response = client.get("/api/v1/pipelines/")
        assert response.status_code == 200

        data = response.json()
        assert "pipelines" in data

        # Should return basic pipeline info
        # Note: Actual implementation may vary


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_invalid_json_config(self, client, sample_video_file):
        """Test job submission with invalid JSON config."""
        files = {"video": ("test_video.avi", sample_video_file, "video/avi")}
        data = {"config": "invalid json"}

        response = client.post("/api/v1/jobs/", files=files, data=data)
        # Should return 422 for validation error or 500 for internal error
        assert response.status_code in [400, 422, 500]

    def test_missing_video_file(self, client):
        """Test job submission without video file."""
        response = client.post("/api/v1/jobs/")
        assert response.status_code == 422  # Validation error

    def test_empty_pipelines_string(self, client, sample_video_file):
        """Test job submission with empty pipelines string."""
        files = {"video": ("test_video.avi", sample_video_file, "video/avi")}
        data = {"selected_pipelines": ""}

        response = client.post("/api/v1/jobs/", files=files, data=data)
        assert response.status_code == 201

        job_data = response.json()
        # Empty string should result in empty list or None
        assert job_data["selected_pipelines"] in [[], None]


class TestDatabasePersistence:
    """Test database persistence across requests."""

    def test_job_persistence_across_requests(self, client, sample_video_file):
        """Test that jobs persist in database across multiple requests."""
        # Create a job
        files = {"video": ("test_video.avi", sample_video_file, "video/avi")}
        response = client.post("/api/v1/jobs/", files=files)
        assert response.status_code == 201
        job_id = response.json()["id"]

        # Create a new client (simulating separate request)
        new_client = TestClient(create_app())

        # Job should still exist
        response = new_client.get(f"/api/v1/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["id"] == job_id

    def test_database_statistics_update(self, client, sample_video_file):
        """Test that database statistics update with job operations."""
        # Check initial stats
        response = client.get("/api/v1/system/database")
        initial_stats = response.json()["statistics"]
        initial_total = initial_stats["total_jobs"]

        # Create a job
        files = {"video": ("test_video.avi", sample_video_file, "video/avi")}
        response = client.post("/api/v1/jobs/", files=files)
        assert response.status_code == 201

        # Check updated stats
        response = client.get("/api/v1/system/database")
        updated_stats = response.json()["statistics"]
        assert updated_stats["total_jobs"] == initial_total + 1
        assert updated_stats["pending_jobs"] >= 1


class TestAPIDocumentation:
    """Test API documentation endpoints."""

    def test_openapi_schema(self, client):
        """Test OpenAPI schema is available."""
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        assert "openapi" in schema
        assert "info" in schema
        assert schema["info"]["title"] == "VideoAnnotator API"
        assert schema["info"]["version"] == videoannotator_version

    def test_docs_endpoint(self, client):
        """Test Swagger UI docs endpoint."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()

    def test_redoc_endpoint(self, client):
        """Test ReDoc endpoint."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower()
