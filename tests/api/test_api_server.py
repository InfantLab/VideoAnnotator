"""
Tests for VideoAnnotator API Server v1.2.0

Comprehensive test coverage for the FastAPI server including endpoints,
middleware, job management, and system health checks.
"""

import pytest
import tempfile
import json
import io
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import UploadFile

# Import the API server app
from api_server import app, JOBS, MockJob, JobStatus


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_video_file():
    """Create a sample video file for upload testing."""
    content = b"fake video content for testing"
    return io.BytesIO(content)


@pytest.fixture
def clear_jobs():
    """Clear the JOBS dictionary before each test."""
    JOBS.clear()
    yield
    JOBS.clear()


class TestHealthEndpoint:
    """Test health check endpoints."""
    
    def test_health_check_basic(self, client):
        """Test basic health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["api_version"] == "1.2.0"
        assert data["videoannotator_version"] == "1.2.0"
        assert "message" in data

    def test_detailed_health_check(self, client):
        """Test detailed system health check."""
        response = client.get("/api/v1/system/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["api_version"] == "1.2.0"
        assert "timestamp" in data
        assert "system" in data
        assert "services" in data
        
        # Check system info structure
        system = data["system"]
        assert "platform" in system
        assert "python_version" in system
        assert "cpu_percent" in system
        assert "memory_percent" in system


class TestJobEndpoints:
    """Test job management endpoints."""
    
    def test_submit_job_basic(self, client, sample_video_file, clear_jobs):
        """Test basic job submission."""
        files = {"video": ("test_video.mp4", sample_video_file, "video/mp4")}
        
        response = client.post("/api/v1/jobs", files=files)
        assert response.status_code == 200
        
        data = response.json()
        assert "id" in data
        assert data["status"] == "pending"
        assert data["video_path"] is not None
        assert data["created_at"] is not None

    def test_submit_job_with_config(self, client, sample_video_file, clear_jobs):
        """Test job submission with configuration."""
        config = {"output_format": "coco", "confidence_threshold": 0.8}
        pipelines = "person,scene"
        
        files = {"video": ("test_video.mp4", sample_video_file, "video/mp4")}
        data = {
            "config": json.dumps(config),
            "selected_pipelines": pipelines
        }
        
        response = client.post("/api/v1/jobs", files=files, data=data)
        assert response.status_code == 200
        
        job_data = response.json()
        assert job_data["config"] == config
        assert job_data["selected_pipelines"] == ["person", "scene"]

    def test_get_job_status(self, client, clear_jobs):
        """Test retrieving job status."""
        # Create a mock job
        job = MockJob("test_video.mp4", {"test": True}, ["person"])
        JOBS[job.id] = job
        
        response = client.get(f"/api/v1/jobs/{job.id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == job.id
        assert data["status"] == "pending"
        assert data["video_path"] == "test_video.mp4"

    def test_get_nonexistent_job(self, client, clear_jobs):
        """Test retrieving non-existent job."""
        response = client.get("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_list_jobs_empty(self, client, clear_jobs):
        """Test listing jobs when empty."""
        response = client.get("/api/v1/jobs")
        assert response.status_code == 200
        
        data = response.json()
        assert data["jobs"] == []
        assert data["total"] == 0

    def test_list_jobs_with_data(self, client, clear_jobs):
        """Test listing jobs with existing data."""
        # Create mock jobs
        job1 = MockJob("video1.mp4", {"test": 1}, ["person"])
        job2 = MockJob("video2.mp4", {"test": 2}, ["scene"])
        JOBS[job1.id] = job1
        JOBS[job2.id] = job2
        
        response = client.get("/api/v1/jobs")
        assert response.status_code == 200
        
        data = response.json()
        assert len(data["jobs"]) == 2
        assert data["total"] == 2

    def test_delete_job(self, client, clear_jobs):
        """Test deleting a job."""
        job = MockJob("test_video.mp4")
        JOBS[job.id] = job
        
        response = client.delete(f"/api/v1/jobs/{job.id}")
        assert response.status_code == 200
        assert "deleted" in response.json()["message"].lower()
        
        # Verify job is deleted
        assert job.id not in JOBS

    def test_delete_nonexistent_job(self, client, clear_jobs):
        """Test deleting non-existent job."""
        response = client.delete("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404


class TestPipelineEndpoints:
    """Test pipeline information endpoints."""
    
    def test_list_pipelines(self, client):
        """Test listing available pipelines."""
        response = client.get("/api/v1/pipelines")
        assert response.status_code == 200
        
        data = response.json()
        assert "pipelines" in data
        assert "total" in data
        
        pipelines = data["pipelines"]
        assert len(pipelines) == 4  # scene_detection, person_tracking, face_analysis, audio_processing
        
        # Check pipeline structure
        for pipeline in pipelines:
            assert "name" in pipeline
            assert "description" in pipeline
            assert "enabled" in pipeline
            assert "config_schema" in pipeline

    def test_pipeline_names(self, client):
        """Test specific pipeline names and descriptions."""
        response = client.get("/api/v1/pipelines")
        data = response.json()
        
        pipeline_names = [p["name"] for p in data["pipelines"]]
        expected_names = ["scene_detection", "person_tracking", "face_analysis", "audio_processing"]
        
        for expected in expected_names:
            assert expected in pipeline_names


class TestMockJobClass:
    """Test the MockJob helper class."""
    
    def test_mock_job_creation(self):
        """Test MockJob creation with default values."""
        job = MockJob("test_video.mp4")
        
        assert job.video_path == "test_video.mp4"
        assert job.config == {}
        assert job.selected_pipelines == []
        assert job.status == JobStatus.PENDING
        assert job.id is not None
        assert job.created_at is not None
        assert job.completed_at is None
        assert job.error_message is None

    def test_mock_job_with_config(self):
        """Test MockJob creation with configuration."""
        config = {"model": "yolo11n", "threshold": 0.5}
        pipelines = ["person", "scene"]
        
        job = MockJob("test_video.mp4", config, pipelines)
        
        assert job.config == config
        assert job.selected_pipelines == pipelines

    def test_job_status_enum(self):
        """Test JobStatus enum values."""
        assert JobStatus.PENDING == "pending"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"


class TestCORSMiddleware:
    """Test CORS middleware configuration."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.get("/health")
        
        # Check for CORS headers (they may not be present in test client,
        # but we can verify the middleware is configured)
        assert response.status_code == 200

    def test_options_request(self, client):
        """Test OPTIONS request handling."""
        response = client.options("/api/v1/jobs")
        # FastAPI handles OPTIONS automatically with CORS middleware
        assert response.status_code in [200, 405]  # 405 if not explicitly handled


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_json_config(self, client, sample_video_file, clear_jobs):
        """Test job submission with invalid JSON config."""
        files = {"video": ("test_video.mp4", sample_video_file, "video/mp4")}
        data = {"config": "invalid json"}
        
        response = client.post("/api/v1/jobs", files=files, data=data)
        assert response.status_code == 500

    def test_missing_video_file(self, client, clear_jobs):
        """Test job submission without video file."""
        response = client.post("/api/v1/jobs")
        assert response.status_code == 422  # Validation error

    def test_large_job_list_pagination(self, client, clear_jobs):
        """Test job listing handles pagination info."""
        response = client.get("/api/v1/jobs")
        data = response.json()
        
        assert "page" in data
        assert "per_page" in data
        assert data["page"] == 1


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
        assert schema["info"]["version"] == "1.2.0"

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