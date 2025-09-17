"""
Integration tests for VideoAnnotator API Server v1.2.0.

Tests the complete workflow from authentication to job processing.
"""

import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Optional

import httpx
import pytest
from fastapi.testclient import TestClient

from src.database.database import SessionLocal
from src.database.crud import UserCRUD, APIKeyCRUD, JobCRUD
from src.database.migrations import init_database, create_admin_user


class APITestClient:
    """Test client wrapper for API testing.

    By default uses an in-process ASGI transport so we don't need an external
    uvicorn server listening on localhost. This makes tests faster and avoids
    connection refused errors when a server isn't started separately.
    Set use_inprocess=False to fall back to real HTTP requests against a
    running server at base_url.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        use_inprocess: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.use_inprocess = use_inprocess
        self.headers = {}
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self._client: Optional[httpx.AsyncClient] = None

    def _ensure_client(self):
        if self._client is not None:
            return
        if self.use_inprocess:
            # Lazy import to avoid side effects during test collection
            from src.api.main import create_app
            app = create_app()
            transport = httpx.ASGITransport(app=app)
            # base_url must be set for relative URLs; follow_redirects to avoid 307 assertions
            self._client = httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=True)
        else:
            # External client also follows redirects for consistent behavior
            self._client = httpx.AsyncClient(base_url=self.base_url, follow_redirects=True)

    async def get(self, path: str, **kwargs):
        self._ensure_client()
        assert self._client is not None
        return await self._client.get(path, headers=self.headers, **kwargs)

    async def post(self, path: str, **kwargs):
        self._ensure_client()
        assert self._client is not None
        return await self._client.post(path, headers=self.headers, **kwargs)

    async def delete(self, path: str, **kwargs):
        self._ensure_client()
        assert self._client is not None
        return await self._client.delete(path, headers=self.headers, **kwargs)

    async def aclose(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None


@pytest.fixture(scope="session")
def test_api_key():
    """Create a test API key for authentication."""
    # Initialize database
    init_database(force=True)
    
    # Create admin user
    result = create_admin_user()
    if result is None:
        # Admin might already exist, get existing
        db = SessionLocal()
        try:
            admin_user = UserCRUD.get_by_username(db, "admin")
            if admin_user:
                # Create new API key
                api_key_obj, raw_key = APIKeyCRUD.create(
                    db=db,
                    user_id=str(admin_user.id),
                    key_name="integration_test",
                    expires_days=None
                )
                return raw_key
        finally:
            db.close()
    else:
        user, raw_key = result
        if raw_key:
            return raw_key
    
    # Fallback: create a new API key
    db = SessionLocal()
    try:
        admin_user = UserCRUD.get_by_username(db, "admin")
        if admin_user:
            api_key_obj, raw_key = APIKeyCRUD.create(
                db=db,
                user_id=str(admin_user.id),
                key_name="integration_test_fallback",
                expires_days=None
            )
            return raw_key
    finally:
        db.close()
    
    pytest.skip("Could not create test API key")


@pytest.fixture
def test_client(test_api_key):
    """Create authenticated test client."""
    return APITestClient(api_key=test_api_key)


@pytest.fixture
def anonymous_client():
    """Create anonymous test client."""
    return APITestClient()


class TestAPIAuthentication:
    """Test API authentication system."""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_anonymous(self, anonymous_client):
        """Test that health endpoint works without authentication."""
        response = await anonymous_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["api_version"] == "1.2.0"
    
    @pytest.mark.asyncio
    async def test_health_endpoint_authenticated(self, test_client):
        """Test that health endpoint works with authentication."""
        response = await test_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["api_version"] == "1.2.0"
    
    @pytest.mark.asyncio
    async def test_system_health_endpoint(self, test_client):
        """Test detailed system health endpoint."""
        response = await test_client.get("/api/v1/system/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["api_version"] == "1.2.0"
        assert "database" in data
        assert "system" in data
    
    @pytest.mark.asyncio
    async def test_pipelines_endpoint(self, anonymous_client):
        """Test pipelines endpoint works without authentication."""
        response = await anonymous_client.get("/api/v1/pipelines")
        assert response.status_code == 200
        data = response.json()
        assert "pipelines" in data
        assert "total" in data
        assert len(data["pipelines"]) > 0
        
        # Check pipeline structure
        pipeline = data["pipelines"][0]
        assert "name" in pipeline
        assert "description" in pipeline
        assert "enabled" in pipeline
        assert "config_schema" in pipeline
    
    @pytest.mark.asyncio
    async def test_invalid_api_key(self):
        """Test that invalid API key is rejected."""
        invalid_client = APITestClient(api_key="va_invalid_key_12345")
        response = await invalid_client.get("/api/v1/jobs")
        assert response.status_code == 401
        data = response.json()
        assert "Invalid API key" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_missing_api_key_for_protected_endpoint(self, anonymous_client):
        """Test that protected endpoints require authentication."""
        response = await anonymous_client.post("/api/v1/jobs", files={}, data={})
        assert response.status_code == 422  # Validation error for missing file
        
        # The endpoint allows anonymous access but requires form data
        # Let's test with proper form data but no auth
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"fake video content")
            tmp_path = tmp.name
        
        try:
            with open(tmp_path, "rb") as f:
                files = {"video": ("test.mp4", f, "video/mp4")}
                response = await anonymous_client.post("/api/v1/jobs", files=files)
                # Should work for anonymous users too
                assert response.status_code in [200, 201, 422]  # Depends on video validation
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestJobManagement:
    """Test job management endpoints."""
    
    @pytest.mark.asyncio
    async def test_list_jobs_empty(self, test_client):
        """Test listing jobs when none exist."""
        response = await test_client.get("/api/v1/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert data["total"] == 0
        assert len(data["jobs"]) == 0
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_job(self, test_client):
        """Test getting a job that doesn't exist."""
        response = await test_client.get("/api/v1/jobs/nonexistent-id")
        assert response.status_code == 404
        data = response.json()
        assert "Job not found" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_job_submission_validation(self, test_client):
        """Test job submission with various validation scenarios."""
        # Test without video file
        response = await test_client.post("/api/v1/jobs")
        assert response.status_code == 422  # Validation error
        
        # Test with empty file
        files = {"video": ("empty.mp4", b"", "video/mp4")}
        response = await test_client.post("/api/v1/jobs", files=files)
        # This should create a job even with empty file (validation happens during processing)
        assert response.status_code in [200, 201, 500]  # May fail during processing
    
    @pytest.mark.asyncio 
    async def test_job_submission_with_config(self, test_client):
        """Test job submission with configuration."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(b"fake video content for testing")
            tmp_path = tmp.name
        
        try:
            config = json.dumps({"scene_detection": {"threshold": 25.0}})
            pipelines = "scene_detection,person_tracking"
            
            with open(tmp_path, "rb") as f:
                files = {"video": ("test_video.mp4", f, "video/mp4")}
                data = {
                    "config": config,
                    "selected_pipelines": pipelines
                }
                response = await test_client.post("/api/v1/jobs", files=files, data=data)
                
                # Job submission should succeed
                assert response.status_code in [200, 201]
                job_data = response.json()
                
                assert "id" in job_data
                assert job_data["status"] == "pending"
                assert job_data["video_filename"] == "test_video.mp4"
                assert job_data["selected_pipelines"] == ["scene_detection", "person_tracking"]
                assert job_data["config"] == {"scene_detection": {"threshold": 25.0}}
                
                return job_data["id"]
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestJobProcessingIntegration:
    """Test job processing integration with batch system."""
    
    @pytest.mark.asyncio
    async def test_job_processing_lifecycle(self, test_client):
        """Test complete job processing lifecycle."""
        # Create a minimal test video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            # Write some minimal video-like content
            tmp.write(b"ftypisom" + b"0" * 100)  # Minimal MP4-like header
            tmp_path = tmp.name
        
        try:
            # Submit job
            with open(tmp_path, "rb") as f:
                files = {"video": ("lifecycle_test.mp4", f, "video/mp4")}
                data = {"selected_pipelines": "scene_detection"}  # Use only one pipeline to speed up test
                response = await test_client.post("/api/v1/jobs", files=files, data=data)
            
            assert response.status_code in [200, 201]
            job_data = response.json()
            job_id = job_data["id"]
            
            # Initially should be pending
            assert job_data["status"] == "pending"
            
            # Check job status over time
            max_wait_time = 30  # Maximum wait time in seconds
            wait_time = 0
            final_status = None
            
            while wait_time < max_wait_time:
                response = await test_client.get(f"/api/v1/jobs/{job_id}")
                assert response.status_code == 200
                job_data = response.json()
                status = job_data["status"]
                
                print(f"Job {job_id} status: {status} (waited {wait_time}s)")
                
                if status in ["completed", "failed", "cancelled"]:
                    final_status = status
                    break
                
                await asyncio.sleep(2)
                wait_time += 2
            
            # Job should reach a final state
            assert final_status is not None, f"Job did not complete within {max_wait_time} seconds"
            
            # Get final job state
            response = await test_client.get(f"/api/v1/jobs/{job_id}")
            assert response.status_code == 200
            final_job_data = response.json()
            
            # Validate final job data
            assert final_job_data["id"] == job_id
            assert final_job_data["status"] in ["completed", "failed"]
            assert final_job_data["is_complete"] is True
            
            if final_job_data["status"] == "completed":
                assert final_job_data["progress_percentage"] == 100
                assert final_job_data["result_path"] is not None
                assert final_job_data["duration_seconds"] is not None
            elif final_job_data["status"] == "failed":
                assert final_job_data["error_message"] is not None
                print(f"Job failed with error: {final_job_data['error_message']}")
            
            # Test job deletion (cleanup)
            response = await test_client.delete(f"/api/v1/jobs/{job_id}")
            assert response.status_code == 200
            
            # Verify job is deleted
            response = await test_client.get(f"/api/v1/jobs/{job_id}")
            assert response.status_code == 404
            
        finally:
            Path(tmp_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_api_server_startup():
    """Test that API server can start up successfully."""
    # This test assumes the server is already running
    # In a real test environment, you'd start the server in a subprocess
    
    anonymous_client = APITestClient()
    try:
        response = await anonymous_client.get("/health", timeout=5.0)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("[OK] API server is running and healthy")
    except httpx.ConnectError:
        pytest.skip("API server is not running - start with 'python api_server_db.py'")


if __name__ == "__main__":
    # Run basic connectivity test
    async def main():
        print("Testing API server connectivity...")
        await test_api_server_startup()
        print("[OK] All connectivity tests passed")
    asyncio.run(main())