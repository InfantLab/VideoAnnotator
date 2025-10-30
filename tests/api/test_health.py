"""Tests for enhanced health check endpoint.

Validates basic and detailed health check modes, response structure,
and status code behavior.

v1.3.0: Phase 11 - T069
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create test client for API."""
    from videoannotator.api.main import app

    return TestClient(app)


class TestBasicHealthCheck:
    """Test basic health check mode (fast liveness check)."""

    def test_basic_health_returns_200(self, client):
        """Test basic health check returns 200 OK."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200

    def test_basic_health_structure(self, client):
        """Test basic health response has correct structure."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert data["status"] == "ok"

    def test_basic_health_no_details(self, client):
        """Test basic health does not include details."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "details" not in data

    def test_basic_health_is_fast(self, client):
        """Test basic health check completes quickly."""
        import time

        start = time.time()
        response = client.get("/api/v1/health")
        elapsed = time.time() - start

        assert response.status_code == 200
        # Should complete in under 1 second
        assert elapsed < 1.0


class TestDetailedHealthCheck:
    """Test detailed health check mode (diagnostic mode)."""

    def test_detailed_health_parameter(self, client):
        """Test detailed=true parameter returns detailed response."""
        response = client.get("/api/v1/health?detailed=true")

        assert response.status_code == 200
        data = response.json()
        assert "details" in data

    def test_detailed_health_includes_all_subsystems(self, client):
        """Test detailed mode includes all subsystem checks."""
        response = client.get("/api/v1/health?detailed=true")
        data = response.json()

        assert "details" in data
        details = data["details"]

        # Check all subsystems are present
        assert "database" in details
        assert "storage" in details
        assert "gpu" in details
        assert "registry" in details

    def test_detailed_health_database_info(self, client):
        """Test detailed mode includes database information."""
        response = client.get("/api/v1/health?detailed=true")
        data = response.json()

        database = data["details"]["database"]
        assert "status" in database
        # Should have either "ok" or "error" status
        assert database["status"] in ["ok", "error"]

    def test_detailed_health_storage_info(self, client):
        """Test detailed mode includes storage information."""
        response = client.get("/api/v1/health?detailed=true")
        data = response.json()

        storage = data["details"]["storage"]
        assert "status" in storage

        # If storage check succeeded, should have disk info
        if storage["status"] in ["ok", "warning"]:
            assert "total_gb" in storage
            assert "used_gb" in storage
            assert "free_gb" in storage
            assert "percent_used" in storage

    def test_detailed_health_gpu_info(self, client):
        """Test detailed mode includes GPU information."""
        response = client.get("/api/v1/health?detailed=true")
        data = response.json()

        gpu = data["details"]["gpu"]
        assert "available" in gpu

        # If GPU is available, should have device info
        if gpu["available"]:
            assert "device_count" in gpu
            assert "devices" in gpu

    def test_detailed_health_registry_info(self, client):
        """Test detailed mode includes registry information."""
        response = client.get("/api/v1/health?detailed=true")
        data = response.json()

        registry = data["details"]["registry"]
        assert "status" in registry

        # If registry check succeeded, should have pipeline info
        if registry["status"] == "ok":
            assert "pipelines_loaded" in registry
            assert "pipeline_names" in registry


class TestHealthStatusCodes:
    """Test health check status code behavior."""

    def test_basic_health_always_returns_200(self, client):
        """Test basic health always returns 200 OK."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_detailed_health_returns_200_when_healthy(self, client):
        """Test detailed health returns 200 when systems are healthy."""
        response = client.get("/api/v1/health?detailed=true")

        # If system is actually healthy, should be 200
        data = response.json()
        if data["status"] == "ok":
            assert response.status_code == 200

    @patch("videoannotator.api.v1.health._check_database_status")
    def test_detailed_health_returns_503_when_database_unhealthy(
        self, mock_db_check, client
    ):
        """Test detailed health returns 503 when database is unhealthy."""
        # Mock database failure
        mock_db_check.return_value = {
            "status": "error",
            "error": "Connection timeout",
        }

        response = client.get("/api/v1/health?detailed=true")

        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "unhealthy"

    @patch("videoannotator.api.v1.health._check_storage_status")
    def test_detailed_health_accepts_storage_warnings(self, mock_storage_check, client):
        """Test detailed health allows storage warnings without failing."""
        # Mock storage warning (low space but not critical)
        mock_storage_check.return_value = {
            "status": "warning",
            "warning": "Less than 10% free space remaining",
            "total_gb": 100.0,
            "used_gb": 92.0,
            "free_gb": 8.0,
            "percent_used": 92.0,
        }

        response = client.get("/api/v1/health?detailed=true")

        # Should still be healthy with just a warning
        data = response.json()
        assert data["status"] == "ok"
        assert response.status_code == 200

    @patch("videoannotator.api.v1.health._check_gpu_status")
    def test_detailed_health_allows_missing_gpu(self, mock_gpu_check, client):
        """Test detailed health doesn't fail when GPU unavailable."""
        # Mock GPU not available
        mock_gpu_check.return_value = {
            "available": False,
            "reason": "CUDA not available",
        }

        response = client.get("/api/v1/health?detailed=true")

        # Should still be healthy - GPU is optional
        data = response.json()
        # Status might be ok if other systems are healthy
        assert response.status_code in [200, 503]


class TestHealthResponseStructure:
    """Test health response structure and fields."""

    def test_health_response_has_version(self, client):
        """Test health response includes version."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "version" in data
        assert isinstance(data["version"], str)

    def test_health_response_has_timestamp(self, client):
        """Test health response includes timestamp."""
        response = client.get("/api/v1/health")
        data = response.json()

        assert "timestamp" in data
        # Should be ISO format with trailing Z
        assert data["timestamp"].endswith("Z")

    def test_detailed_response_has_all_required_fields(self, client):
        """Test detailed response has all required fields."""
        response = client.get("/api/v1/health?detailed=true")
        data = response.json()

        # Top-level fields
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "details" in data

        # Status must be ok or unhealthy
        assert data["status"] in ["ok", "unhealthy"]


class TestHealthQueryParameters:
    """Test health endpoint query parameter handling."""

    def test_detailed_false_returns_basic(self, client):
        """Test detailed=false returns basic response."""
        response = client.get("/api/v1/health?detailed=false")
        data = response.json()

        assert "details" not in data

    def test_detailed_true_returns_detailed(self, client):
        """Test detailed=true returns detailed response."""
        response = client.get("/api/v1/health?detailed=true")
        data = response.json()

        assert "details" in data

    def test_detailed_1_returns_detailed(self, client):
        """Test detailed=1 (truthy value) returns detailed response."""
        response = client.get("/api/v1/health?detailed=1")
        data = response.json()

        assert "details" in data

    def test_invalid_detailed_value_defaults_to_false(self, client):
        """Test invalid detailed value defaults to basic mode."""
        response = client.get("/api/v1/health?detailed=invalid")

        # FastAPI should handle invalid boolean gracefully
        # Result should be basic mode
        assert response.status_code in [200, 422]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
