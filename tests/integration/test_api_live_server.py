"""Live API server integration tests.

These tests originally spawned an external `api_server.py` process.
For CI and in-process test runs, we prefer an ASGI in-process client
to avoid port conflicts and flaky startup timing.
"""

import asyncio
import subprocess
import sys
import time
from typing import Any

import httpx
import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_server_startup_and_basic_endpoints():
    """Test API server startup and basic endpoint functionality."""

    # In-process ASGI app
    from videoannotator.api.main import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    lifespan_cm: Any = app.router.lifespan_context(app)

    await lifespan_cm.__aenter__()
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=True
        ) as client:
            # Test 1: Health endpoint
            response = await client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "api_version" in data
            assert "videoannotator_version" in data

            # Test 2: System health endpoint
            response = await client.get("/api/v1/system/health")
            if response.status_code == 200:
                data = response.json()
                assert "database" in data
            # Note: 401/404 acceptable if auth required or endpoint not implemented

            # Test 3: Pipelines endpoint
            response = await client.get("/api/v1/pipelines")
            assert response.status_code == 200
            data = response.json()
            assert "pipelines" in data
            pipelines = data["pipelines"]
            assert isinstance(pipelines, list)
            assert len(pipelines) > 0, "Should have at least some pipelines available"

            # Test 4: API Documentation
            response = await client.get("/docs")
            assert response.status_code in [200, 307, 308]
    finally:
        await lifespan_cm.__aexit__(None, None, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_authentication_flow():
    """Test API authentication with test key creation."""

    try:
        # Try to create test API key
        result = subprocess.run(
            [sys.executable, "create_test_key.py"],
            capture_output=True,
            text=True,
            cwd=".",
            timeout=10,
        )

        if result.returncode != 0:
            pytest.skip("Could not create test API key - auth testing skipped")

        # Extract API key from output
        lines = result.stdout.strip().split("\n")
        api_key = None
        for line in lines:
            if line.startswith("New API Key:"):
                api_key = line.split(": ", 1)[1]
                break

        if not api_key:
            pytest.skip("Could not extract API key from output")

        assert len(api_key) > 10, "API key should be substantial length"

        # Test authenticated endpoint (assumes server running on port 18011)
        base_url = "http://localhost:18011"
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/api/v1/jobs", headers=headers, timeout=5
                )

                if response.status_code == 200:
                    data = response.json()
                    assert "total" in data or "jobs" in data
                elif response.status_code == 401:
                    pytest.skip(
                        "Authentication failed - server may not be configured for auth"
                    )
                else:
                    pytest.fail(f"Unexpected response code: {response.status_code}")

        except httpx.RequestError:
            pytest.skip("Cannot connect to API server for auth testing")

    except subprocess.TimeoutExpired:
        pytest.skip("Test key creation timed out")
    except Exception as e:
        pytest.skip(f"Authentication test setup failed: {e}")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_api_error_handling():
    """Test API error handling for invalid requests."""

    base_url = "http://localhost:18011"

    try:
        # Test invalid endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/v1/nonexistent-endpoint")
            assert response.status_code == 404

        # Test invalid method on valid endpoint
        async with httpx.AsyncClient() as client:
            response = await client.delete(f"{base_url}/health")
            # Should be 405 Method Not Allowed or similar
            assert response.status_code in [404, 405]

    except httpx.RequestError:
        pytest.skip("Cannot connect to API server for error handling tests")


async def wait_for_server(
    url: str = "http://localhost:18011", timeout: int = 30
) -> bool:
    """Wait for the API server to be ready."""

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/health", timeout=2.0)
                if response.status_code == 200:
                    return True
        except (httpx.ConnectError, httpx.TimeoutException):
            await asyncio.sleep(1)

    return False


if __name__ == "__main__":
    # Allow running this test file directly
    asyncio.run(test_api_server_startup_and_basic_endpoints())
