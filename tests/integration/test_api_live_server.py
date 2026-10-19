"""Live API server integration tests.

These tests originally spawned an external `api_server.py` process.
For CI and in-process test runs, we prefer an ASGI in-process client
to avoid port conflicts and flaky startup timing.
"""

import asyncio
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
            assert response.status_code == 200
            assert "database" in response.json()

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
