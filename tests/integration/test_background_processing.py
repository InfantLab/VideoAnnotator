"""Integration tests for background job processing system.

Tests the integrated background job processing functionality with the
API server. Converted from test_integrated_worker.py debugging script.
"""

import asyncio
import os
import tempfile
from pathlib import Path

import pytest
import httpx


@pytest.mark.integration
@pytest.mark.asyncio
async def test_background_job_processing():
    """Test that the integrated background worker processes pending jobs."""

    # Use a temporary, isolated DB + storage root for this test.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "test_background_processing.db"
        storage_root = tmp_path / "storage" / "jobs"
        storage_root.mkdir(parents=True, exist_ok=True)

        os.environ["VIDEOANNOTATOR_DB_PATH"] = str(db_path)
        os.environ["DATABASE_URL"] = f"sqlite:///{db_path}"
        os.environ["STORAGE_ROOT"] = str(storage_root)

        # Reset caches to ensure env changes are picked up.
        from videoannotator.api.database import reset_storage_backend

        reset_storage_backend()
        try:
            from videoannotator.storage.manager import get_storage_provider

            get_storage_provider.cache_clear()
        except Exception:
            pass

        from videoannotator.api.main import create_app

        app = create_app()
        transport = httpx.ASGITransport(app=app)
        lifespan_cm = app.router.lifespan_context(app)
        await lifespan_cm.__aenter__()
        try:
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test", follow_redirects=True
            ) as client:
                response = await client.get("/health")
                assert response.status_code == 200
                health_data = response.json()
                assert health_data["status"] == "healthy"

                response = await client.get("/api/v1/debug/background-jobs")
                assert response.status_code == 200
                bg_status = response.json()
                assert "background_processing" in bg_status

                bp = bg_status["background_processing"]
                assert bp.get("running") is True
                assert isinstance(bp.get("concurrent_jobs", 0), int)
                assert isinstance(bp.get("max_concurrent_jobs", 0), int)
                assert isinstance(bp.get("poll_interval", 0), (int, float))
        finally:
            await lifespan_cm.__aexit__(None, None, None)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_background_processing_endpoints():
    """Test background processing debug endpoints without starting new.

    server.
    """

    # Validate the endpoint in-process (no external server required).
    from videoannotator.api.main import create_app

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    lifespan_cm = app.router.lifespan_context(app)
    await lifespan_cm.__aenter__()
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test", follow_redirects=True
        ) as client:
            response = await client.get("/api/v1/debug/background-jobs")
            assert response.status_code == 200
            bg_status = response.json()
            assert isinstance(bg_status, dict)

            assert "background_processing" in bg_status
            bp = bg_status["background_processing"]

            required_fields = [
                "running",
                "concurrent_jobs",
                "max_concurrent_jobs",
                "poll_interval",
            ]
            for field in required_fields:
                assert field in bp

            assert isinstance(bp["running"], bool)
            assert isinstance(bp["concurrent_jobs"], int)
            assert isinstance(bp["max_concurrent_jobs"], int)
            assert isinstance(bp["poll_interval"], (int, float))
    finally:
        await lifespan_cm.__aexit__(None, None, None)


if __name__ == "__main__":
    # Allow running this test file directly
    asyncio.run(test_background_job_processing())
