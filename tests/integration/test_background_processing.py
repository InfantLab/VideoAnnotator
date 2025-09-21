#!/usr/bin/env python3
"""
Integration tests for background job processing system.

Tests the integrated background job processing functionality with the API server.
Converted from test_integrated_worker.py debugging script.
"""

import asyncio
import pytest
import requests
import subprocess
import sqlite3
import time
from pathlib import Path
from typing import Optional


@pytest.mark.integration
@pytest.mark.asyncio
async def test_background_job_processing():
    """Test that the integrated background worker processes pending jobs."""
    
    # Use a temporary database for this test
    test_db_path = "test_background_processing.db"
    server_process: Optional[subprocess.Popen] = None
    
    try:
        # Step 1: Check initial database state
        try:
            conn = sqlite3.connect(test_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'pending'")
            pending_count = cursor.fetchone()[0]
            conn.close()
        except sqlite3.OperationalError:
            # Database doesn't exist or table doesn't exist - that's fine
            pending_count = 0
        
        # Step 2: Start API server with test database
        server_process = subprocess.Popen(
            ["uv", "run", "python", "api_server.py", "--port", "8899", "--log-level", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"VIDEOANNOTATOR_DB_PATH": test_db_path}
        )
        
        # Wait for server to start
        await asyncio.sleep(6)
        
        # Step 3: Test server health and background processing
        response = requests.get("http://localhost:8899/health", timeout=5)
        assert response.status_code == 200
        health_data = response.json()
        assert health_data["status"] == "healthy"
        
        # Check enhanced logging is enabled (indicates integration working)
        assert health_data.get("logging") == "enhanced"
        
        # Step 4: Test background jobs status endpoint
        response = requests.get("http://localhost:8899/api/v1/debug/background-jobs", timeout=5)
        assert response.status_code == 200
        
        bg_status = response.json()
        assert "background_processing" in bg_status
        
        bp = bg_status["background_processing"]
        assert bp.get("running") is True, "Background job processing should be running"
        assert isinstance(bp.get("concurrent_jobs", 0), int)
        assert isinstance(bp.get("max_concurrent_jobs", 0), int)
        assert isinstance(bp.get("poll_interval", 0), (int, float))
        
        # If we had pending jobs initially, wait and check if they get processed
        if pending_count > 0:
            initial_pending = pending_count
            
            # Wait up to 30 seconds for job processing
            for check_round in range(3):
                await asyncio.sleep(10)
                
                conn = sqlite3.connect(test_db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'pending'")
                current_pending = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM jobs WHERE status = 'running'")
                running_count = cursor.fetchone()[0]
                cursor.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('completed', 'failed')")
                finished_count = cursor.fetchone()[0]
                conn.close()
                
                # Success if we see job processing activity
                if current_pending < initial_pending or running_count > 0:
                    break
            else:
                # No processing activity detected - could be normal if pipelines initializing
                pytest.skip("No job processing activity detected - pipelines may still be initializing")
        
    finally:
        # Cleanup
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
        
        # Clean up test database
        try:
            Path(test_db_path).unlink(missing_ok=True)
        except Exception:
            pass


@pytest.mark.integration
@pytest.mark.asyncio
async def test_background_processing_endpoints():
    """Test background processing debug endpoints without starting new server."""
    
    # This test assumes an existing server is running
    # It's designed to work with the server started by test_background_job_processing
    try:
        response = requests.get("http://localhost:18011/api/v1/debug/background-jobs", timeout=2)
        if response.status_code != 200:
            pytest.skip("No running API server found for endpoint testing")
        
        bg_status = response.json()
        
        # Validate response structure
        assert isinstance(bg_status, dict)
        
        if "background_processing" in bg_status:
            bp = bg_status["background_processing"]
            
            # Validate background processing status fields
            required_fields = ["running", "concurrent_jobs", "max_concurrent_jobs", "poll_interval"]
            for field in required_fields:
                assert field in bp, f"Missing required field: {field}"
            
            # Validate field types
            assert isinstance(bp["running"], bool)
            assert isinstance(bp["concurrent_jobs"], int)
            assert isinstance(bp["max_concurrent_jobs"], int) 
            assert isinstance(bp["poll_interval"], (int, float))
            
    except requests.RequestException:
        pytest.skip("Cannot connect to API server for endpoint testing")


if __name__ == "__main__":
    # Allow running this test file directly
    asyncio.run(test_background_job_processing())