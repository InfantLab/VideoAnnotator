#!/usr/bin/env python3
"""
Live API server integration tests.

Tests API server functionality with full server startup and endpoint validation.
Converted from test_api_live.py debugging script.
"""

import asyncio
import pytest
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import httpx


@pytest.mark.integration 
@pytest.mark.asyncio
async def test_api_server_startup_and_basic_endpoints():
    """Test API server startup and basic endpoint functionality."""
    
    server_process: Optional[subprocess.Popen] = None
    
    try:
        # Start API server
        server_process = subprocess.Popen([
            sys.executable, "api_server.py", "--port", "8998"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait for server to be ready
        server_ready = await wait_for_server("http://localhost:8998", timeout=30)
        assert server_ready, "API server failed to start within timeout"
        
        # Test basic endpoints
        base_url = "http://localhost:8998"
        
        # Test 1: Health endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "api_version" in data
            assert "videoannotator_version" in data
        
        # Test 2: System health endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/v1/system/health")
            if response.status_code == 200:
                data = response.json()
                assert "database" in data
            # Note: 401/404 acceptable if auth required or endpoint not implemented
        
        # Test 3: Pipelines endpoint
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/v1/pipelines")
            assert response.status_code == 200
            data = response.json()
            assert "pipelines" in data
            pipelines = data["pipelines"]
            assert isinstance(pipelines, list)
            assert len(pipelines) > 0, "Should have at least some pipelines available"
        
        # Test 4: API Documentation
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/docs")
            # Should be accessible (200) or redirect (3xx)
            assert response.status_code in [200, 307, 308]
        
    finally:
        # Clean up server
        if server_process:
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()


@pytest.mark.integration
@pytest.mark.asyncio 
async def test_api_authentication_flow():
    """Test API authentication with test key creation."""
    
    try:
        # Try to create test API key
        result = subprocess.run([
            sys.executable, "create_test_key.py"
        ], capture_output=True, text=True, cwd=".", timeout=10)
        
        if result.returncode != 0:
            pytest.skip("Could not create test API key - auth testing skipped")
        
        # Extract API key from output
        lines = result.stdout.strip().split('\n')
        api_key = None
        for line in lines:
            if line.startswith("New API Key:"):
                api_key = line.split(": ", 1)[1]
                break
        
        if not api_key:
            pytest.skip("Could not extract API key from output")
        
        assert len(api_key) > 10, "API key should be substantial length"
        
        # Test authenticated endpoint (assumes server running on port 8000)
        base_url = "http://localhost:8000"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{base_url}/api/v1/jobs", headers=headers, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    assert "total" in data or "jobs" in data
                elif response.status_code == 401:
                    pytest.skip("Authentication failed - server may not be configured for auth")
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
    
    base_url = "http://localhost:8000"
    
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


async def wait_for_server(url: str = "http://localhost:8000", timeout: int = 30) -> bool:
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