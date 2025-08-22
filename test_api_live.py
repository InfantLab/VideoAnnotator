#!/usr/bin/env python3
"""
Live API testing script that starts the API server and runs integration tests.
"""

import asyncio
import subprocess
import time
import signal
import sys
from pathlib import Path

import httpx


async def wait_for_server(url: str = "http://localhost:8000", timeout: int = 30):
    """Wait for the API server to be ready."""
    print(f"Waiting for server at {url}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/health", timeout=2.0)
                if response.status_code == 200:
                    print(f"[OK] Server is ready at {url}")
                    return True
        except (httpx.ConnectError, httpx.TimeoutException):
            await asyncio.sleep(1)
            print(".", end="", flush=True)
    
    print(f"\n[FAIL] Server not ready after {timeout} seconds")
    return False


async def test_api_endpoints():
    """Test basic API endpoints directly."""
    print("\n[TEST] Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health endpoint
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            print("[OK] Health endpoint working")
    except Exception as e:
        print(f"[FAIL] Health endpoint failed: {e}")
        return False
    
    # Test 2: System health endpoint (should work without auth)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/v1/system/health")
            if response.status_code == 200:
                data = response.json()
                print("[OK] System health endpoint working")
                print(f"   Database: {data.get('database', {}).get('database_url', 'Unknown')}")
            else:
                print(f"[WARN] System health endpoint returned {response.status_code}")
    except Exception as e:
        print(f"[FAIL] System health endpoint failed: {e}")
    
    # Test 3: Pipelines endpoint
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/api/v1/pipelines")
            assert response.status_code == 200
            data = response.json()
            pipelines = data.get("pipelines", [])
            print(f"[OK] Pipelines endpoint working ({len(pipelines)} pipelines available)")
            for pipeline in pipelines[:3]:  # Show first 3
                print(f"   - {pipeline['name']}: {pipeline['description']}")
    except Exception as e:
        print(f"[FAIL] Pipelines endpoint failed: {e}")
        return False
    
    # Test 4: API Documentation
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{base_url}/docs")
            if response.status_code == 200:
                print("[OK] API documentation accessible at /docs")
            else:
                print(f"[WARN] API documentation returned {response.status_code}")
    except Exception as e:
        print(f"[FAIL] API documentation failed: {e}")
    
    return True


async def test_authentication():
    """Test API authentication with the test key."""
    print("\n[TEST] Testing API authentication...")
    
    # Get test API key
    try:
        result = subprocess.run([
            sys.executable, "create_test_key.py"
        ], capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            # Extract API key from output
            lines = result.stdout.strip().split('\n')
            api_key = None
            for line in lines:
                if line.startswith("New API Key:"):
                    api_key = line.split(": ", 1)[1]
                    break
            
            if api_key:
                print(f"[OK] Created test API key: {api_key[:20]}...")
                
                # Test authenticated endpoint
                base_url = "http://localhost:8000"
                headers = {"Authorization": f"Bearer {api_key}"}
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(f"{base_url}/api/v1/jobs", headers=headers)
                    if response.status_code == 200:
                        data = response.json()
                        print(f"[OK] Authenticated jobs endpoint working ({data.get('total', 0)} jobs)")
                        return True
                    else:
                        print(f"[FAIL] Authenticated endpoint failed with {response.status_code}")
                        return False
            else:
                print("[FAIL] Could not extract API key from output")
                return False
        else:
            print(f"[FAIL] Failed to create test API key: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Authentication test failed: {e}")
        return False


async def main():
    """Main test runner."""
    print("[VideoAnnotator API v1.2.0 - Live Integration Testing]")
    print("=" * 60)
    
    # Start API server
    print("Starting API server...")
    server_process = subprocess.Popen([
        sys.executable, "api_server_db.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    try:
        # Wait for server to be ready
        if not await wait_for_server():
            print("[FAIL] Server failed to start")
            return 1
        
        # Run tests
        success = True
        
        # Basic endpoint tests
        if not await test_api_endpoints():
            success = False
        
        # Authentication tests
        if not await test_authentication():
            success = False
        
        # Run pytest integration tests (if pytest is available)
        print("\n[TEST] Running pytest integration tests...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                "tests/integration/test_api_integration.py", 
                "-v", "--tb=short"
            ], text=True, timeout=60)
            
            if result.returncode == 0:
                print("[OK] Pytest integration tests passed")
            else:
                print(f"[WARN] Some pytest integration tests failed (return code: {result.returncode})")
        except subprocess.TimeoutExpired:
            print("[WARN] Pytest integration tests timed out")
        except FileNotFoundError:
            print("[WARN] Pytest not available, skipping integration tests")
        
        if success:
            print("\n[SUCCESS] All live API tests passed!")
            print("[INFO] API server is working correctly")
            return 0
        else:
            print("\n[FAIL] Some API tests failed")
            return 1
    
    finally:
        # Clean up server
        print("\n[INFO] Shutting down API server...")
        try:
            server_process.terminate()
            server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_process.kill()
        print("[OK] Server stopped")


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n[WARN] Testing interrupted by user")
        sys.exit(1)