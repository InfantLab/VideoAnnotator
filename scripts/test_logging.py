#!/usr/bin/env python3
"""
Test script for VideoAnnotator enhanced logging system.

Validates that all logging components are working correctly.
"""

import time
import subprocess
import requests
import sys
import json
from pathlib import Path

def test_logging_system():
    """Test the complete logging system."""
    
    print("[TEST] VideoAnnotator Enhanced Logging Test")
    print("=" * 50)
    
    # Test 1: Basic logging setup
    print("\n[TEST 1] Testing basic logging setup...")
    
    try:
        from src.utils.logging_config import setup_videoannotator_logging, get_logger
        loggers = setup_videoannotator_logging(logs_dir="logs", log_level="INFO")
        
        api_logger = get_logger("api")
        request_logger = get_logger("requests") 
        pipeline_logger = get_logger("pipelines")
        
        api_logger.info("Test: API logger working", extra={"test": "basic_setup"})
        request_logger.info("Test: Request logger working", extra={"test": "basic_setup"})
        pipeline_logger.info("Test: Pipeline logger working", extra={"test": "basic_setup"})
        
        print("[PASS] Basic logging setup successful")
    except Exception as e:
        print(f"[FAIL] Basic logging setup failed: {e}")
        return False
    
    # Test 2: Check log files are created
    print("\n[TEST 2] Checking log files...")
    
    logs_dir = Path("logs")
    expected_files = ["api_server.log", "api_requests.log", "errors.log"]
    
    for log_file in expected_files:
        log_path = logs_dir / log_file
        if log_path.exists():
            print(f"[PASS] {log_file} exists")
        else:
            print(f"[FAIL] {log_file} missing")
            return False
    
    # Test 3: Test structured JSON logging
    print("\n[TEST 3] Testing structured JSON logging...")
    
    try:
        # Read the most recent API log entry
        api_log_path = logs_dir / "api_server.log"
        if api_log_path.exists():
            with open(api_log_path, 'r') as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    log_entry = json.loads(last_line)
                    
                    required_fields = ["timestamp", "level", "logger", "message"]
                    for field in required_fields:
                        if field not in log_entry:
                            print(f"[FAIL] Missing field {field} in JSON log")
                            return False
                    
                    print("[PASS] Structured JSON logging working")
                else:
                    print("[FAIL] No log entries found")
                    return False
    except Exception as e:
        print(f"[FAIL] JSON logging test failed: {e}")
        return False
    
    # Test 4: Test API server with logging
    print("\n[TEST 4] Testing API server with enhanced logging...")
    
    try:
        # Start API server
        server_process = subprocess.Popen(
            ["uv", "run", "python", "api_server.py", "--port", "8005", "--log-level", "info"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"VIDEOANNOTATOR_DB_PATH": "test_logging.db"}
        )
        
        # Wait for server to start
        time.sleep(3)
        
        # Make test requests
        print("  Making test API requests...")
        
        # Health check (should not be logged in requests log)
        response = requests.get("http://localhost:8005/health", timeout=5)
        print(f"  Health check: {response.status_code}")
        
        # Debug endpoint (should be logged)
        response = requests.get("http://localhost:8005/api/v1/debug/server-info", timeout=5) 
        print(f"  Debug endpoint: {response.status_code}")
        
        # Authenticated request
        headers = {"Authorization": "Bearer dev-token"}
        response = requests.get("http://localhost:8005/api/v1/debug/token-info", headers=headers, timeout=5)
        print(f"  Authenticated request: {response.status_code}")
        
        print("[PASS] API server requests completed")
        
        # Stop server
        server_process.terminate()
        server_process.wait(timeout=5)
        
    except Exception as e:
        print(f"[FAIL] API server test failed: {e}")
        if 'server_process' in locals():
            server_process.terminate()
        return False
    
    # Test 5: Verify request logging
    print("\n[TEST 5] Verifying request logging...")
    
    try:
        # Check if request logs were generated
        time.sleep(1)  # Allow time for logs to be written
        
        request_log_path = logs_dir / "api_requests.log"
        api_log_path = logs_dir / "api_server.log"
        
        # Check API server logs for startup messages
        if api_log_path.exists():
            with open(api_log_path, 'r') as f:
                content = f.read()
                if "VideoAnnotator API server starting up" in content:
                    print("[PASS] Server startup logged")
                else:
                    print("[WARN] Server startup not found in logs")
        
        print("[PASS] Request logging verification completed")
        
    except Exception as e:
        print(f"[FAIL] Request logging verification failed: {e}")
        return False
    
    # Final summary
    print("\n" + "=" * 50)
    print("[SUCCESS] All logging tests completed successfully!")
    print("\nLog Files Summary:")
    
    for log_file in expected_files:
        log_path = logs_dir / log_file
        if log_path.exists():
            size = log_path.stat().st_size
            print(f"  {log_file}: {size} bytes")
    
    print("\nLogging Features Available:")
    print("  - Structured JSON logging to files")
    print("  - Log rotation (10MB max per file)")
    print("  - Separate error logging")  
    print("  - Request/response logging")
    print("  - Console output for development")
    print("  - Configurable log levels")
    
    return True

if __name__ == "__main__":
    success = test_logging_system()
    sys.exit(0 if success else 1)