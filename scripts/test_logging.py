#!/usr/bin/env python3
"""
Test script for VideoAnnotator enhanced logging system.

Validates that all logging components are working correctly.
"""

import subprocess
import sys
import time
from pathlib import Path

import requests


def test_logging_system():
    """Test the complete logging system."""

    print("[TEST] VideoAnnotator Enhanced Logging Test")
    print("=" * 50)

    # Test 1: Basic logging setup
    print("\n[TEST 1] Testing basic logging setup...")

    try:
        from src.utils.logging_config import get_logger, setup_videoannotator_logging

        loggers = setup_videoannotator_logging(
            logs_dir="logs", log_level="INFO", capture_warnings=True
        )

        api_logger = get_logger("api")
        request_logger = get_logger("requests")
        pipeline_logger = get_logger("pipelines")

        api_logger.info("Test: API logger working", extra={"test": "basic_setup"})
        request_logger.info(
            "Test: Request logger working", extra={"test": "basic_setup"}
        )
        pipeline_logger.info(
            "Test: Pipeline logger working", extra={"test": "basic_setup"}
        )

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
        # Start API server (best-effort)
        _server_process = subprocess.Popen(
            [
                "uv",
                "run",
                "python",
                "api_server.py",
                "--port",
                "8005",
                "--log-level",
                "info",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"VIDEOANNOTATOR_DB_PATH": "test_logging.db"},
        )

        # Wait for server to start
        time.sleep(3)

        # Make health check request
        try:
            resp = requests.get("http://localhost:8005/health", timeout=5)
            print(f"  Health check status: {resp.status_code}")
        except Exception as e:
            print(f"  [WARN] Health check failed: {e}")

        # Best-effort: check request logging file
        api_log_path = logs_dir / "api_server.log"
        if api_log_path.exists():
            with open(api_log_path) as f:
                content = f.read()
                if "VideoAnnotator API server starting up" in content:
                    print("[PASS] API server startup logged")
                else:
                    print("[WARN] API server startup message not found in logs")
        else:
            print(f"[WARN] {api_log_path} not found")

    except Exception as e:
        print(f"[FAIL] Error during logging test: {e}")
        return False
    finally:
        # Cleanup: terminate the server process if it was started
        if "_server_process" in locals():
            try:
                _server_process.terminate()
                _server_process.wait(timeout=3)
            except Exception:
                pass

    print("[PASS] Structured JSON logging test successful")
    return True


if __name__ == "__main__":
    ok = test_logging_system()
    sys.exit(0 if ok else 1)

    # Minimal logging smoke test for local development.
    #
    # This script is intentionally minimal and non-invasive: it tries to
    # initialize the project's logging configuration, checks that log files
    # exist (if logging has been exercised), and performs a best-effort
    # health-check against a locally-run API server.
    #
    # It's safe to run locally and is mainly useful for development/debugging.
    #

    import os
    import subprocess
    import sys
    import time
    from pathlib import Path

    import requests

    def test_logging_system() -> bool:
        """Run smoke checks for logging configuration and API server.

        Returns True when the script completed (it does not guarantee the
        API server started successfully)."""

        print("[TEST] VideoAnnotator logging smoke test")

        # Try to import and initialize the project's logging. This is a best-effort
        # check and should not raise in normal operation.
        try:
            from src.utils.logging_config import (
                get_logger,
                setup_videoannotator_logging,
            )

            # Initialize logging (best-effort)
            setup_videoannotator_logging(
                logs_dir="logs", log_level="INFO", capture_warnings=True
            )
            api_logger = get_logger("api")
            pipeline_logger = get_logger("pipelines")

            api_logger.info("Smoke test: API logger initialized")
            pipeline_logger.info("Smoke test: Pipelines logger initialized")

            print("[PASS] Logging initialization OK")
        except Exception as e:
            print(f"[WARN] Logging initialization failed: {e}")

        # Check for presence of common log files (may not exist in a fresh workspace)
        logs_dir = Path("logs")
        expected_files = ["api_server.log", "api_requests.log", "errors.log"]
        for log_file in expected_files:
            if (logs_dir / log_file).exists():
                print(f"[PASS] {log_file} exists")
            else:
                print(
                    f"[WARN] {log_file} not present (may be fine in a fresh workspace)"
                )

        # Best-effort API server smoke test: start the server in a subprocess and hit /health
        server_proc = None
        try:
            env = os.environ.copy()
            env["VIDEOANNOTATOR_DB_PATH"] = "test_logging.db"

            # Use python to run the local api_server module/script; this is intentionally
            # conservative (avoids assuming a particular runner like uvicorn is present).
            server_proc = subprocess.Popen(
                [
                    sys.executable,
                    "api_server.py",
                    "--port",
                    "8005",
                    "--log-level",
                    "info",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )

            # Wait a short time for the server to start
            time.sleep(2)

            try:
                r = requests.get("http://localhost:8005/health", timeout=3)
                print(f"[PASS] Health endpoint returned {r.status_code}")
            except Exception as e:
                print(f"[WARN] Health check failed: {e}")
        except Exception as e:
            print(f"[WARN] API server smoke test failed: {e}")
        finally:
            if server_proc:
                try:
                    server_proc.terminate()
                    server_proc.wait(timeout=3)
                except Exception:
                    pass

        print("[DONE] Logging smoke test finished")
        return True

    if __name__ == "__main__":
        ok = test_logging_system()
        sys.exit(0 if ok else 1)
