#!/usr/bin/env python3
"""
Integration test execution script for VideoAnnotator.
Runs cross-component tests with moderate execution time.
Target: <5 minutes execution time.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_integration_tests():
    """Run the integration test tier."""
    print("🔗 Running VideoAnnotator Integration Test Suite")
    print("=" * 50)

    start_time = time.time()

    # Test command for integration tests
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/",
        "tests/integration/",
        "-v",
        "--maxfail=10",  # Allow more failures for complex tests
        "--tb=short",
        "--timeout=300",  # 5 minute timeout per test
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    duration = time.time() - start_time

    print()
    print("=" * 50)
    if result.returncode == 0:
        print(f"✅ Integration tests PASSED in {duration:.1f} seconds")
    else:
        print(f"❌ Integration tests FAILED in {duration:.1f} seconds")
        print(f"Exit code: {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_integration_tests())
