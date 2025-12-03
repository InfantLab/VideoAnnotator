#!/usr/bin/env python3
"""
Fast test execution script for VideoAnnotator.
Runs unit tests and basic integration tests quickly for development workflow.
Target: <30 seconds execution time.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_fast_tests():
    """Run the fast test tier - unit tests and basic integration."""
    print("Running VideoAnnotator Fast Test Suite")
    print("=" * 50)

    start_time = time.time()

    # Test command for fast execution
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "tests/unit/",
        "-v",
        "--maxfail=5",  # Stop after 5 failures
        "--tb=short",  # Short traceback format
        "--quiet",  # Less verbose output
        "-x",  # Stop on first failure for speed
    ]

    print(f"Running: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    duration = time.time() - start_time

    print()
    print("=" * 50)
    if result.returncode == 0:
        print(f"PASSED: Fast tests completed in {duration:.1f} seconds")
    else:
        print(f"FAILED: Fast tests failed in {duration:.1f} seconds")
        print(f"Exit code: {result.returncode}")

    return result.returncode


if __name__ == "__main__":
    sys.exit(run_fast_tests())
