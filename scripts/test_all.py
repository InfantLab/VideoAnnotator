#!/usr/bin/env python3
"""
Complete test suite execution script for VideoAnnotator.
Runs all organized tests in sequence with comprehensive reporting.
"""

import subprocess
import sys
import time
from pathlib import Path


def run_test_tier(name, paths, extra_args=None):
    """Run a specific test tier and return results."""
    print(f"\n{'=' * 20} {name.upper()} TESTS {'=' * 20}")
    start_time = time.time()

    cmd = [sys.executable, "-m", "pytest"] + paths + ["-v", "--tb=short"]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    duration = time.time() - start_time

    status = "PASSED" if result.returncode == 0 else "FAILED"
    print(f"{name} tests: {status} in {duration:.1f}s")

    return {
        "name": name,
        "status": status,
        "duration": duration,
        "returncode": result.returncode,
    }


def run_all_tests():
    """Run all test tiers in sequence."""
    print("🧪 Running VideoAnnotator Complete Test Suite")
    print("=" * 60)

    total_start = time.time()
    results = []

    # Tier 1: Unit Tests (Fast)
    results.append(run_test_tier("Unit", ["tests/unit/"], ["--maxfail=5", "-x"]))

    # Tier 2: Integration Tests
    results.append(
        run_test_tier(
            "Integration", ["tests/integration/"], ["--maxfail=10", "--timeout=300"]
        )
    )

    # Tier 3: Pipeline Tests
    results.append(
        run_test_tier(
            "Pipeline", ["tests/pipelines/"], ["--maxfail=3", "--timeout=600"]
        )
    )

    # Summary Report
    total_duration = time.time() - total_start
    print(f"\n{'=' * 60}")
    print("📊 TEST SUITE SUMMARY")
    print(f"{'=' * 60}")

    passed = 0
    for result in results:
        status_emoji = "✅" if result["status"] == "PASSED" else "❌"
        print(
            f"{status_emoji} {result['name']:12} {result['status']:8} {result['duration']:6.1f}s"
        )
        if result["status"] == "PASSED":
            passed += 1

    print(f"\nTotal Results: {passed}/{len(results)} tiers passed")
    print(f"Total Duration: {total_duration:.1f} seconds")

    # Overall exit code
    overall_success = all(r["returncode"] == 0 for r in results)
    return 0 if overall_success else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
