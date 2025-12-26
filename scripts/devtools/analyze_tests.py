#!/usr/bin/env python3
"""Analyze test collection and distribution.

Run:
  uv run python scripts/devtools/analyze_tests.py
"""

from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def analyze_test_collection():
    """Analyze what tests are collected and their distribution."""
    print("Analyzing test suite structure...")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--co", "-q"],
        capture_output=True,
        text=True,
        cwd=_repo_root(),
    )

    if result.returncode != 0:
        print("[ERROR] Error collecting tests:")
        print(result.stderr)
        return {}, {}

    lines = result.stdout.splitlines()
    total_tests = len([line for line in lines if "::" in line])

    print(f"Total tests collected: {total_tests}")

    file_counts = defaultdict(int)
    category_counts = defaultdict(int)

    for line in lines:
        if "::" not in line:
            continue
        file_part = line.split("::")[0]
        file_counts[file_part] += 1

        if "batch" in file_part:
            category_counts["Batch Processing"] += 1
        elif any(
            x in file_part for x in ["pipeline", "face", "person", "audio", "scene"]
        ):
            category_counts["Pipeline Tests"] += 1
        elif "integration" in file_part:
            category_counts["Integration Tests"] += 1
        elif "storage" in file_part:
            category_counts["Storage Tests"] += 1
        elif "laion" in file_part:
            category_counts["LAION Tests"] += 1
        else:
            category_counts["Other/Misc"] += 1

    print("\nTests by file:")
    for file, count in sorted(file_counts.items()):
        print(f"  {file}: {count} tests")

    print("\nTests by category:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} tests")

    return file_counts, category_counts


def analyze_test_patterns():
    """Analyze test naming patterns and types."""
    print("\nAnalyzing test patterns...")

    test_files = list((_repo_root() / "tests").rglob("test_*.py"))

    patterns = {
        "Unit Tests": 0,
        "Integration Tests": 0,
        "Modern Pipeline Tests": 0,
        "Legacy Tests": 0,
        "Real/Live Tests": 0,
        "Mock/Placeholder Tests": 0,
    }

    for test_file in test_files:
        name = test_file.name
        if "integration" in name:
            patterns["Integration Tests"] += 1
        elif "modern" in name:
            patterns["Modern Pipeline Tests"] += 1
        elif "real" in name:
            patterns["Real/Live Tests"] += 1
        elif any(x in name for x in ["batch", "storage", "recovery", "types"]):
            patterns["Unit Tests"] += 1
        else:
            patterns["Legacy Tests"] += 1

    print("Test file patterns:")
    for pattern, count in patterns.items():
        print(f"  {pattern}: {count} files")

    return patterns


def propose_organization():
    """Propose a test organization structure."""
    print("\nProposed test organization structure:")

    structure = {
        "tests/unit/": [
            "Core data structures (BatchJob, types, etc.)",
            "Individual pipeline components",
            "Utility functions",
            "Configuration validation",
        ],
        "tests/integration/": [
            "Pipeline-to-pipeline interactions",
            "Storage system integration",
            "Batch orchestrator workflows",
        ],
        "tests/pipelines/": [
            "Full pipeline processing tests",
            "Performance benchmarks",
            "Real video processing tests",
        ],
        "tests/experimental/": [
            "Placeholder tests for future features",
            "Research/prototype testing",
            "Incomplete implementations",
        ],
        "tests/fixtures/": [
            "Test data and mock objects",
            "Shared test utilities",
            "Test video samples",
        ],
    }

    for directory, contents in structure.items():
        print(f"\n{directory}")
        for item in contents:
            print(f"  - {item}")


if __name__ == "__main__":
    print("=" * 60)
    print("VideoAnnotator test suite analysis")
    print("=" * 60)

    try:
        file_counts, _ = analyze_test_collection()
        analyze_test_patterns()
        propose_organization()

        print("\nAnalysis complete")
        print(
            f"Next steps: Reorganize {sum(file_counts.values())} tests into a logical structure"
        )
    except Exception as e:
        print(f"[ERROR] Error during analysis: {e}")
        raise
