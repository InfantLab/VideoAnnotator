#!/usr/bin/env python3
"""Coverage validation script for VideoAnnotator.

This script runs pytest with coverage, validates against thresholds, and generates reports
for JOSS publication requirements.

Exit codes:
    0: All coverage thresholds met
    1: Coverage thresholds not met
    2: Test execution failed

Usage:
    python scripts/validate_coverage.py [options]

Options:
    --verbose, -v: Show detailed coverage output
    --html: Generate HTML coverage report (saved to htmlcov/)
    --xml: Generate XML coverage report (saved to coverage.xml)
    --fail-under PERCENT: Fail if total coverage is below this percentage (default: 80)
    --skip-tests: Skip running tests, just analyze existing .coverage file
    --module MODULE: Check coverage for specific module only

Examples:
    # Run with default settings
    python scripts/validate_coverage.py

    # Generate HTML report and use custom threshold
    python scripts/validate_coverage.py --html --fail-under 85

    # Check coverage for API module only
    python scripts/validate_coverage.py --module src/api

    # Verbose output with XML report for CI
    python scripts/validate_coverage.py -v --xml
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar


class CoverageValidator:
    """Validates test coverage against configured thresholds."""

    # Module-specific coverage thresholds (percentage)
    THRESHOLDS: ClassVar[dict[str, float]] = {
        "src/api": 90.0,  # API endpoints should have high coverage
        "src/pipelines": 80.0,  # Pipeline modules (ML code, may have some untestable paths)
        "src/database": 85.0,  # Database operations
        "src/storage": 85.0,  # Storage layer
        "src/registry": 90.0,  # Registry system
        "src/validation": 90.0,  # Validation logic
        "src/worker": 85.0,  # Worker execution
        "src/batch": 80.0,  # Batch processing
        "src/exporters": 75.0,  # Exporters (complex I/O patterns)
        "src/utils": 80.0,  # Utility functions
    }

    # Global minimum threshold
    GLOBAL_THRESHOLD: ClassVar[float] = 80.0

    def __init__(self, verbose: bool = False):
        """Initialize validator.

        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.workspace_root = Path(__file__).parent.parent

    def run_tests_with_coverage(
        self,
        html: bool = False,
        xml: bool = False,
        module: str | None = None,
    ) -> bool:
        """Run pytest with coverage collection.

        Args:
            html: Generate HTML coverage report
            xml: Generate XML coverage report
            module: Specific module to test (or None for all)

        Returns:
            True if tests passed, False otherwise
        """
        cmd = [
            "uv",
            "run",
            "pytest",
            "-q",  # Quiet test output
            "--cov=src",
            "--cov-report=term-missing",
        ]

        if html:
            cmd.append("--cov-report=html")

        if xml:
            cmd.append("--cov-report=xml")

        # Add module filter if specified
        if module and module.startswith("src/"):
            # Convert module path to test directory
            test_module = module.replace("src/", "tests/")
            if Path(self.workspace_root / test_module).exists():
                cmd.append(test_module)
            else:
                print(
                    f"[WARNING] Test directory {test_module} not found, running all tests"
                )

        if self.verbose:
            print(f"[START] Running tests with coverage: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                cwd=self.workspace_root,
                capture_output=not self.verbose,
                text=True,
            )

            if result.returncode != 0:
                print("[ERROR] Test execution failed")
                if not self.verbose and result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                return False

            if self.verbose and result.stdout:
                print(result.stdout)

            return True

        except Exception as e:
            print(f"[ERROR] Failed to run tests: {e}")
            return False

    def get_coverage_json(self) -> dict[str, Any] | None:
        """Get coverage data as JSON.

        Returns:
            Coverage data dictionary or None if failed
        """
        cmd = ["uv", "run", "coverage", "json", "-o", "-"]

        try:
            result = subprocess.run(
                cmd,
                cwd=self.workspace_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return json.loads(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to get coverage JSON: {e}")
            if e.stderr:
                print(e.stderr)
            return None
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to parse coverage JSON: {e}")
            return None

    def validate_coverage(self, fail_under: float | None = None) -> bool:
        """Validate coverage against thresholds.

        Args:
            fail_under: Override global threshold

        Returns:
            True if all thresholds met, False otherwise
        """
        coverage_data = self.get_coverage_json()
        if not coverage_data:
            return False

        total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0.0)
        files = coverage_data.get("files", {})

        # Use override or default
        global_threshold = (
            fail_under if fail_under is not None else self.GLOBAL_THRESHOLD
        )

        print("\n" + "=" * 70)
        print("COVERAGE VALIDATION REPORT")
        print("=" * 70)
        print(f"\n[OK] Total Coverage: {total_coverage:.2f}%")
        print(f"[OK] Global Threshold: {global_threshold:.2f}%")

        # Check global threshold
        passed = True
        if total_coverage < global_threshold:
            print(
                f"\n[FAIL] Total coverage {total_coverage:.2f}% is below threshold {global_threshold:.2f}%"
            )
            passed = False
        else:
            print("\n[OK] Total coverage meets global threshold")

        # Check module-specific thresholds
        print("\n" + "-" * 70)
        print("MODULE-SPECIFIC COVERAGE")
        print("-" * 70)

        module_results: dict[str, dict[str, float]] = {}

        for module_path, threshold in self.THRESHOLDS.items():
            # Calculate coverage for this module
            module_files = {
                file_path: data
                for file_path, data in files.items()
                if file_path.startswith(module_path)
            }

            if not module_files:
                print(f"\n[SKIP] {module_path}: No files found")
                continue

            # Calculate weighted average coverage for the module
            total_statements = sum(
                data["summary"]["num_statements"] for data in module_files.values()
            )
            covered_statements = sum(
                data["summary"]["covered_lines"] for data in module_files.values()
            )

            if total_statements == 0:
                module_coverage = 100.0
            else:
                module_coverage = (covered_statements / total_statements) * 100

            module_results[module_path] = {
                "coverage": module_coverage,
                "threshold": threshold,
                "passed": module_coverage >= threshold,
            }

            status = "[OK]" if module_coverage >= threshold else "[FAIL]"
            print(
                f"{status} {module_path}: {module_coverage:.2f}% (threshold: {threshold:.2f}%)"
            )

            if module_coverage < threshold:
                passed = False
                # Show files below threshold
                if self.verbose:
                    print(f"     Files below threshold in {module_path}:")
                    for file_path, data in module_files.items():
                        file_cov = data["summary"]["percent_covered"]
                        if file_cov < threshold:
                            print(f"       - {file_path}: {file_cov:.2f}%")

        # Summary
        print("\n" + "=" * 70)
        if passed:
            print("[OK] All coverage thresholds met!")
            print("=" * 70 + "\n")
            return True
        else:
            print("[FAIL] Some coverage thresholds not met")
            print("=" * 70)
            print("\n[SUGGESTION] To improve coverage:")
            print("  1. Run: uv run pytest --cov=src --cov-report=html")
            print("  2. Open: htmlcov/index.html")
            print("  3. Focus on modules/files with low coverage")
            print("  4. Add unit tests for uncovered code paths\n")
            return False

    def analyze_existing_coverage(self) -> bool:
        """Analyze existing .coverage file without running tests.

        Returns:
            True if coverage file exists and is valid
        """
        coverage_file = self.workspace_root / ".coverage"
        if not coverage_file.exists():
            print("[ERROR] No .coverage file found. Run tests with coverage first.")
            return False

        print("[OK] Found existing .coverage file")
        return True


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0=success, 1=coverage fail, 2=test fail)
    """
    docstring = __doc__ or ""
    epilog_text = docstring.split("Usage:")[1] if "Usage:" in docstring else ""

    parser = argparse.ArgumentParser(
        description="Validate test coverage for VideoAnnotator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog_text,
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show detailed coverage output",
    )

    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report (saved to htmlcov/)",
    )

    parser.add_argument(
        "--xml",
        action="store_true",
        help="Generate XML coverage report (saved to coverage.xml)",
    )

    parser.add_argument(
        "--fail-under",
        type=float,
        metavar="PERCENT",
        help="Fail if total coverage is below this percentage (default: 80)",
    )

    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running tests, just analyze existing .coverage file",
    )

    parser.add_argument(
        "--module",
        type=str,
        metavar="MODULE",
        help="Check coverage for specific module only (e.g., src/api)",
    )

    args = parser.parse_args()

    validator = CoverageValidator(verbose=args.verbose)

    print("[START] VideoAnnotator Coverage Validation")
    print("=" * 70 + "\n")

    # Run tests or analyze existing coverage
    if args.skip_tests:
        if not validator.analyze_existing_coverage():
            return 2
    else:
        if not validator.run_tests_with_coverage(
            html=args.html, xml=args.xml, module=args.module
        ):
            return 2

    # Validate coverage against thresholds
    if not validator.validate_coverage(fail_under=args.fail_under):
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
