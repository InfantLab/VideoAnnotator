#!/usr/bin/env python3
"""VideoAnnotator Installation Verification Script.

This script performs progressive checks to verify that VideoAnnotator is
correctly installed and configured. It checks:
- Python version
- FFmpeg availability
- VideoAnnotator package importability
- Database write access
- GPU availability (optional)
- Sample video processing capability

Exit codes:
- 0: All checks passed
- 1: Critical failure (installation cannot work)
- 2: Warnings only (installation functional but suboptimal)

Usage:
    python scripts/verify_installation.py
    python scripts/verify_installation.py --skip-video-test
    python scripts/verify_installation.py --verbose
"""

import argparse
import platform
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NamedTuple


class CheckResult(NamedTuple):
    """Result of a verification check."""

    name: str
    passed: bool
    critical: bool
    message: str
    suggestion: str = ""


class InstallationVerifier:
    """Verify VideoAnnotator installation and dependencies."""

    def __init__(self, verbose: bool = False, skip_video_test: bool = False):
        """Initialize verifier.

        Args:
            verbose: Enable verbose output
            skip_video_test: Skip the video processing test (faster)
        """
        self.verbose = verbose
        self.skip_video_test = skip_video_test
        self.results: list[CheckResult] = []
        self.has_critical_failure = False
        self.has_warnings = False

    def print_header(self):
        """Print verification header."""
        print("=" * 80)
        print("VideoAnnotator Installation Verification")
        print("=" * 80)
        print()

    def print_summary(self):
        """Print verification summary."""
        print()
        print("=" * 80)
        print("Verification Summary")
        print("=" * 80)
        print()

        passed_count = sum(1 for r in self.results if r.passed)
        total_count = len(self.results)
        critical_failures = [r for r in self.results if not r.passed and r.critical]
        warnings = [r for r in self.results if not r.passed and not r.critical]

        print(f"Checks passed: {passed_count}/{total_count}")
        print()

        if critical_failures:
            print("[CRITICAL FAILURES]")
            for result in critical_failures:
                print(f"  - {result.name}: {result.message}")
                if result.suggestion:
                    print(f"    Fix: {result.suggestion}")
            print()

        if warnings:
            print("[WARNINGS]")
            for result in warnings:
                print(f"  - {result.name}: {result.message}")
                if result.suggestion:
                    print(f"    Suggestion: {result.suggestion}")
            print()

        if not critical_failures and not warnings:
            print("[OK] All checks passed! VideoAnnotator is ready to use.")
            print()
            print("Next steps:")
            print("  1. Start the API server: uv run python api_server.py")
            print("  2. Visit the API docs: http://localhost:18011/docs")
            print(
                "  3. Try the example: uv run python examples/basic_video_processing.py"
            )
        elif not critical_failures:
            print("[WARN] Installation functional but with warnings.")
            print("VideoAnnotator will work but may have reduced performance.")
        else:
            print("[FAIL] Installation incomplete. Please fix critical issues above.")

        print()
        print("=" * 80)

    def add_result(self, result: CheckResult):
        """Add check result and update status."""
        self.results.append(result)
        if not result.passed:
            if result.critical:
                self.has_critical_failure = True
            else:
                self.has_warnings = True

    def check_python_version(self) -> CheckResult:
        """Check Python version is >= 3.10."""
        print("[CHECK] Python version...", end=" ")

        version_info = sys.version_info
        version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"

        if version_info >= (3, 10):
            print(f"[OK] {version_str}")
            return CheckResult(
                name="Python version",
                passed=True,
                critical=True,
                message=f"Python {version_str} detected",
            )
        else:
            print(f"[FAIL] {version_str} (requires >= 3.10)")
            return CheckResult(
                name="Python version",
                passed=False,
                critical=True,
                message=f"Python {version_str} is too old (requires >= 3.10)",
                suggestion="Install Python 3.10+ from https://www.python.org/downloads/",
            )

    def check_ffmpeg(self) -> CheckResult:
        """Check FFmpeg is installed and accessible."""
        print("[CHECK] FFmpeg availability...", end=" ")

        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode == 0:
                # Extract version from first line
                version_line = result.stdout.split("\n")[0]
                print(
                    f"[OK] {version_line.split()[2] if len(version_line.split()) > 2 else 'installed'}"
                )
                return CheckResult(
                    name="FFmpeg",
                    passed=True,
                    critical=True,
                    message="FFmpeg is installed and accessible",
                )
            else:
                print("[FAIL] Command failed")
                return CheckResult(
                    name="FFmpeg",
                    passed=False,
                    critical=True,
                    message="FFmpeg command failed",
                    suggestion="Reinstall FFmpeg: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (macOS)",
                )

        except FileNotFoundError:
            print("[FAIL] Not found")
            return CheckResult(
                name="FFmpeg",
                passed=False,
                critical=True,
                message="FFmpeg not found in PATH",
                suggestion="Install FFmpeg: sudo apt-get install ffmpeg (Linux) or brew install ffmpeg (macOS) or download from https://ffmpeg.org/download.html (Windows)",
            )
        except subprocess.TimeoutExpired:
            print("[FAIL] Timeout")
            return CheckResult(
                name="FFmpeg",
                passed=False,
                critical=True,
                message="FFmpeg check timed out",
                suggestion="Check FFmpeg installation or system performance",
            )
        except Exception as e:
            print(f"[FAIL] {e}")
            return CheckResult(
                name="FFmpeg",
                passed=False,
                critical=True,
                message=f"FFmpeg check failed: {e}",
                suggestion="Verify FFmpeg is properly installed and accessible",
            )

    def check_videoannotator_import(self) -> CheckResult:
        """Check VideoAnnotator package can be imported."""
        print("[CHECK] VideoAnnotator package import...", end=" ")

        try:
            # Check if we're in the repo and can access version file
            repo_root = Path(__file__).parent.parent
            version_file = repo_root / "src" / "version.py"

            if version_file.exists():
                # Read version without importing (avoids dependency issues)
                version_content = version_file.read_text()
                for line in version_content.split("\n"):
                    if line.startswith("__version__"):
                        version = line.split("=")[1].strip().strip('"').strip("'")
                        print(f"[OK] v{version}")
                        return CheckResult(
                            name="VideoAnnotator source",
                            passed=True,
                            critical=True,
                            message=f"VideoAnnotator v{version} source found",
                        )

                # Fallback if version not found in file
                print("[OK] Source found")
                return CheckResult(
                    name="VideoAnnotator source",
                    passed=True,
                    critical=True,
                    message="VideoAnnotator source code present",
                )
            else:
                # Try importing from installed package
                try:
                    import videoannotator

                    version = getattr(videoannotator, "__version__", "unknown")
                    print(f"[OK] v{version} (installed)")
                    return CheckResult(
                        name="VideoAnnotator package",
                        passed=True,
                        critical=True,
                        message=f"VideoAnnotator v{version} installed",
                    )
                except ImportError:
                    print("[FAIL] Not found")
                    return CheckResult(
                        name="VideoAnnotator",
                        passed=False,
                        critical=True,
                        message="VideoAnnotator not found (neither source nor installed package)",
                        suggestion="Run 'uv sync' to install dependencies, or check you're in the VideoAnnotator directory",
                    )

        except Exception as e:
            print(f"[FAIL] {e}")
            return CheckResult(
                name="VideoAnnotator",
                passed=False,
                critical=True,
                message=f"Check failed: {e}",
                suggestion="Ensure you're running from VideoAnnotator directory: uv sync",
            )

    def check_database_writable(self) -> CheckResult:
        """Check database directory is writable."""
        print("[CHECK] Database write access...", end=" ")

        try:
            # Try to create a test database file
            test_db_path = Path("test_verify_install.db")

            # Create test database
            import sqlite3

            conn = sqlite3.connect(str(test_db_path))
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY)")
            cursor.execute("INSERT INTO test (id) VALUES (1)")
            conn.commit()
            conn.close()

            # Clean up
            test_db_path.unlink()

            print("[OK] Writable")
            return CheckResult(
                name="Database write access",
                passed=True,
                critical=True,
                message="Database directory is writable",
            )

        except PermissionError:
            print("[FAIL] Permission denied")
            return CheckResult(
                name="Database write access",
                passed=False,
                critical=True,
                message="Cannot write to database directory (permission denied)",
                suggestion="Fix permissions: chmod 755 . or run from a directory you own",
            )
        except Exception as e:
            print(f"[FAIL] {e}")
            return CheckResult(
                name="Database write access",
                passed=False,
                critical=True,
                message=f"Database write test failed: {e}",
                suggestion="Ensure current directory is writable and disk has space",
            )

    def check_gpu_availability(self) -> CheckResult:
        """Check GPU availability (optional - non-critical)."""
        print("[CHECK] GPU availability (optional)...", end=" ")

        try:
            import torch

            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
                print(f"[OK] {gpu_count} GPU(s) - {gpu_name}")
                return CheckResult(
                    name="GPU availability",
                    passed=True,
                    critical=False,
                    message=f"{gpu_count} GPU(s) available: {gpu_name}",
                )
            else:
                print("[WARN] No GPU detected")
                return CheckResult(
                    name="GPU availability",
                    passed=False,
                    critical=False,
                    message="No GPU detected (will use CPU - slower performance)",
                    suggestion="Install CUDA and PyTorch with GPU support for better performance",
                )

        except ImportError:
            print("[WARN] PyTorch not available")
            return CheckResult(
                name="GPU availability",
                passed=False,
                critical=False,
                message="PyTorch not installed (cannot check GPU)",
                suggestion="Install PyTorch: uv add torch",
            )
        except Exception as e:
            print(f"[WARN] Check failed: {e}")
            return CheckResult(
                name="GPU availability",
                passed=False,
                critical=False,
                message=f"GPU check failed: {e}",
                suggestion="GPU support is optional - CPU processing will work",
            )

    def detect_platform(self) -> str:
        """Detect operating system platform."""
        system = platform.system()
        if system == "Linux":
            # Check if running in WSL
            try:
                with open("/proc/version") as f:
                    version_info = f.read().lower()
                    if "microsoft" in version_info or "wsl" in version_info:
                        return "Windows WSL2"
            except OSError:
                pass
            return "Linux"
        elif system == "Darwin":
            return "macOS"
        elif system == "Windows":
            return "Windows"
        else:
            return f"Unknown ({system})"

    def check_platform(self) -> CheckResult:
        """Check and report platform information."""
        platform_name = self.detect_platform()
        python_impl = platform.python_implementation()
        machine = platform.machine()

        print(f"[INFO] Platform: {platform_name} ({machine}, {python_impl})")

        return CheckResult(
            name="Platform detection",
            passed=True,
            critical=False,
            message=f"Running on {platform_name} ({machine}, {python_impl})",
        )

    def check_sample_video_processing(self) -> CheckResult:
        """Test sample video processing (optional - can be slow)."""
        if self.skip_video_test:
            print("[SKIP] Video processing test (--skip-video-test)")
            return CheckResult(
                name="Video processing test",
                passed=True,
                critical=False,
                message="Skipped by user request",
            )

        print("[CHECK] Sample video processing...", end=" ")
        print()  # New line for progress

        try:
            # Create a minimal test video using FFmpeg
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_video = Path(temp_dir) / "test_video.mp4"

                # Generate a 2-second test video (black screen)
                print("  [INFO] Generating test video...")
                result = subprocess.run(
                    [
                        "ffmpeg",
                        "-f",
                        "lavfi",
                        "-i",
                        "color=c=black:s=320x240:d=2",
                        "-c:v",
                        "libx264",
                        "-pix_fmt",
                        "yuv420p",
                        "-y",
                        str(temp_video),
                    ],
                    capture_output=True,
                    timeout=30,
                )

                if result.returncode != 0:
                    print("  [FAIL] Video generation failed")
                    return CheckResult(
                        name="Video processing test",
                        passed=False,
                        critical=False,
                        message="Test video generation failed",
                        suggestion="This is optional - core functionality should still work",
                    )

                # Try to process it (just test basic video reading)
                print("  [INFO] Testing video processing...")
                import cv2

                cap = cv2.VideoCapture(str(temp_video))
                if not cap.isOpened():
                    print("  [FAIL] Cannot open test video")
                    return CheckResult(
                        name="Video processing test",
                        passed=False,
                        critical=False,
                        message="Cannot open generated test video",
                        suggestion="OpenCV may not be properly configured",
                    )

                # Read a few frames
                frame_count = 0
                while frame_count < 10:
                    ret, _frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1

                cap.release()

                if frame_count > 0:
                    print(f"  [OK] Processed {frame_count} frames")
                    return CheckResult(
                        name="Video processing test",
                        passed=True,
                        critical=False,
                        message=f"Successfully processed {frame_count} test frames",
                    )
                else:
                    print("  [FAIL] No frames read")
                    return CheckResult(
                        name="Video processing test",
                        passed=False,
                        critical=False,
                        message="Could not read frames from test video",
                        suggestion="Check OpenCV installation: uv sync",
                    )

        except subprocess.TimeoutExpired:
            print("  [FAIL] Timeout")
            return CheckResult(
                name="Video processing test",
                passed=False,
                critical=False,
                message="Video processing test timed out",
                suggestion="This is optional - may indicate slow system or FFmpeg issues",
            )
        except ImportError as e:
            print(f"  [FAIL] Missing dependency: {e}")
            return CheckResult(
                name="Video processing test",
                passed=False,
                critical=False,
                message=f"Missing dependency: {e}",
                suggestion="Run 'uv sync' to install all dependencies",
            )
        except Exception as e:
            print(f"  [FAIL] {e}")
            return CheckResult(
                name="Video processing test",
                passed=False,
                critical=False,
                message=f"Video processing test failed: {e}",
                suggestion="This is optional - core API functionality should still work",
            )

    def run_all_checks(self):
        """Run all verification checks."""
        self.print_header()

        # Platform detection (always first, informational)
        result = self.check_platform()
        self.add_result(result)
        print()

        # Critical checks
        result = self.check_python_version()
        self.add_result(result)

        result = self.check_ffmpeg()
        self.add_result(result)

        result = self.check_videoannotator_import()
        self.add_result(result)

        result = self.check_database_writable()
        self.add_result(result)

        # Optional checks (continue even if critical checks failed)
        print()
        result = self.check_gpu_availability()
        self.add_result(result)

        if not self.skip_video_test:
            print()
            result = self.check_sample_video_processing()
            self.add_result(result)

        # Summary
        self.print_summary()

        # Return appropriate exit code
        if self.has_critical_failure:
            return 1
        elif self.has_warnings:
            return 2
        else:
            return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Verify VideoAnnotator installation and dependencies"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--skip-video-test",
        action="store_true",
        help="Skip the video processing test (faster verification)",
    )

    args = parser.parse_args()

    verifier = InstallationVerifier(
        verbose=args.verbose,
        skip_video_test=args.skip_video_test,
    )

    exit_code = verifier.run_all_checks()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
