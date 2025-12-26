"""Validate batch processing APIs without pytest.

This script is intentionally lightweight: it exercises a few core types and
components to validate import paths and basic behaviors.

Run:
  uv run python scripts/validation/validate_apis.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path


def test_imports() -> bool:
    """Test that we can import required modules."""
    print("Testing imports...")
    try:
        import videoannotator  # noqa: F401

        print("[OK] All imports successful")
        return True
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False


def test_batch_job_creation() -> bool:
    """Test BatchJob creation."""
    print("\nTesting BatchJob creation...")
    try:
        from videoannotator.batch.types import BatchJob, JobStatus

        job = BatchJob()
        print(f"[OK] Created job with ID: {job.job_id}")

        job2 = BatchJob(
            video_path=Path("/test/video.mp4"),
            output_dir=Path("/test/output"),
            config={"test": "value"},
            status=JobStatus.PENDING,
            selected_pipelines=["scene_detection"],
        )
        print(f"[OK] Created job with full params: {job2.job_id}")

        assert job2.video_id == "video"  # From video.mp4
        assert job2.status == JobStatus.PENDING
        print("[OK] All BatchJob properties work correctly")
        return True
    except Exception as e:
        print(f"[ERROR] BatchJob test failed: {e}")
        return False


def test_orchestrator_basic() -> bool:
    """Test BatchOrchestrator basic functionality."""
    print("\nTesting BatchOrchestrator...")
    try:
        from videoannotator.batch.batch_orchestrator import BatchOrchestrator

        temp_dir = Path(tempfile.mkdtemp())
        test_video = temp_dir / "test.mp4"
        test_video.write_bytes(b"fake video content")

        orchestrator = BatchOrchestrator()
        print("[OK] Created BatchOrchestrator")

        job_id = orchestrator.add_job(str(test_video))
        print(f"[OK] Added job: {job_id}")

        assert len(orchestrator.jobs) == 1
        assert orchestrator.jobs[0].job_id == job_id
        print("[OK] Job correctly added to orchestrator")

        test_video.unlink(missing_ok=True)
        temp_dir.rmdir()
        return True
    except Exception as e:
        print(f"[ERROR] Orchestrator test failed: {e}")
        return False


def test_progress_tracker() -> bool:
    """Test ProgressTracker functionality."""
    print("\nTesting ProgressTracker...")
    try:
        from videoannotator.batch.progress_tracker import ProgressTracker
        from videoannotator.batch.types import BatchJob, JobStatus

        tracker = ProgressTracker()
        print("[OK] Created ProgressTracker")

        jobs = [
            BatchJob(status=JobStatus.COMPLETED),
            BatchJob(status=JobStatus.RUNNING),
            BatchJob(status=JobStatus.PENDING),
        ]

        status = tracker.get_status(jobs)
        print(
            f"[OK] Got status: {status.total_jobs} total, {status.completed_jobs} completed"
        )

        assert status.total_jobs == 3
        assert status.completed_jobs == 1
        assert status.running_jobs == 1
        assert status.pending_jobs == 1
        print("[OK] Progress tracking works correctly")
        return True
    except Exception as e:
        print(f"[ERROR] ProgressTracker test failed: {e}")
        return False


def test_failure_recovery() -> bool:
    """Test FailureRecovery functionality."""
    print("\nTesting FailureRecovery...")
    try:
        from videoannotator.batch.recovery import FailureRecovery
        from videoannotator.batch.types import BatchJob

        recovery = FailureRecovery()
        print("[OK] Created FailureRecovery")

        job = BatchJob(retry_count=0)
        error = Exception("Test error")

        should_retry = recovery.should_retry(job, error)
        print(f"[OK] Should retry: {should_retry}")

        delay = recovery.calculate_retry_delay(job)
        print(f"[OK] Retry delay: {delay} seconds")

        assert isinstance(should_retry, bool)
        assert isinstance(delay, (int, float))
        print("[OK] Failure recovery works correctly")
        return True
    except Exception as e:
        print(f"[ERROR] FailureRecovery test failed: {e}")
        return False


def test_storage_backend() -> bool:
    """Test FileStorageBackend functionality."""
    print("\nTesting FileStorageBackend...")
    try:
        from videoannotator.batch.types import BatchJob
        from videoannotator.storage.file_backend import FileStorageBackend

        temp_dir = Path(tempfile.mkdtemp())
        storage = FileStorageBackend(temp_dir)
        print("[OK] Created FileStorageBackend")

        job = BatchJob(job_id="test_job")
        storage.save_job_metadata(job)
        print("[OK] Saved job metadata")

        loaded_job = storage.load_job_metadata("test_job")
        print(f"[OK] Loaded job: {loaded_job.job_id}")

        assert loaded_job.job_id == job.job_id
        print("[OK] Storage backend works correctly")
        return True
    except Exception as e:
        print(f"[ERROR] Storage backend test failed: {e}")
        return False


def main() -> bool:
    """Run all validation tests."""
    print("Validating VideoAnnotator batch processing APIs...")
    print("=" * 60)

    tests = [
        test_imports,
        test_batch_job_creation,
        test_orchestrator_basic,
        test_progress_tracker,
        test_failure_recovery,
        test_storage_backend,
    ]

    results: list[bool] = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"[ERROR] Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("[OK] All APIs validated successfully")
    else:
        print("[WARNING] Some APIs need investigation before creating tests")

    return all(results)


if __name__ == "__main__":
    success = main()
    raise SystemExit(0 if success else 1)
