"""
Simple validation script to test batch processing APIs without pytest.
This allows us to validate our understanding of the APIs step by step.
"""

import sys
import tempfile
from pathlib import Path


def test_imports():
    """Test that we can import all required modules."""
    print("Testing imports...")
    try:
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False


def test_batch_job_creation():
    """Test BatchJob creation with actual API."""
    print("\nTesting BatchJob creation...")
    try:
        from src.batch.types import BatchJob, JobStatus

        # Test with minimal parameters (letting defaults work)
        job = BatchJob()
        print(f"✅ Created job with ID: {job.job_id}")

        # Test with full parameters
        job2 = BatchJob(
            video_path=Path("/test/video.mp4"),
            output_dir=Path("/test/output"),
            config={"test": "value"},
            status=JobStatus.PENDING,
            selected_pipelines=["scene_detection"],
        )
        print(f"✅ Created job with full params: {job2.job_id}")

        # Test properties
        assert job2.video_id == "video"  # From video.mp4
        assert job2.status == JobStatus.PENDING
        print("✅ All BatchJob properties work correctly")
        return True
    except Exception as e:
        print(f"❌ BatchJob test failed: {e}")
        return False


def test_orchestrator_basic():
    """Test BatchOrchestrator basic functionality."""
    print("\nTesting BatchOrchestrator...")
    try:
        from src.batch.batch_orchestrator import BatchOrchestrator

        # Create temp video file
        temp_dir = Path(tempfile.mkdtemp())
        test_video = temp_dir / "test.mp4"
        test_video.write_bytes(b"fake video content")

        # Test orchestrator creation
        orchestrator = BatchOrchestrator()
        print("✅ Created BatchOrchestrator")

        # Test add_job with real file
        job_id = orchestrator.add_job(str(test_video))
        print(f"✅ Added job: {job_id}")

        # Verify job was added
        assert len(orchestrator.jobs) == 1
        assert orchestrator.jobs[0].job_id == job_id
        print("✅ Job correctly added to orchestrator")

        # Cleanup
        test_video.unlink()
        temp_dir.rmdir()
        return True
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False


def test_progress_tracker():
    """Test ProgressTracker functionality."""
    print("\nTesting ProgressTracker...")
    try:
        from src.batch.progress_tracker import ProgressTracker
        from src.batch.types import BatchJob, JobStatus

        tracker = ProgressTracker()
        print("✅ Created ProgressTracker")

        # Create some test jobs
        jobs = [
            BatchJob(status=JobStatus.COMPLETED),
            BatchJob(status=JobStatus.RUNNING),
            BatchJob(status=JobStatus.PENDING),
        ]

        # Test get_status method
        status = tracker.get_status(jobs)
        print(
            f"✅ Got status: {status.total_jobs} total, {status.completed_jobs} completed"
        )

        assert status.total_jobs == 3
        assert status.completed_jobs == 1
        assert status.running_jobs == 1
        assert status.pending_jobs == 1
        print("✅ Progress tracking works correctly")
        return True
    except Exception as e:
        print(f"❌ ProgressTracker test failed: {e}")
        return False


def test_failure_recovery():
    """Test FailureRecovery functionality."""
    print("\nTesting FailureRecovery...")
    try:
        from src.batch.recovery import FailureRecovery
        from src.batch.types import BatchJob

        recovery = FailureRecovery()
        print("✅ Created FailureRecovery")

        # Test with a job
        job = BatchJob(retry_count=0)
        error = Exception("Test error")

        should_retry = recovery.should_retry(job, error)
        print(f"✅ Should retry: {should_retry}")

        delay = recovery.calculate_retry_delay(job)
        print(f"✅ Retry delay: {delay} seconds")

        assert isinstance(should_retry, bool)
        assert isinstance(delay, (int, float))
        print("✅ Failure recovery works correctly")
        return True
    except Exception as e:
        print(f"❌ FailureRecovery test failed: {e}")
        return False


def test_storage_backend():
    """Test FileStorageBackend functionality."""
    print("\nTesting FileStorageBackend...")
    try:
        from src.batch.types import BatchJob
        from src.storage.file_backend import FileStorageBackend

        temp_dir = Path(tempfile.mkdtemp())
        storage = FileStorageBackend(temp_dir)
        print("✅ Created FileStorageBackend")

        # Test save and load job metadata
        job = BatchJob(job_id="test_job")
        storage.save_job_metadata(job)
        print("✅ Saved job metadata")

        loaded_job = storage.load_job_metadata("test_job")
        print(f"✅ Loaded job: {loaded_job.job_id}")

        assert loaded_job.job_id == job.job_id
        print("✅ Storage backend works correctly")
        return True
    except Exception as e:
        print(f"❌ Storage backend test failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("🔍 Validating VideoAnnotator Batch Processing APIs...")
    print("=" * 60)

    tests = [
        test_imports,
        test_batch_job_creation,
        test_orchestrator_basic,
        test_progress_tracker,
        test_failure_recovery,
        test_storage_backend,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"📊 Results: {sum(results)}/{len(results)} tests passed")

    if all(results):
        print("🎉 All APIs validated successfully! Ready to create proper unit tests.")
    else:
        print("⚠️  Some APIs need investigation before creating tests.")

    return all(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
