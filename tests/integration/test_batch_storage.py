"""Unit tests for batch processing storage backend functionality.

Tests the FileStorageBackend class that handles job persistence, state
management, and data serialization for batch processing.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from videoannotator.batch.types import BatchJob, BatchReport, JobStatus
from videoannotator.storage.file_backend import FileStorageBackend


class TestFileStorageBackend:
    """Test FileStorageBackend functionality."""

    def setup_method(self):
        """Set up test fixtures with temporary directory."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = FileStorageBackend(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_storage_initialization(self):
        """Test storage backend initialization."""
        assert self.storage.base_dir == self.temp_dir
        assert self.storage.base_dir.exists()

        # Verify subdirectories are created
        assert (self.storage.base_dir / "jobs").exists()
        # Note: FileStorageBackend only creates jobs directory, not reports/checkpoints

    def test_custom_initialization(self):
        """Test storage with custom base directory."""
        custom_dir = self.temp_dir / "custom_storage"
        storage = FileStorageBackend(custom_dir)

        assert storage.base_dir == custom_dir
        assert custom_dir.exists()

    def test_save_job_metadata(self):
        """Test saving job to storage."""
        job = BatchJob(
            video_path=Path("test_video.mp4"),
            output_dir=Path("output"),
            config={"test": "config"},
        )

        # Save job
        self.storage.save_job_metadata(job)

        # Verify file exists
        job_file = self.storage.base_dir / "jobs" / job.job_id / "job_metadata.json"
        assert job_file.exists()

        # Verify content
        with open(job_file) as f:
            saved_data = json.load(f)

        assert saved_data["job_id"] == job.job_id
        assert saved_data["video_path"] == str(job.video_path)
        assert saved_data["config"] == {"test": "config"}

    def test_load_job_metadata(self):
        """Test loading job from storage."""
        # Create and save job
        original_job = BatchJob(
            video_path=Path("test_video.mp4"), config={"test": "config"}
        )
        self.storage.save_job_metadata(original_job)

        # Load job
        loaded_job = self.storage.load_job_metadata(original_job.job_id)

        assert loaded_job is not None
        assert loaded_job.job_id == original_job.job_id
        assert loaded_job.video_path == original_job.video_path
        assert loaded_job.config == original_job.config

    def test_load_nonexistent_job(self):
        """Test loading non-existent job."""
        result = self.storage.load_job_metadata("nonexistent_job_id")
        assert result is None

    def test_delete_job(self):
        """Test deleting job from storage."""
        # Create and save job
        job = BatchJob(video_path=Path("test_video.mp4"))
        self.storage.save_job_metadata(job)

        # Verify job exists
        assert self.storage.load_job_metadata(job.job_id) is not None

        # Delete job
        result = self.storage.delete_job(job.job_id)
        assert result is True

        # Verify job is gone
        assert self.storage.load_job_metadata(job.job_id) is None

    def test_delete_nonexistent_job(self):
        """Test deleting non-existent job."""
        result = self.storage.delete_job("nonexistent_job_id")
        assert result is False

    def test_list_jobs_empty(self):
        """Test listing jobs when none exist."""
        jobs = self.storage.list_jobs()
        assert jobs == []

    def test_list_jobs_with_data(self):
        """Test listing jobs with saved data."""
        # Create and save multiple jobs
        jobs = []
        for i in range(3):
            job = BatchJob(video_path=Path(f"video{i}.mp4"))
            jobs.append(job)
            self.storage.save_job_metadata(job)

        # List jobs
        listed_jobs = self.storage.list_jobs()
        assert len(listed_jobs) == 3

        # Verify job IDs match
        listed_ids = {job.job_id for job in listed_jobs}
        expected_ids = {job.job_id for job in jobs}
        assert listed_ids == expected_ids

    def test_save_report(self):
        """Test saving batch report."""
        report = BatchReport(
            batch_id="test_batch_123",
            start_time=None,  # Will be set automatically
            total_jobs=5,
            completed_jobs=3,
            failed_jobs=1,
        )

        # Save report
        self.storage.save_report(report)

        # Verify file exists
        report_file = (
            self.storage.base_dir / "reports" / f"batch_report_{report.batch_id}.json"
        )
        assert report_file.exists()

        # Verify content
        with open(report_file) as f:
            saved_data = json.load(f)

        assert saved_data["batch_id"] == report.batch_id
        assert saved_data["total_jobs"] == 5
        assert saved_data["completed_jobs"] == 3

    def test_load_report(self):
        """Test loading batch report."""
        # Create and save report
        original_report = BatchReport(
            batch_id="test_batch_123", start_time=None, total_jobs=10
        )
        self.storage.save_report(original_report)

        # Load report
        loaded_report = self.storage.load_report(original_report.batch_id)

        assert loaded_report is not None
        assert loaded_report.batch_id == original_report.batch_id
        assert loaded_report.total_jobs == original_report.total_jobs

    def test_load_nonexistent_report(self):
        """Test loading non-existent report."""
        result = self.storage.load_report("nonexistent_batch_id")
        assert result is None

    def test_list_reports_empty(self):
        """Test listing reports when none exist."""
        reports = self.storage.list_reports()
        assert reports == []

    def test_list_reports_with_data(self):
        """Test listing reports with saved data."""
        # Create and save multiple reports
        reports = []
        for i in range(3):
            report = BatchReport(batch_id=f"batch_{i}", start_time=None)
            reports.append(report)
            self.storage.save_report(report)

        # List reports
        listed_reports = self.storage.list_reports()
        assert len(listed_reports) == 3

        # Verify batch IDs match
        listed_ids = {report.batch_id for report in listed_reports}
        expected_ids = {report.batch_id for report in reports}
        assert listed_ids == expected_ids

    def test_cleanup_old_files(self):
        """Test cleaning up old files."""
        # Create old jobs and reports
        old_jobs = []
        for i in range(3):
            job = BatchJob(video_path=Path(f"old_video{i}.mp4"))
            old_jobs.append(job)
            self.storage.save_job_metadata(job)

        old_reports = []
        for i in range(2):
            report = BatchReport(batch_id=f"old_batch_{i}", start_time=None)
            old_reports.append(report)
            self.storage.save_report(report)

        # Cleanup with max_age=0 (delete all)
        deleted_jobs, deleted_reports = self.storage.cleanup_old_files(max_age_days=0)

        assert deleted_jobs == 3
        assert deleted_reports == 2

        # Verify files are gone
        assert len(self.storage.list_jobs()) == 0
        assert len(self.storage.list_reports()) == 0


class TestFileStorageBackendErrorHandling:
    """Test error handling in FileStorageBackend."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = FileStorageBackend(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("builtins.open", side_effect=PermissionError("Access denied"))
    def test_save_job_permission_error(self, mock_open):
        """Test handling permission errors during job save."""
        job = BatchJob(video_path=Path("test_video.mp4"))

        # Should handle permission error gracefully
        with pytest.raises(PermissionError):
            self.storage.save_job_metadata(job)

    @patch("builtins.open", new_callable=mock_open, read_data="invalid json")
    def test_load_job_invalid_json(self, mock_file):
        """Test handling invalid JSON during job load."""
        # Create a job directory and metadata file first
        job = BatchJob(video_path=Path("test_video.mp4"))
        job_dir = self.storage.base_dir / "jobs" / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        job_file = job_dir / "job_metadata.json"
        job_file.touch()

        # Should handle invalid JSON gracefully
        result = self.storage.load_job_metadata(job.job_id)
        assert result is None

    def test_save_job_invalid_data(self):
        """Test saving job with invalid data."""
        # Job with None values that can't be serialized properly
        job = BatchJob()
        job.video_path = None  # This might cause issues

        # Should handle gracefully or raise appropriate error
        try:
            self.storage.save_job_metadata(job)
        except (TypeError, ValueError):
            # Expected for invalid data
            pass

    def test_readonly_directory(self):
        """Test behavior with read-only directory."""
        # Create storage in temp directory
        readonly_dir = self.temp_dir / "readonly"
        readonly_dir.mkdir()

        try:
            # Make directory read-only
            readonly_dir.chmod(0o444)

            # Try to create storage backend
            # This should handle the permission issue gracefully
            storage = FileStorageBackend(readonly_dir)

            # Basic operations should fail gracefully
            job = BatchJob(video_path=Path("test_video.mp4"))

            with pytest.raises(PermissionError):
                storage.save_job_metadata(job)

        finally:
            # Restore permissions for cleanup
            readonly_dir.chmod(0o755)


@pytest.mark.integration
class TestFileStorageBackendIntegration:
    """Integration tests for FileStorageBackend."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = FileStorageBackend(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_complete_job_lifecycle(self):
        """Test complete job save/load/delete lifecycle."""
        # Create job with complex data
        job = BatchJob(
            video_path=Path("complex_video.mp4"),
            output_dir=Path("output"),
            config={
                "pipelines": {
                    "audio": {"enabled": True, "params": {"sample_rate": 16000}},
                    "face_analysis": {"enabled": True},
                },
                "quality": "high",
            },
            selected_pipelines=["audio", "face_analysis"],
        )

        # Add pipeline results
        from videoannotator.batch.types import PipelineResult

        result = PipelineResult(
            pipeline_name="audio",
            status=JobStatus.COMPLETED,
            processing_time=15.5,
            annotation_count=42,
        )
        job.pipeline_results["audio"] = result

        # Complete lifecycle
        # 1. Save
        self.storage.save_job_metadata(job)

        # 2. Load and verify
        loaded_job = self.storage.load_job_metadata(job.job_id)
        assert loaded_job.job_id == job.job_id
        assert loaded_job.config == job.config
        assert "audio" in loaded_job.pipeline_results

        # 3. Update and save again
        loaded_job.status = JobStatus.COMPLETED
        self.storage.save_job_metadata(loaded_job)

        # 4. Load updated version
        updated_job = self.storage.load_job_metadata(job.job_id)
        assert updated_job.status == JobStatus.COMPLETED

        # 5. Delete
        assert self.storage.delete_job(job.job_id) is True
        assert self.storage.load_job_metadata(job.job_id) is None

    def test_batch_operations(self):
        """Test batch operations with multiple jobs and reports."""
        # Create multiple jobs
        jobs = []
        for i in range(10):
            job = BatchJob(
                video_path=Path(f"batch_video_{i}.mp4"),
                config={"batch_id": f"batch_{i // 5}"},  # 2 batches of 5
            )
            jobs.append(job)
            self.storage.save_job_metadata(job)

        # Create batch reports
        reports = []
        for batch_id in ["batch_0", "batch_1"]:
            report = BatchReport(batch_id=batch_id, start_time=None, total_jobs=5)
            reports.append(report)
            self.storage.save_report(report)

        # Verify all data exists
        listed_jobs = self.storage.list_jobs()
        assert len(listed_jobs) == 10

        listed_reports = self.storage.list_reports()
        assert len(listed_reports) == 2

        # Cleanup test
        deleted_jobs, deleted_reports = self.storage.cleanup_old_files(max_age_days=0)
        assert deleted_jobs == 10
        assert deleted_reports == 2
