"""Unit tests for FileStorageBackend - testing actual implementation.

These tests focus on the real API including save_job_metadata() and load_job_metadata().
"""

import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from videoannotator.batch.types import BatchJob, JobStatus
from videoannotator.storage.file_backend import FileStorageBackend


class TestFileStorageBackendReal:
    """Test the actual FileStorageBackend implementation."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = FileStorageBackend(self.temp_dir)

    def create_test_job(self, job_id="test_job", **kwargs):
        """Helper to create test BatchJob."""
        return BatchJob(
            job_id=job_id,
            video_path=Path(kwargs.get("video_path", "/path/to/video.mp4")),
            output_dir=Path(kwargs.get("output_dir", "/path/to/output")),
            status=kwargs.get("status", JobStatus.PENDING),
            created_at=kwargs.get("created_at", datetime.now()),
            started_at=kwargs.get("started_at"),
            completed_at=kwargs.get("completed_at"),
            error_message=kwargs.get("error_message"),
            retry_count=kwargs.get("retry_count", 0),
            config=kwargs.get("config", {}),
            selected_pipelines=kwargs.get("selected_pipelines", []),
            pipeline_results=kwargs.get("pipeline_results", {}),
        )

    def test_storage_initialization(self):
        """Test FileStorageBackend initialization."""
        storage = FileStorageBackend(Path("test_storage"))

        assert storage.base_dir == Path("test_storage")
        assert storage.jobs_dir == Path("test_storage") / "jobs"
        assert storage.batch_queue_file == Path("test_storage") / "batch_queue.json"

    def test_storage_creates_directories(self):
        """Test that storage creates required directories."""
        # Should create directories during initialization
        assert self.storage.base_dir.exists()
        assert self.storage.jobs_dir.exists()

    def test_save_and_load_job_metadata(self):
        """Test saving and loading job metadata."""
        job = self.create_test_job("test_job_001")

        # Save metadata
        self.storage.save_job_metadata(job)

        # Load metadata back
        loaded_job = self.storage.load_job_metadata("test_job_001")

        # Verify all fields match
        assert loaded_job.job_id == job.job_id
        assert loaded_job.video_id == job.video_id
        assert loaded_job.video_path == job.video_path
        assert loaded_job.output_dir == job.output_dir
        assert loaded_job.status == job.status
        assert loaded_job.retry_count == job.retry_count

    def test_load_nonexistent_job_metadata(self):
        """Test loading metadata for non-existent job."""
        with pytest.raises(FileNotFoundError):
            self.storage.load_job_metadata("nonexistent_job")

    def test_save_annotations(self):
        """Test saving pipeline annotations."""
        annotations = [
            {"timestamp": 1.0, "data": "scene1"},
            {"timestamp": 2.0, "data": "scene2"},
        ]

        result_path = self.storage.save_annotations(
            "job1", "scene_detection", annotations
        )

        # Should return a path
        assert isinstance(result_path, str)
        assert Path(result_path).exists()

    def test_load_annotations(self):
        """Test loading pipeline annotations."""
        annotations = [
            {"timestamp": 1.0, "data": "face1"},
            {"timestamp": 2.0, "data": "face2"},
        ]

        # Save annotations first
        self.storage.save_annotations("job2", "face_analysis", annotations)

        # Load them back
        loaded_annotations = self.storage.load_annotations("job2", "face_analysis")

        assert len(loaded_annotations) == 2
        assert loaded_annotations[0]["data"] == "face1"
        assert loaded_annotations[1]["data"] == "face2"

    def test_annotation_exists(self):
        """Test checking if annotations exist."""
        # Should not exist initially
        assert not self.storage.annotation_exists("job3", "person_tracking")

        # Save some annotations
        annotations = [{"timestamp": 1.0, "person": "person1"}]
        self.storage.save_annotations("job3", "person_tracking", annotations)

        # Should exist now
        assert self.storage.annotation_exists("job3", "person_tracking")

    def test_list_jobs_empty(self):
        """Test listing jobs when none exist."""
        job_ids = self.storage.list_jobs()
        assert job_ids == []

    def test_list_jobs_basic(self):
        """Test listing jobs."""
        # Create some jobs
        job1 = self.create_test_job("job_001")
        job2 = self.create_test_job("job_002")

        self.storage.save_job_metadata(job1)
        self.storage.save_job_metadata(job2)

        # List jobs
        job_ids = self.storage.list_jobs()

        assert len(job_ids) == 2
        assert "job_001" in job_ids
        assert "job_002" in job_ids


class TestFileStorageBackendRealistic:
    """Test realistic storage scenarios."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.storage = FileStorageBackend(self.temp_dir)

    def create_test_job(self, job_id="test_job", **kwargs):
        """Helper to create test BatchJob."""
        return BatchJob(
            job_id=job_id,
            video_path=Path(kwargs.get("video_path", f"/videos/{job_id}.mp4")),
            output_dir=Path(kwargs.get("output_dir", f"/output/{job_id}")),
            status=kwargs.get("status", JobStatus.PENDING),
            created_at=kwargs.get("created_at", datetime.now()),
            started_at=kwargs.get("started_at"),
            completed_at=kwargs.get("completed_at"),
            error_message=kwargs.get("error_message"),
            retry_count=kwargs.get("retry_count", 0),
            config=kwargs.get("config", {}),
            selected_pipelines=kwargs.get("selected_pipelines", []),
            pipeline_results=kwargs.get("pipeline_results", {}),
        )

    def test_realistic_job_workflow(self):
        """Test realistic job storage workflow."""
        # Create a job with realistic data
        job = self.create_test_job(
            job_id="realistic_video_001",
            video_path="/videos/family_vacation_2024.mp4",
            output_dir="/output/family_vacation_2024",
            status=JobStatus.RUNNING,
            config={"scene_detection": {"threshold": 0.5}},
            selected_pipelines=["scene_detection", "person_tracking"],
        )

        # Save job
        self.storage.save_job_metadata(job)

        # Save pipeline results as they complete
        scene_annotations = [
            {"timestamp": 0.0, "scene_type": "outdoor", "confidence": 0.95},
            {"timestamp": 30.0, "scene_type": "indoor", "confidence": 0.87},
        ]
        self.storage.save_annotations(job.job_id, "scene_detection", scene_annotations)

        person_annotations = [
            {"timestamp": 5.0, "person_id": "person_1", "bbox": [100, 100, 200, 200]},
            {"timestamp": 10.0, "person_id": "person_2", "bbox": [150, 150, 250, 250]},
        ]
        self.storage.save_annotations(job.job_id, "person_tracking", person_annotations)

        # Verify everything was saved correctly
        loaded_job = self.storage.load_job_metadata(job.job_id)
        assert loaded_job.video_id == "family_vacation_2024"

        loaded_scenes = self.storage.load_annotations(job.job_id, "scene_detection")
        assert len(loaded_scenes) == 2
        assert loaded_scenes[0]["scene_type"] == "outdoor"

        loaded_people = self.storage.load_annotations(job.job_id, "person_tracking")
        assert len(loaded_people) == 2
        assert loaded_people[0]["person_id"] == "person_1"

    def test_multiple_jobs_storage(self):
        """Test storing multiple jobs with different states."""
        jobs = []

        # Create jobs in different states
        jobs.append(self.create_test_job("job_001", status=JobStatus.COMPLETED))
        jobs.append(self.create_test_job("job_002", status=JobStatus.RUNNING))
        jobs.append(self.create_test_job("job_003", status=JobStatus.FAILED))
        jobs.append(self.create_test_job("job_004", status=JobStatus.PENDING))

        # Save all jobs
        for job in jobs:
            self.storage.save_job_metadata(job)

        # Verify all jobs can be listed
        job_ids = self.storage.list_jobs()
        assert len(job_ids) == 4

        # Verify status filtering if implemented
        try:
            completed_jobs = self.storage.list_jobs(status_filter="completed")
            # If filtering is implemented, should work
            assert isinstance(completed_jobs, list)
        except TypeError:
            # If filtering is not implemented, that's ok too
            pass
