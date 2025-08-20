"""
Unit tests for batch processing types and data structures.

Tests the core data types used throughout the batch processing system,
including JobStatus, BatchJob, PipelineResult, BatchStatus, and BatchReport.
"""

import pytest
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock

from src.batch.types import (
    JobStatus,
    BatchJob,
    PipelineResult,
    BatchStatus,
    BatchReport
)


class TestJobStatus:
    """Test JobStatus enumeration."""
    
    def test_job_status_values(self):
        """Test that all job status values are correct."""
        assert JobStatus.PENDING.value == "pending"
        assert JobStatus.RUNNING.value == "running"
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"
        assert JobStatus.RETRYING.value == "retrying"
        assert JobStatus.CANCELLED.value == "cancelled"
    
    def test_job_status_string_conversion(self):
        """Test string conversion of job status."""
        assert str(JobStatus.PENDING) == "JobStatus.PENDING"
        assert JobStatus.PENDING.value == "pending"


class TestPipelineResult:
    """Test PipelineResult data class."""
    
    def test_pipeline_result_creation(self):
        """Test basic pipeline result creation."""
        result = PipelineResult(
            pipeline_name="test_pipeline",
            status=JobStatus.COMPLETED
        )
        
        assert result.pipeline_name == "test_pipeline"
        assert result.status == JobStatus.COMPLETED
        assert result.start_time is None
        assert result.end_time is None
        assert result.processing_time is None
        assert result.annotation_count is None
        assert result.output_file is None
        assert result.error_message is None
    
    def test_pipeline_result_with_timing(self):
        """Test pipeline result with timing information."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=10)
        
        result = PipelineResult(
            pipeline_name="test_pipeline",
            status=JobStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            processing_time=10.0,
            annotation_count=5,
            output_file=Path("test_output.json")
        )
        
        assert result.duration == 10.0
        assert result.annotation_count == 5
        assert result.output_file == Path("test_output.json")
    
    def test_pipeline_result_duration_calculation(self):
        """Test duration calculation from start/end times."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=5.5)
        
        result = PipelineResult(
            pipeline_name="test_pipeline",
            status=JobStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time
        )
        
        assert abs(result.duration - 5.5) < 0.1  # Allow small floating point differences
    
    def test_pipeline_result_duration_fallback(self):
        """Test duration fallback to processing_time."""
        result = PipelineResult(
            pipeline_name="test_pipeline",
            status=JobStatus.COMPLETED,
            processing_time=7.5
        )
        
        assert result.duration == 7.5
    
    def test_pipeline_result_no_duration(self):
        """Test pipeline result with no duration information."""
        result = PipelineResult(
            pipeline_name="test_pipeline",
            status=JobStatus.PENDING
        )
        
        assert result.duration is None


class TestBatchJob:
    """Test BatchJob data class."""
    
    def test_batch_job_creation(self):
        """Test basic batch job creation."""
        video_path = Path("test_video.mp4")
        output_dir = Path("output")
        
        job = BatchJob(
            video_path=video_path,
            output_dir=output_dir,
            config={"test": "config"}
        )
        
        assert job.video_path == video_path
        assert job.output_dir == output_dir
        assert job.config == {"test": "config"}
        assert job.status == JobStatus.PENDING
        assert job.retry_count == 0
        assert job.error_message is None
        assert job.selected_pipelines is None
        assert isinstance(job.job_id, str)
        assert len(job.job_id) == 36  # UUID length
    
    def test_batch_job_video_id(self):
        """Test video ID generation from path."""
        job = BatchJob(video_path=Path("/path/to/test_video.mp4"))
        assert job.video_id == "test_video"
        
        job_no_path = BatchJob()
        assert job_no_path.video_id == "unknown"
    
    def test_batch_job_duration(self):
        """Test job duration calculation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=15)
        
        job = BatchJob(
            started_at=start_time,
            completed_at=end_time
        )
        
        assert abs(job.duration - 15.0) < 0.1
    
    def test_batch_job_duration_incomplete(self):
        """Test duration when job is not complete."""
        job = BatchJob(started_at=datetime.now())
        assert job.duration is None
    
    def test_batch_job_is_complete(self):
        """Test job completion status."""
        # Pending job
        job = BatchJob(status=JobStatus.PENDING)
        assert not job.is_complete
        
        # Running job
        job.status = JobStatus.RUNNING
        assert not job.is_complete
        
        # Retrying job
        job.status = JobStatus.RETRYING
        assert not job.is_complete
        
        # Completed job
        job.status = JobStatus.COMPLETED
        assert job.is_complete
        
        # Failed job
        job.status = JobStatus.FAILED
        assert job.is_complete
        
        # Cancelled job
        job.status = JobStatus.CANCELLED
        assert job.is_complete
    
    def test_batch_job_serialization(self):
        """Test job serialization to dictionary."""
        video_path = Path("test_video.mp4")
        output_dir = Path("output")
        created_at = datetime.now()
        
        # Create job with pipeline result
        job = BatchJob(
            video_path=video_path,
            output_dir=output_dir,
            config={"test": "config"},
            status=JobStatus.RUNNING,
            created_at=created_at,
            retry_count=1,
            error_message="Test error",
            selected_pipelines=["pipeline1", "pipeline2"]
        )
        
        # Add pipeline result
        pipeline_result = PipelineResult(
            pipeline_name="test_pipeline",
            status=JobStatus.COMPLETED,
            processing_time=5.0,
            annotation_count=3
        )
        job.pipeline_results["test_pipeline"] = pipeline_result
        
        # Serialize
        job_dict = job.to_dict()
        
        # Verify serialization
        assert job_dict["job_id"] == job.job_id
        assert job_dict["video_path"] == str(video_path)
        assert job_dict["output_dir"] == str(output_dir)
        assert job_dict["config"] == {"test": "config"}
        assert job_dict["status"] == "running"
        assert job_dict["retry_count"] == 1
        assert job_dict["error_message"] == "Test error"
        assert job_dict["selected_pipelines"] == ["pipeline1", "pipeline2"]
        assert "test_pipeline" in job_dict["pipeline_results"]
        
        pipeline_dict = job_dict["pipeline_results"]["test_pipeline"]
        assert pipeline_dict["pipeline_name"] == "test_pipeline"
        assert pipeline_dict["status"] == "completed"
        assert pipeline_dict["processing_time"] == 5.0
        assert pipeline_dict["annotation_count"] == 3
    
    def test_batch_job_deserialization(self):
        """Test job deserialization from dictionary."""
        job_dict = {
            "job_id": str(uuid.uuid4()),
            "video_path": "test_video.mp4",
            "output_dir": "output",
            "config": {"test": "config"},
            "status": "completed",
            "pipeline_results": {
                "test_pipeline": {
                    "pipeline_name": "test_pipeline",
                    "status": "completed",
                    "start_time": datetime.now().isoformat(),
                    "end_time": (datetime.now() + timedelta(seconds=5)).isoformat(),
                    "processing_time": 5.0,
                    "annotation_count": 3,
                    "output_file": "output.json",
                    "error_message": None
                }
            },
            "created_at": datetime.now().isoformat(),
            "started_at": datetime.now().isoformat(),
            "completed_at": (datetime.now() + timedelta(seconds=10)).isoformat(),
            "retry_count": 0,
            "error_message": None,
            "selected_pipelines": ["pipeline1"]
        }
        
        # Deserialize
        job = BatchJob.from_dict(job_dict)
        
        # Verify deserialization
        assert job.job_id == job_dict["job_id"]
        assert job.video_path == Path("test_video.mp4")
        assert job.output_dir == Path("output")
        assert job.config == {"test": "config"}
        assert job.status == JobStatus.COMPLETED
        assert job.retry_count == 0
        assert job.selected_pipelines == ["pipeline1"]
        assert "test_pipeline" in job.pipeline_results
        
        pipeline_result = job.pipeline_results["test_pipeline"]
        assert pipeline_result.pipeline_name == "test_pipeline"
        assert pipeline_result.status == JobStatus.COMPLETED
        assert pipeline_result.processing_time == 5.0
        assert pipeline_result.annotation_count == 3


class TestBatchStatus:
    """Test BatchStatus data class."""
    
    def test_batch_status_creation(self):
        """Test basic batch status creation."""
        status = BatchStatus()
        
        assert status.total_jobs == 0
        assert status.pending_jobs == 0
        assert status.running_jobs == 0
        assert status.completed_jobs == 0
        assert status.failed_jobs == 0
        assert status.cancelled_jobs == 0
        assert status.total_processing_time == 0.0
        assert status.average_processing_time == 0.0
        assert status.estimated_completion_time is None
        assert status.current_jobs == []
    
    def test_progress_percentage(self):
        """Test progress percentage calculation."""
        # Empty batch
        status = BatchStatus()
        assert status.progress_percentage == 0.0
        
        # Partially complete batch
        status.total_jobs = 10
        status.completed_jobs = 3
        status.failed_jobs = 1
        status.cancelled_jobs = 1
        assert status.progress_percentage == 50.0  # (3+1+1)/10 * 100
        
        # Fully complete batch
        status.completed_jobs = 6
        status.failed_jobs = 2
        status.cancelled_jobs = 2
        assert status.progress_percentage == 100.0
    
    def test_success_rate(self):
        """Test success rate calculation."""
        # No completed jobs
        status = BatchStatus()
        assert status.success_rate == 0.0  # No jobs completed yet
        
        # Mixed results
        status.completed_jobs = 7
        status.failed_jobs = 2
        status.cancelled_jobs = 1
        assert status.success_rate == 70.0  # 7/(7+2+1) * 100
        
        # All successful
        status.completed_jobs = 5
        status.failed_jobs = 0
        status.cancelled_jobs = 0
        assert status.success_rate == 100.0
        
        # All failed
        status.completed_jobs = 0
        status.failed_jobs = 5
        status.cancelled_jobs = 0
        assert status.success_rate == 0.0


class TestBatchReport:
    """Test BatchReport data class."""
    
    def test_batch_report_creation(self):
        """Test basic batch report creation."""
        batch_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        report = BatchReport(
            batch_id=batch_id,
            start_time=start_time
        )
        
        assert report.batch_id == batch_id
        assert report.start_time == start_time
        assert report.end_time is None
        assert report.total_jobs == 0
        assert report.completed_jobs == 0
        assert report.failed_jobs == 0
        assert report.cancelled_jobs == 0
        assert report.total_processing_time == 0.0
        assert report.jobs == []
        assert report.errors == []
    
    def test_batch_report_duration(self):
        """Test batch duration calculation."""
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=30)
        
        report = BatchReport(
            batch_id="test",
            start_time=start_time,
            end_time=end_time
        )
        
        assert abs(report.duration - 30.0) < 0.1
    
    def test_batch_report_duration_incomplete(self):
        """Test duration when batch is not complete."""
        report = BatchReport(
            batch_id="test",
            start_time=datetime.now()
        )
        
        assert report.duration is None
    
    def test_batch_report_success_rate(self):
        """Test success rate calculation."""
        report = BatchReport(
            batch_id="test",
            start_time=datetime.now(),
            total_jobs=10,
            completed_jobs=7,
            failed_jobs=2,
            cancelled_jobs=1
        )
        
        assert report.success_rate == 70.0  # 7/10 * 100
    
    def test_batch_report_serialization(self):
        """Test batch report serialization."""
        batch_id = str(uuid.uuid4())
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=20)
        
        # Create sample job
        job = BatchJob(video_path=Path("test.mp4"))
        
        report = BatchReport(
            batch_id=batch_id,
            start_time=start_time,
            end_time=end_time,
            total_jobs=1,
            completed_jobs=1,
            failed_jobs=0,
            cancelled_jobs=0,
            total_processing_time=15.0,
            jobs=[job],
            errors=["Test error"]
        )
        
        # Serialize
        report_dict = report.to_dict()
        
        # Verify serialization
        assert report_dict["batch_id"] == batch_id
        assert report_dict["total_jobs"] == 1
        assert report_dict["completed_jobs"] == 1
        assert report_dict["failed_jobs"] == 0
        assert report_dict["cancelled_jobs"] == 0
        assert report_dict["total_processing_time"] == 15.0
        assert abs(report_dict["duration"] - 20.0) < 0.1
        assert report_dict["success_rate"] == 100.0
        assert len(report_dict["jobs"]) == 1
        assert report_dict["errors"] == ["Test error"]


@pytest.mark.integration
class TestTypesIntegration:
    """Integration tests for batch types."""
    
    def test_complete_job_workflow(self):
        """Test complete job workflow through status changes."""
        # Create job
        job = BatchJob(
            video_path=Path("test_video.mp4"),
            output_dir=Path("output"),
            selected_pipelines=["pipeline1", "pipeline2"]
        )
        
        assert job.status == JobStatus.PENDING
        assert not job.is_complete
        
        # Start job
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        assert not job.is_complete
        
        # Add pipeline results
        result1 = PipelineResult(
            pipeline_name="pipeline1",
            status=JobStatus.COMPLETED,
            processing_time=5.0,
            annotation_count=3
        )
        job.pipeline_results["pipeline1"] = result1
        
        result2 = PipelineResult(
            pipeline_name="pipeline2",
            status=JobStatus.COMPLETED,
            processing_time=7.0,
            annotation_count=5
        )
        job.pipeline_results["pipeline2"] = result2
        
        # Complete job
        job.status = JobStatus.COMPLETED
        job.completed_at = datetime.now()
        assert job.is_complete
        
        # Verify job can be serialized and deserialized
        job_dict = job.to_dict()
        restored_job = BatchJob.from_dict(job_dict)
        
        assert restored_job.job_id == job.job_id
        assert restored_job.status == JobStatus.COMPLETED
        assert len(restored_job.pipeline_results) == 2
        assert restored_job.is_complete
    
    def test_batch_status_calculation_from_jobs(self):
        """Test batch status calculation from list of jobs."""
        # Create various jobs in different states
        jobs = [
            BatchJob(status=JobStatus.PENDING),
            BatchJob(status=JobStatus.PENDING),
            BatchJob(status=JobStatus.RUNNING),
            BatchJob(status=JobStatus.COMPLETED),
            BatchJob(status=JobStatus.COMPLETED),
            BatchJob(status=JobStatus.FAILED),
            BatchJob(status=JobStatus.CANCELLED),
        ]
        
        # Manually count (this would normally be done by progress tracker)
        status = BatchStatus(
            total_jobs=len(jobs),
            pending_jobs=2,
            running_jobs=1,
            completed_jobs=2,
            failed_jobs=1,
            cancelled_jobs=1,
            total_processing_time=25.0
        )
        
        assert status.progress_percentage == 57.14285714285714  # (2+1+1)/7 * 100
        assert status.success_rate == 50.0  # 2/(2+1+1) * 100
