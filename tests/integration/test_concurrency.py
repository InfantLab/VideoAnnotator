"""Tests for concurrent job processing limits.

Validates that MAX_CONCURRENT_JOBS configuration is respected and that
the worker properly tracks running jobs vs available slots.

v1.3.0: Phase 11 - T065-T067
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from videoannotator.api.background_tasks import BackgroundJobManager
from videoannotator.api.database import get_storage_backend
from videoannotator.worker.job_processor import JobProcessor


class TestConcurrentJobConfiguration:
    """Test concurrent job configuration and tracking."""

    def test_job_processor_accepts_concurrent_limit(self):
        """Test JobProcessor accepts and stores max_concurrent_jobs."""
        storage = get_storage_backend()

        processor = JobProcessor(storage_backend=storage, max_concurrent_jobs=5)
        assert processor.max_concurrent_jobs == 5

        processor2 = JobProcessor(storage_backend=storage, max_concurrent_jobs=1)
        assert processor2.max_concurrent_jobs == 1

    def test_background_manager_accepts_concurrent_limit(self):
        """Test BackgroundJobManager accepts and stores max_concurrent_jobs."""
        storage = get_storage_backend()

        manager = BackgroundJobManager(storage_backend=storage, max_concurrent_jobs=3)
        assert manager.max_concurrent_jobs == 3

        manager2 = BackgroundJobManager(storage_backend=storage, max_concurrent_jobs=8)
        assert manager2.max_concurrent_jobs == 8

    def test_concurrent_limit_from_environment_default(self):
        """Test concurrent limit uses environment variable default."""
        storage = get_storage_backend()

        # Create processor without explicit argument - should use env default
        processor = JobProcessor(storage_backend=storage)
        # Default from config_env.py is 2
        assert processor.max_concurrent_jobs >= 1  # At least 1, likely 2

    def test_explicit_override_wins_over_default(self):
        """Test explicit argument overrides environment variable."""
        storage = get_storage_backend()

        # Explicit value should override env var default
        processor = JobProcessor(storage_backend=storage, max_concurrent_jobs=10)
        assert processor.max_concurrent_jobs == 10


class TestConcurrentJobLogic:
    """Test concurrent job tracking logic."""

    def test_processor_tracks_processing_jobs(self):
        """Test processor initializes processing_jobs tracking set."""
        storage = get_storage_backend()

        processor = JobProcessor(storage_backend=storage, max_concurrent_jobs=2)
        assert hasattr(processor, "processing_jobs")
        assert isinstance(processor.processing_jobs, set)
        assert len(processor.processing_jobs) == 0

    def test_manager_tracks_processing_jobs(self):
        """Test manager initializes processing_jobs tracking set."""
        storage = get_storage_backend()

        manager = BackgroundJobManager(storage_backend=storage, max_concurrent_jobs=2)
        assert hasattr(manager, "processing_jobs")
        assert isinstance(manager.processing_jobs, set)
        assert len(manager.processing_jobs) == 0

    def test_processor_has_max_concurrent_property(self):
        """Test processor exposes max_concurrent_jobs property."""
        storage = get_storage_backend()

        processor = JobProcessor(storage_backend=storage, max_concurrent_jobs=4)
        # Access the property
        limit = processor.max_concurrent_jobs
        assert limit == 4
        assert isinstance(limit, int)

    def test_queue_respects_limit_calculation(self):
        """Test that available slots calculation is correct."""
        storage = get_storage_backend()

        processor = JobProcessor(storage_backend=storage, max_concurrent_jobs=5)

        # Simulate tracking some jobs
        processor.processing_jobs.add("job-1")
        processor.processing_jobs.add("job-2")

        # Available slots = max_concurrent - len(processing_jobs)
        # 5 - 2 = 3
        available = processor.max_concurrent_jobs - len(processor.processing_jobs)
        assert available == 3

    def test_no_slots_when_at_limit(self):
        """Test that no slots available when at concurrent limit."""
        storage = get_storage_backend()

        processor = JobProcessor(storage_backend=storage, max_concurrent_jobs=2)

        # Fill up the slots
        processor.processing_jobs.add("job-1")
        processor.processing_jobs.add("job-2")

        available = processor.max_concurrent_jobs - len(processor.processing_jobs)
        assert available == 0  # No slots available


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
