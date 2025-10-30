"""Unit tests for CancellationManager.

Tests job cancellation tracking, async task cancellation, and cleanup.
"""

import asyncio

import pytest

from videoannotator.worker.cancellation import CancellationManager


class TestCancellationManager:
    """Test suite for CancellationManager class."""

    def test_initialization(self):
        """Test manager initializes correctly."""
        manager = CancellationManager()

        assert manager.running_tasks == {}
        assert manager.cancellation_requests == set()

    @pytest.mark.asyncio
    async def test_register_task(self):
        """Test task registration."""
        manager = CancellationManager()

        async def dummy_task():
            await asyncio.sleep(1)

        task = asyncio.create_task(dummy_task())
        manager.register_task("job1", task)

        assert "job1" in manager.running_tasks
        assert manager.running_tasks["job1"] == task

        # Cleanup
        task.cancel()

    @pytest.mark.asyncio
    async def test_unregister_task(self):
        """Test task unregistration."""
        manager = CancellationManager()

        async def dummy_task():
            await asyncio.sleep(1)

        task = asyncio.create_task(dummy_task())
        manager.register_task("job1", task)
        manager.unregister_task("job1")

        assert "job1" not in manager.running_tasks

        # Cleanup
        task.cancel()

    def test_unregister_nonexistent_task(self):
        """Test unregistering task that doesn't exist (no-op)."""
        manager = CancellationManager()

        # Should not raise error
        manager.unregister_task("nonexistent")
        assert manager.running_tasks == {}

    @pytest.mark.asyncio
    async def test_cancel_running_job(self):
        """Test cancelling a running job successfully."""
        manager = CancellationManager()

        async def long_running_task():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                # Task was cancelled
                raise

        task = asyncio.create_task(long_running_task())
        manager.register_task("job1", task)

        # Cancel the job
        result = await manager.cancel_job("job1", timeout=1.0)

        assert result is True
        assert "job1" in manager.cancellation_requests
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_queued_job(self):
        """Test cancelling a job that's not running (queued)."""
        manager = CancellationManager()

        # Job not registered as running task
        result = await manager.cancel_job("job1", timeout=1.0)

        # Should succeed (job will be marked as cancelled when it tries to start)
        assert result is True
        assert "job1" in manager.cancellation_requests

    @pytest.mark.asyncio
    async def test_cancel_completed_job(self):
        """Test cancelling a job that already completed."""
        manager = CancellationManager()

        async def quick_task():
            await asyncio.sleep(0.01)
            return "done"

        task = asyncio.create_task(quick_task())
        manager.register_task("job1", task)

        # Wait for task to complete
        await task

        # Try to cancel completed task
        result = await manager.cancel_job("job1", timeout=1.0)

        # Should handle gracefully
        assert result is True
        assert "job1" in manager.cancellation_requests

    def test_is_cancellation_requested(self):
        """Test checking if cancellation was requested."""
        manager = CancellationManager()

        assert manager.is_cancellation_requested("job1") is False

        manager.cancellation_requests.add("job1")
        assert manager.is_cancellation_requested("job1") is True

    def test_clear_cancellation_request(self):
        """Test clearing cancellation request."""
        manager = CancellationManager()

        manager.cancellation_requests.add("job1")
        manager.clear_cancellation_request("job1")

        assert "job1" not in manager.cancellation_requests

    def test_clear_nonexistent_cancellation_request(self):
        """Test clearing non-existent request (no-op)."""
        manager = CancellationManager()

        # Should not raise error
        manager.clear_cancellation_request("nonexistent")
        assert manager.cancellation_requests == set()

    @pytest.mark.asyncio
    async def test_get_running_job_ids(self):
        """Test getting set of running job IDs."""
        manager = CancellationManager()

        async def dummy_task():
            await asyncio.sleep(1)

        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())

        manager.register_task("job1", task1)
        manager.register_task("job2", task2)

        running_ids = manager.get_running_job_ids()

        assert running_ids == {"job1", "job2"}

        # Cleanup
        task1.cancel()
        task2.cancel()

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup cancels all tasks and clears state."""
        manager = CancellationManager()

        async def dummy_task():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise

        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())

        manager.register_task("job1", task1)
        manager.register_task("job2", task2)
        manager.cancellation_requests.add("job3")

        manager.cleanup()

        # Give the event loop a moment to process cancellations
        await asyncio.sleep(0.01)

        assert manager.running_tasks == {}
        assert manager.cancellation_requests == set()
        assert task1.cancelled()
        assert task2.cancelled()

    @pytest.mark.asyncio
    async def test_cancel_timeout(self):
        """Test cancellation succeeds even if task takes time to cancel."""
        manager = CancellationManager()

        async def quick_cancel_task():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                # Task responds to cancellation
                raise

        task = asyncio.create_task(quick_cancel_task())
        manager.register_task("job1", task)

        # Cancel with timeout
        result = await manager.cancel_job("job1", timeout=0.2)

        # Should succeed
        assert result is True
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_multiple_cancellations_idempotent(self):
        """Test that multiple cancel requests for same job are idempotent."""
        manager = CancellationManager()

        async def long_task():
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                raise

        task = asyncio.create_task(long_task())
        manager.register_task("job1", task)

        # Cancel multiple times
        result1 = await manager.cancel_job("job1", timeout=1.0)
        result2 = await manager.cancel_job("job1", timeout=1.0)

        assert result1 is True
        assert result2 is True  # Second call should also succeed (idempotent)
        assert task.cancelled()

    @pytest.mark.asyncio
    async def test_register_multiple_tasks_same_id_overwrites(self):
        """Test that registering a new task with same ID overwrites the old one."""
        manager = CancellationManager()

        async def dummy_task():
            await asyncio.sleep(1)

        task1 = asyncio.create_task(dummy_task())
        task2 = asyncio.create_task(dummy_task())

        manager.register_task("job1", task1)
        manager.register_task("job1", task2)  # Overwrites task1

        assert manager.running_tasks["job1"] == task2
        assert manager.running_tasks["job1"] != task1

        # Cleanup
        task1.cancel()
        task2.cancel()
