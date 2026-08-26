"""The single job-execution path (spec 006).

Before this module existed, "run a job's selected pipelines" was implemented
three separate times — `api/job_processor.py`, `batch/batch_orchestrator.py`'s
`_process_single_job`, and (via that same orchestrator)
`worker/job_processor.py` — with real, previously-confirmed drift between
them (per-pipeline config was silently dropped on one path; only one path
checked for cancellation at all, and it checked an in-memory,
orchestrator-instance-scoped flag rather than the job's own persisted
status, so an API-triggered `/cancel` was invisible to it).

`run_job_pipelines()` is now the only place that dispatches a job's
pipelines. Both `api/job_processor.py` (used by the API server's default
background poller) and `batch/batch_orchestrator.py` (used by the CLI
worker path) call it. See specs/006-job-execution-consolidation/spec.md.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from ..storage.base import StorageBackend
from .types import BatchJob, JobStatus, PipelineResult

logger = logging.getLogger(__name__)


def run_job_pipelines(
    job: BatchJob,
    storage: StorageBackend,
    pipeline_classes: dict[str, type],
) -> BatchJob:
    """Run every pipeline in `job.selected_pipelines` and settle the job's
    final status. Mutates and returns `job`; the final status (COMPLETED /
    FAILED / CANCELLED) is already persisted via `storage.save_job_metadata`
    by the time this returns — callers should not re-decide or re-save it.

    Never raises: any unexpected exception is caught, recorded as a failed
    job, and persisted, matching the defensive behaviour both prior
    implementations already had.
    """
    try:
        return _run(job, storage, pipeline_classes)
    except Exception as e:  # top-level safety net, see docstring
        logger.error(f"Unexpected error running job {job.job_id}: {e}", exc_info=True)
        job.status = JobStatus.FAILED
        job.error_message = str(e)
        job.completed_at = datetime.now()
        storage.save_job_metadata(job)
        return job


def _run(
    job: BatchJob,
    storage: StorageBackend,
    pipeline_classes: dict[str, type],
) -> BatchJob:
    # Idempotent: the API's background poller already does this before
    # calling in; the CLI/BatchOrchestrator path didn't previously mark a
    # job RUNNING at all, so it's set here unconditionally to guarantee
    # both paths give the same visibility while a job is in progress.
    job.status = JobStatus.RUNNING
    job.started_at = job.started_at or datetime.now()
    storage.save_job_metadata(job)

    if job.output_dir is None and job.storage_path:
        job.output_dir = job.storage_path
    if job.output_dir is None:
        job.output_dir = Path.cwd() / "storage" / "jobs" / job.job_id / "output"
    job.output_dir.mkdir(parents=True, exist_ok=True)

    pipelines_to_run = _resolve_pipelines(job, pipeline_classes)
    if job.pipeline_results is None:
        job.pipeline_results = {}

    total = len(pipelines_to_run)
    for completed_count, pipeline_name in enumerate(pipelines_to_run):
        _run_one_pipeline(job, pipeline_name, pipeline_classes, storage)

        # Checked AFTER each pipeline (including the last/only one) but
        # BEFORE the progress save below — both to catch a cancellation
        # requested while the just-finished pipeline was running (a
        # single-pipeline job, or one cancelled during its final pipeline,
        # must not silently settle as COMPLETED — FR-003), and because the
        # progress save writes `job.status` from this stale in-memory copy
        # (still RUNNING). Saving first and checking after was confirmed
        # live to lose the race: the progress save clobbers a CANCELLED
        # status written concurrently by a /cancel request back to RUNNING,
        # an instant before the checkpoint's own fresh read — which then
        # correctly, but too late, sees RUNNING again.
        if _cancellation_requested(job.job_id, storage):
            logger.info(
                f"Job {job.job_id} cancelled after {pipeline_name} "
                f"({completed_count + 1}/{total})"
            )
            job.status = JobStatus.CANCELLED
            job.error_message = "Job cancelled by user request"
            job.completed_at = datetime.now()
            storage.save_job_metadata(job)
            return job

        job.progress_percentage = round((completed_count + 1) / total * 100, 1)
        storage.save_job_metadata(job)

    _settle_final_status(job, pipelines_to_run)
    job.completed_at = datetime.now()
    storage.save_job_metadata(job)
    return job


def _resolve_pipelines(job: BatchJob, pipeline_classes: dict[str, type]) -> list[str]:
    if not job.selected_pipelines:
        return list(pipeline_classes.keys())

    requested = job.selected_pipelines
    available = [p for p in requested if p in pipeline_classes]
    missing = [p for p in requested if p not in pipeline_classes]
    if missing:
        logger.warning(f"Job {job.job_id} requested unavailable pipelines: {missing}")
    if not available:
        raise ValueError(f"No requested pipelines are available: {requested}")
    return available


def _cancellation_requested(job_id: str, storage: StorageBackend) -> bool:
    """Re-read the job's own persisted status — not an in-memory flag — so a
    `/cancel` API call from a different request/thread/process is visible
    here. This is the checkpoint: cancellation is checked between
    pipelines, not mid-pipeline-call (see spec's Assumptions)."""
    try:
        current = storage.load_job_metadata(job_id)
    except Exception as e:
        logger.warning(f"Could not re-check status for job {job_id}: {e}")
        return False
    return current is not None and current.status == JobStatus.CANCELLED


def _run_one_pipeline(
    job: BatchJob,
    pipeline_name: str,
    pipeline_classes: dict[str, type],
    storage: StorageBackend,
) -> None:
    if storage.annotation_exists(job.job_id, pipeline_name):
        logger.info(f"Skipping {pipeline_name} for job {job.job_id} (already exists)")
        job.pipeline_results[pipeline_name] = PipelineResult(
            pipeline_name=pipeline_name, status=JobStatus.COMPLETED
        )
        return

    start_time = datetime.now()
    pipeline_config = job.config.get(pipeline_name, {}) if job.config else {}
    pipeline_class = pipeline_classes[pipeline_name]
    pipeline = pipeline_class(pipeline_config)

    try:
        pipeline.initialize()
        try:
            annotations = _process(pipeline, pipeline_name, pipeline_class, job)
        finally:
            try:
                pipeline.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Pipeline cleanup error: {cleanup_error}")

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        output_file = storage.save_annotations(job.job_id, pipeline_name, annotations)

        job.pipeline_results[pipeline_name] = PipelineResult(
            pipeline_name=pipeline_name,
            status=JobStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            processing_time=processing_time,
            annotation_count=len(annotations)
            if isinstance(annotations, list)
            else None,
            output_file=Path(output_file) if isinstance(output_file, str) else None,
        )
        logger.info(
            f"Completed {pipeline_name} for job {job.job_id} in {processing_time:.2f}s"
        )
    except Exception as e:
        logger.error(
            f"Pipeline {pipeline_name} failed for job {job.job_id}: {e}",
            exc_info=True,
        )
        job.pipeline_results[pipeline_name] = PipelineResult(
            pipeline_name=pipeline_name,
            status=JobStatus.FAILED,
            start_time=start_time,
            end_time=datetime.now(),
            error_message=str(e),
        )


def _process(pipeline: Any, pipeline_name: str, pipeline_class: type, job: BatchJob):
    # AudioPipelineModular's process() takes output_dir, not pps — matches
    # api/job_processor.py's existing special-case, carried over unchanged.
    if (
        pipeline_name in ("audio_processing", "audio")
        and pipeline_class.__name__ == "AudioPipelineModular"
    ):
        return pipeline.process(
            video_path=str(job.video_path),
            start_time=0,
            end_time=None,
            output_dir=str(job.output_dir),
        )
    return pipeline.process(
        video_path=str(job.video_path),
        start_time=0,
        end_time=None,
        pps=job.config.get("pps", 1) if job.config else 1,
        output_dir=str(job.output_dir),
    )


def _settle_final_status(job: BatchJob, pipelines_to_run: list[str]) -> None:
    failed = [
        name
        for name in pipelines_to_run
        if job.pipeline_results.get(name)
        and job.pipeline_results[name].status == JobStatus.FAILED
    ]
    if failed and len(failed) == len(pipelines_to_run):
        job.status = JobStatus.FAILED
        job.error_message = f"All pipelines failed: {', '.join(failed)}"
    elif failed:
        job.status = JobStatus.COMPLETED
        job.error_message = (
            f"Completed with errors. Failed pipelines: {', '.join(failed)}"
        )
    else:
        job.status = JobStatus.COMPLETED
