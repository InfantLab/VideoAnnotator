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

from ..prompt_library import record_use_quietly
from ..provenance import build_record, file_sha256, stamp_file
from ..registry.pipeline_loader import (
    deprecation_message,
    import_error_for,
    removed_pipeline_message,
)
from ..registry.pipeline_registry import get_registry
from ..storage.base import StorageBackend
from ..utils.torch_settings import apply_torch_settings, restored_torch_settings
from .result_files import job_folder
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
    # A /cancel can land while the job waits on a cold-start pipeline import
    # (minutes); setting RUNNING below would otherwise overwrite it.
    if _cancellation_requested(job.job_id, storage):
        job.status = JobStatus.CANCELLED
        job.error_message = "Job cancelled by user request"
        job.completed_at = datetime.now()
        storage.save_job_metadata(job)
        return job

    job.status = JobStatus.RUNNING
    job.started_at = job.started_at or datetime.now()
    storage.save_job_metadata(job)

    if job.output_dir is None and job.storage_path:
        job.output_dir = job.storage_path
    if job.output_dir is None:
        from ..storage.config import get_job_storage_path

        job.output_dir = get_job_storage_path(job.job_id) / "output"
    job.output_dir.mkdir(parents=True, exist_ok=True)

    pipelines_to_run = _resolve_pipelines(job, pipeline_classes)
    if job.pipeline_results is None:
        job.pipeline_results = {}
    unavailable = [
        p for p in (job.selected_pipelines or []) if p not in pipeline_classes
    ]
    for name in unavailable:
        job.pipeline_results[name] = PipelineResult(
            pipeline_name=name,
            status=JobStatus.FAILED,
            error_message=_unavailable_reason(name),
        )

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

    _settle_final_status(job, unavailable + pipelines_to_run)
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


def _unavailable_reason(pipeline_name: str) -> str:
    removed = removed_pipeline_message(pipeline_name)
    if removed:
        return removed
    reason = import_error_for(pipeline_name)
    if reason:
        return f"Pipeline not available on this server: {reason}"
    return "Pipeline not available on this server (not installed or not registered)"


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

    meta = get_registry().get(pipeline_name)
    deprecation = deprecation_message(meta) if meta else None
    if deprecation:
        logger.warning(f"Job {job.job_id}: {deprecation}")

    start_time = datetime.now()
    pipeline_config = job.config.get(pipeline_name, {}) if job.config else {}
    pipeline_class = pipeline_classes[pipeline_name]
    pipeline = pipeline_class(pipeline_config)

    deterministic = (
        bool(job.config.get("deterministic", False)) if job.config else False
    )

    record: dict[str, Any] | None = None
    try:
        with restored_torch_settings():
            apply_torch_settings(deterministic)
            pipeline.initialize()
            # Again: a library's setup can change them (OpenFace 3 did).
            settings = apply_torch_settings(deterministic)
            if settings:
                logger.info(f"{pipeline_name} runs with torch settings {settings}")
            record = _provenance(job, pipeline_name, pipeline, settings, deterministic)
            if record.get("vlm"):  # spec 020: every prompt that runs is kept
                record_use_quietly(
                    pipeline.config["prompt"],
                    pipeline.config["model"],
                    "job",
                    job_id=job.job_id,
                )
            try:
                annotations = _process(pipeline, pipeline_name, pipeline_class, job)
                # Some load a model only when it's first needed (scene's CLIP).
                record["models"] = [m.to_dict() for m in _models(pipeline)]
            finally:
                try:
                    pipeline.cleanup()
                except Exception as cleanup_error:
                    logger.warning(f"Pipeline cleanup error: {cleanup_error}")
        _stamp_outputs(job, pipeline_name, pipeline, record)

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
            provenance=record,
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
            provenance=record,
        )


def _provenance(
    job: BatchJob,
    pipeline_name: str,
    pipeline: Any,
    torch_settings: dict[str, Any],
    deterministic: bool,
) -> dict[str, Any]:
    video = Path(job.video_path) if job.video_path else None
    provenance_vlm = getattr(pipeline, "provenance_vlm", None)
    return build_record(
        pipeline_name,
        models=_models(pipeline),
        settings=getattr(pipeline, "config", {}),
        determinism={"deterministic": deterministic, **torch_settings}
        if torch_settings
        else {},
        job_id=job.job_id,
        input_name=video.name if video else None,
        input_sha256=file_sha256(video) if video and video.is_file() else None,
        vlm=provenance_vlm() if callable(provenance_vlm) else None,
    )


def _models(pipeline: Any) -> list[Any]:
    # Pipelines that don't subclass BasePipeline (plugins) may not report any.
    provenance_models = getattr(pipeline, "provenance_models", None)
    return list(provenance_models()) if callable(provenance_models) else []


def _stamp_outputs(
    job: BatchJob, pipeline_name: str, pipeline: Any, record: dict[str, Any] | None
) -> None:
    """Record `record` in every file the pipeline wrote (spec 017)."""
    meta = get_registry().get(pipeline_name)
    if record is None or meta is None or job.video_path is None:
        return
    stem = Path(job.video_path).stem
    sub_pipelines = getattr(pipeline, "audio_pipelines", {})
    for output in meta.outputs:
        if not output.file:
            continue
        path = job_folder(job) / f"{stem}_{output.file}"
        if not path.is_file():
            continue
        sub = Path(output.file).stem
        file_record = (
            {**record, "pipeline": {"name": pipeline_name, "sub_pipeline": sub}}
            if sub in sub_pipelines
            else record
        )
        try:
            stamp_file(path, file_record)
        except Exception as e:  # a good result shouldn't fail on its label
            logger.error(f"Could not record provenance in {path}: {e}")


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
    # Each failure's own error goes in the job-level message: clients that show
    # one message per job (the viewer's status tooltip, batch rows) otherwise
    # say *which* pipelines failed but never *why*.
    reasons = "; ".join(
        f"{name}: {job.pipeline_results[name].error_message or 'unknown error'}"
        for name in failed
    )
    if failed and len(failed) == len(pipelines_to_run):
        job.status = JobStatus.FAILED
        job.error_message = f"All pipelines failed. {reasons}"
    elif failed:
        job.status = JobStatus.COMPLETED
        job.error_message = f"Completed with errors. {reasons}"
    else:
        job.status = JobStatus.COMPLETED
