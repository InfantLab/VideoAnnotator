"""Run one job in this process, without a server (`videoannotator process`).

The job is checked the way the API checks a submission, recorded in the same
database as the server's jobs (so it appears in the viewer and in
`videoannotator job list`), and run through `run_job_pipelines`, the one
job-execution path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..api.database import get_storage_backend
from ..api.v1.jobs import validate_pipeline_selection
from ..registry.pipeline_loader import get_pipeline_loader, removed_pipeline_message
from ..registry.pipeline_registry import get_registry
from ..results_folder import ResultsFolderError, RunFolder, run_name
from ..storage.manager import get_storage_provider
from .job_execution import run_job_pipelines
from .result_files import pipeline_result_files
from .types import BatchJob, JobStatus


class LocalJobError(Exception):
    """A job that can't start, with a hint on what to do instead."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


def check_pipelines(pipelines: list[str] | None, config: dict[str, Any] | None) -> None:
    """Reject names the registry doesn't know before any work starts.

    The server leaves an unknown name to fail inside the job; for a command
    run by hand, failing at once with the valid names is more useful.
    """
    from ..api.v1.exceptions import VideoAnnotatorException

    registry = get_registry()
    registry.load()
    known = sorted(m.name for m in registry.list())
    for name in pipelines or []:
        if registry.get(name) is None and not removed_pipeline_message(name):
            raise LocalJobError(
                f"Unknown pipeline '{name}'",
                hint=f"Choose from: {', '.join(known)}",
            )
    try:
        validate_pipeline_selection(pipelines, config)
    except VideoAnnotatorException as e:
        raise LocalJobError(e.message, e.hint) from e


def run_local_job(
    video: Path,
    pipelines: list[str] | None,
    config: dict[str, Any] | None = None,
) -> BatchJob:
    """Create a job for `video`, run it here, and return it finished."""
    check_pipelines(pipelines, config)

    storage = get_storage_backend()
    provider = get_storage_provider()
    # Saved as RUNNING, never PENDING: a server sharing this database polls
    # for pending jobs and would otherwise run it a second time.
    # The video is read where it is (spec 022): no link or copy, and results
    # go to the results folder, in a run named after the video.
    video = video.resolve()
    job = BatchJob(
        video_path=video,
        config=config or {},
        status=JobStatus.RUNNING,
        selected_pipelines=pipelines,
    )
    try:
        run = RunFolder.create(
            run_name(video=video), batch_id=None, pipelines=pipelines, config=config
        )
    except ResultsFolderError as e:
        raise LocalJobError(
            str(e), hint="Set VIDEOANNOTATOR_RESULTS_DIR to a folder you can write to."
        ) from e
    run.add_video(job, video)
    provider.create_job_dir(job.job_id)
    job.storage_path = provider.get_absolute_path(job.job_id, "")
    storage.save_job_metadata(job)

    pipeline_classes = get_pipeline_loader().load_all_pipelines()
    return run_job_pipelines(job, storage, pipeline_classes)


def result_files(job: BatchJob) -> dict[str, list[Path]]:
    """Each pipeline's output files, main output first."""
    return {name: pipeline_result_files(job, name) for name in job.pipeline_results}
