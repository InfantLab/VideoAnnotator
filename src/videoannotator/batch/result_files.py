"""Where a job's per-pipeline output files are."""

from __future__ import annotations

from pathlib import Path

from ..registry.pipeline_registry import get_registry
from .types import BatchJob


def job_folder(job: BatchJob) -> Path:
    """The folder a job's pipelines write to."""
    from ..storage.config import get_job_storage_path

    return Path(job.output_dir or job.storage_path or get_job_storage_path(job.job_id))


def pipeline_result_files(job: BatchJob, pipeline_name: str) -> list[Path]:
    """The files `pipeline_name` wrote to the job folder, main output first.

    Pipelines write `<video stem>_<suffix>` there; the suffixes come from each
    pipeline's registry metadata (`outputs[].file`). The storage backend's
    `output_file` can't be used: for the database backend it isn't a path.
    """
    meta = get_registry().get(pipeline_name)
    if meta is None or job.video_path is None:
        return []
    folder = job_folder(job)
    stem = Path(job.video_path).stem
    files = [folder / f"{stem}_{o.file}" for o in meta.outputs if o.file]
    return [f for f in files if f.is_file()]
