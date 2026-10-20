"""Every run on one video (spec 028 R2).

A video is every path the server can tell holds one file (spec 029 R1): every
job whose `video_path` is one of them is one of its runs, whichever spelling,
symlink or stored copy the job was given. The projects that hold it come from
`projects.py`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .batch.result_files import pipeline_result_files
from .exporters.readers import classify
from .provenance import COMPANION_SUFFIX
from .results_folder import folder_ref, run_folder_of
from .video_index import Video, cached_index

logger = logging.getLogger(__name__)


@dataclass
class RunFile:
    name: str
    kind: str  # result | provenance | other
    download_url: str
    path: Path

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "download_url": self.download_url}


@dataclass
class PipelineRunInfo:
    name: str
    status: str
    error: str | None
    files: list[RunFile] = field(default_factory=list)
    provenance: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "error": self.error,
            "files": [f.to_dict() for f in self.files],
        }


@dataclass
class RunInfo:
    job_id: str
    status: str
    created_at: datetime | None
    completed_at: datetime | None
    batch_id: str | None
    dataset_id: str | None
    run_name: str | None
    pipelines: list[PipelineRunInfo] = field(default_factory=list)
    batch_name: str | None = None
    rerun_of: str | None = None
    progress: float = 0.0
    queue_position: int | None = None
    error: str | None = None
    selected_pipelines: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    results_folder: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_name": self.batch_name,
            "rerun_of": self.rerun_of,
            "progress": self.progress,
            "queue_position": self.queue_position,
            "error": self.error,
            "selected_pipelines": self.selected_pipelines,
            "config": self.config,
            "results_folder": self.results_folder,
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "batch_id": self.batch_id,
            "dataset_id": self.dataset_id,
            "run_name": self.run_name,
            "pipelines": [p.to_dict() for p in self.pipelines],
        }


def _file_kind(path: Path) -> str:
    if path.name.lower().endswith(COMPANION_SUFFIX):
        return "provenance"
    kind, _ = classify(path)
    return "result" if kind is not None else "other"


def _created(run: RunInfo) -> float:
    return run.created_at.timestamp() if run.created_at else 0.0


def _run_info(job: Any) -> RunInfo:
    folder = run_folder_of(job)
    pipelines = []
    for name, result in (job.pipeline_results or {}).items():
        url = f"/api/v1/jobs/{job.job_id}/results/files/{quote(name)}"
        try:
            paths = pipeline_result_files(job, name)
        except Exception as e:  # registry or folder unreadable: list no files
            logger.debug(f"Could not list {name}'s files for job {job.job_id}: {e}")
            paths = []
        pipelines.append(
            PipelineRunInfo(
                name=name,
                status=getattr(result.status, "value", str(result.status)),
                error=result.error_message,
                files=[
                    RunFile(
                        name=p.name,
                        kind=_file_kind(p),
                        download_url=f"{url}?name={quote(p.name)}",
                        path=p,
                    )
                    for p in paths
                ],
                provenance=result.provenance,
            )
        )
    results = folder_ref(job.output_dir)
    if results is not None:
        results["exists"] = Path(results["path"]).is_dir()
    return RunInfo(
        job_id=job.job_id,
        status=getattr(job.status, "value", str(job.status)),
        created_at=job.created_at,
        completed_at=job.completed_at,
        batch_id=job.batch_id,
        dataset_id=job.dataset_id,
        run_name=folder.name if folder is not None else None,
        pipelines=pipelines,
        batch_name=job.batch_name,
        rerun_of=job.rerun_of,
        progress=float(job.progress_percentage or 0),
        error=job.error_message,
        selected_pipelines=list(job.selected_pipelines or []),
        config=dict(job.config or {}),
        results_folder=results,
    )


def video_of(storage: Any, path: Path | str) -> Video | None:
    """The video (every path of one file, spec 029 R1) that `path` is one of."""
    from .storage.config import get_storage_root

    return cached_index(storage, uploads_root=get_storage_root()).video_for(path)


def runs_for(storage: Any, path: Path | str) -> list[RunInfo]:
    """Every job that read the video at `path`, under any of its paths,
    newest first."""
    video = video_of(storage, path)
    if video is None:
        return []
    runs = []
    for summary in video.jobs:
        try:
            job = storage.load_job_metadata(summary.job_id)
        except Exception:
            continue
        if job is not None:
            runs.append(_run_info(job))
    if any(r.status == "pending" for r in runs):
        waiting = storage.list_jobs(status_filter="pending")
        for run in runs:
            if run.status == "pending" and run.job_id in waiting:
                run.queue_position = waiting.index(run.job_id) + 1
    return sorted(runs, key=_created, reverse=True)


def _sha256(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    sha = (record.get("input") or {}).get("sha256")
    return sha if isinstance(sha, str) and sha else None


def sha256_for(runs: list[RunInfo]) -> str | None:
    """The video's content hash as a run's provenance recorded it, else None.
    Never computed here: hashing a large video takes too long for a request."""
    for run in runs:
        for pipeline in run.pipelines:
            sha = _sha256(pipeline.provenance)
            if sha:
                return sha
    for run in runs:
        for pipeline in run.pipelines:
            for f in pipeline.files:
                if f.kind != "provenance":
                    continue
                try:
                    sha = _sha256(json.loads(f.path.read_text(encoding="utf-8")))
                except (OSError, ValueError):
                    continue
                if sha:
                    return sha
    return None
