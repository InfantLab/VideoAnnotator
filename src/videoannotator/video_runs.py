"""Every run on one video, and the projects that hold it (spec 028 R2).

A video is its resolved server path (R1): every job whose `video_path`
resolves to it is one of its runs, whichever spelling or symlink the job was
given. A scan of the jobs, as `has_video_path` is; spec 029's video list will
index it.
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
from .results_folder import run_folder_of

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "batch_id": self.batch_id,
            "dataset_id": self.dataset_id,
            "run_name": self.run_name,
            "pipelines": [p.to_dict() for p in self.pipelines],
        }


def _resolved(path: Any) -> Path | None:
    try:
        return Path(str(path)).expanduser().resolve()
    except (OSError, RuntimeError):
        return None


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
    return RunInfo(
        job_id=job.job_id,
        status=getattr(job.status, "value", str(job.status)),
        created_at=job.created_at,
        completed_at=job.completed_at,
        batch_id=job.batch_id,
        dataset_id=job.dataset_id,
        run_name=folder.name if folder is not None else None,
        pipelines=pipelines,
    )


def runs_for(storage: Any, path: Path) -> list[RunInfo]:
    """Every job that read the video at `path` (resolved), newest first."""
    runs = []
    for job_id in storage.list_jobs():
        try:
            job = storage.load_job_metadata(job_id)
        except Exception:
            continue
        if job is None or not job.video_path or _resolved(job.video_path) != path:
            continue
        runs.append(_run_info(job))
    return sorted(runs, key=_created, reverse=True)


def _in_folder(path: Path, folder: Path, recursive: bool) -> bool:
    return path.parent == folder or (recursive and folder in path.parents)


def _manifest_lists(dataset: Any, path: Path, folder: Path) -> bool:
    relative = path.relative_to(folder).as_posix()
    for entry in dataset.video_manifest or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("relative_path"):
            if entry["relative_path"] == relative:
                return True
        elif entry.get("filename") == path.name:
            return True
    return False


def projects_for(db: Any, path: Path, runs: list[RunInfo]) -> list[dict[str, str]]:
    """The saved datasets that hold the video: by a run's `dataset_id`, or
    because the video is in the dataset's server folder (and, for a chosen
    subset of it, in its manifest)."""
    from .database.models import SavedDataset

    ids = {r.dataset_id for r in runs if r.dataset_id}
    found = []
    for dataset in db.query(SavedDataset).all():
        holds = str(dataset.id) in ids
        folder = _resolved(dataset.server_folder) if dataset.server_folder else None
        if not holds and folder is not None:
            holds = _in_folder(
                path, folder, bool(dataset.server_folder_recursive)
            ) and (
                not dataset.server_selection or _manifest_lists(dataset, path, folder)
            )
        if holds:
            found.append({"id": str(dataset.id), "name": dataset.name})
    return sorted(found, key=lambda d: d["name"].lower())


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
