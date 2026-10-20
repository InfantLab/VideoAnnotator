"""Jobs, videos, projects and workflows for spec 029's server tests.

Jobs go through the real `SQLiteStorageBackend`, so `job_summaries()` and the
index read what the server would. Uploads live in a job's own folder under a
storage root, and a rerun hard-links that copy into its own folder, as
`api/v1/jobs.py: job_from_stored_video` does.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from tests.fixtures.synthetic_video import synthetic_video_bytes_avi
from videoannotator.batch.types import BatchJob, JobStatus, PipelineResult
from videoannotator.storage.sqlite_backend import SQLiteStorageBackend

T0 = datetime(2026, 10, 12, 9, 0)

PipelineSpec = tuple[str, str | None, str | None]  # status, error, sha256


def video_bytes(tag: str = "") -> bytes:
    """A small real video; `tag` makes its bytes (and so its hash) differ."""
    return synthetic_video_bytes_avi() + tag.encode()


def write_video(path: Path, tag: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(video_bytes(tag) if tag else b"video " + path.name.encode())
    return path


class JobsDb:
    """A jobs database and a storage root in `tmp_path`."""

    def __init__(self, tmp_path: Path, storage: Any | None = None):
        self.tmp_path = tmp_path
        self.storage_root = tmp_path / "storage"
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.storage = storage or SQLiteStorageBackend(tmp_path / "jobs.db")
        self._minute = 0

    def _next_time(self) -> datetime:
        self._minute += 1
        return T0 + timedelta(minutes=self._minute)

    def make_job(
        self,
        video_path: Path | str,
        *,
        batch_id: str | None = None,
        batch_name: str | None = None,
        dataset_id: str | None = None,
        rerun_of: str | None = None,
        pipelines: dict[str, PipelineSpec] | None = None,
        created_at: datetime | None = None,
        status: JobStatus | None = None,
        storage_path: Path | None = None,
        job_id: str | None = None,
        config: dict[str, Any] | None = None,
        output_dir: Path | None = None,
        save: bool = True,
    ) -> BatchJob:
        pipelines = pipelines or {}
        results = {
            name: PipelineResult(
                name,
                JobStatus(state),
                error_message=error,
                provenance={"input": {"sha256": sha}} if sha else None,
            )
            for name, (state, error, sha) in pipelines.items()
        }
        if status is None:
            states = {r.status for r in results.values()}
            status = (
                JobStatus.RUNNING
                if JobStatus.RUNNING in states
                else JobStatus.PENDING
                if JobStatus.PENDING in states or not states
                else JobStatus.COMPLETED
            )
        created = created_at or self._next_time()
        finished = status in (JobStatus.COMPLETED, JobStatus.FAILED)
        job = BatchJob(
            job_id=job_id or str(uuid.uuid4()),
            video_path=Path(video_path),
            status=status,
            config=config or {},
            created_at=created,
            completed_at=created + timedelta(seconds=30) if finished else None,
            selected_pipelines=list(pipelines),
            pipeline_results=results,
            batch_id=batch_id,
            batch_name=batch_name,
            dataset_id=dataset_id,
            rerun_of=rerun_of,
            storage_path=storage_path,
            output_dir=output_dir,
        )
        if save:
            self.storage.save_job_metadata(job)
        return job

    def upload(self, name: str, *, tag: str = "", **job: Any) -> tuple[BatchJob, Path]:
        """A job whose video was uploaded with it: stored in its own folder."""
        job_id = str(uuid.uuid4())
        folder = self.storage_root / job_id
        video = write_video(folder / name, tag)
        return (
            self.make_job(video, job_id=job_id, storage_path=folder, **job),
            video,
        )

    def rerun(self, original: BatchJob, **job: Any) -> tuple[BatchJob, Path]:
        """A rerun of `original`'s stored upload, hard-linked into its own folder."""
        job_id = str(uuid.uuid4())
        folder = self.storage_root / job_id
        folder.mkdir(parents=True)
        source = Path(str(original.video_path))
        target = folder / source.name
        os.link(source, target)
        return (
            self.make_job(
                target,
                job_id=job_id,
                storage_path=folder,
                rerun_of=original.job_id,
                **job,
            ),
            target,
        )


def make_dataset(
    db: Any,
    name: str,
    *,
    manifest: list[dict[str, Any]] | None = None,
    server_folder: Path | str | None = None,
    recursive: bool = False,
    selection: bool = False,
    owner: str = "owner-1",
    current_settings: dict[str, Any] | None = None,
) -> Any:
    from videoannotator.database.crud import SavedDatasetCRUD

    dataset = SavedDatasetCRUD.create(
        db,
        owner_user_id=owner,
        name=name,
        video_manifest=manifest or [],
        server_folder=str(server_folder) if server_folder else None,
        server_folder_recursive=recursive,
        server_selection=selection,
        current_settings=current_settings,
    )
    assert dataset is not None, f"a dataset named {name!r} exists already"
    return dataset


def make_preset(
    db: Any,
    name: str,
    *,
    pipelines: list[str] | None = None,
    config: dict[str, Any] | None = None,
    owner: str = "owner-1",
    export_choices: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
) -> Any:
    from videoannotator.database.crud import SavedPipelinePresetCRUD

    preset = SavedPipelinePresetCRUD.create(
        db,
        owner_user_id=owner,
        name=name,
        selected_pipelines=pipelines or ["face_analysis"],
        config=config or {},
        export_choices=export_choices,
        source=source,
    )
    assert preset is not None, f"a preset named {name!r} exists already"
    return preset
