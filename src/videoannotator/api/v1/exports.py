"""Results out: the table, ELAN files and the methods paragraph (spec 025).

Every route resolves its job, run or dataset to jobs, then hands their folders
to the same exporters the CLI runs on a folder, so both give the same result
(FR-010). Exports are built in a worker thread, into a temporary file, so a
large run neither blocks the server nor has to fit in memory.

Contract: specs/025-results-out/contracts/api.md.
"""

from __future__ import annotations

import os
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path as FilePath
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, Path, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from sqlalchemy.orm import Session
from starlette.background import BackgroundTask

from ...database.crud import SavedDatasetCRUD
from ...database.database import get_db
from ...exporters.elan import file_name, write_videos
from ...exporters.methods import describe
from ...exporters.results_set import ResultsSet, from_jobs
from ...exporters.table import (
    PARQUET_EXTRA,
    PARQUET_HINT,
    ExportSummary,
    parquet_available,
    write_csv,
    write_parquet,
)
from ...results_folder import run_folder_of, sanitize_component
from ...storage.base import StorageBackend
from ..database import get_storage_backend
from ..errors import APIError
from ..middleware.auth import validate_api_key

router = APIRouter()

SUMMARY_HEADER = "X-VideoAnnotator-Export-Summary"
TABLE_TYPES = {
    "csv": "text/csv; charset=utf-8",
    "parquet": "application/vnd.apache.parquet",
}


def get_storage() -> StorageBackend:
    return get_storage_backend()


def _load(storage: StorageBackend, job_id: str) -> Any:
    # Backends differ: the database one raises for an unknown id.
    try:
        return storage.load_job_metadata(job_id)
    except FileNotFoundError:
        return None


def _jobs(storage: StorageBackend, job_ids: list[str]) -> list[Any]:
    jobs = (_load(storage, job_id) for job_id in job_ids)
    return [job for job in jobs if job is not None]


def _job_or_404(storage: StorageBackend, job_id: str) -> Any:
    job = _load(storage, job_id)
    if job is None:
        raise APIError(
            status_code=404,
            code="JOB_NOT_FOUND",
            message=f"Job {job_id} not found",
            hint="Check job ID or use GET /api/v1/jobs to list all jobs",
        )
    return job


def _batch_or_404(storage: StorageBackend, batch_id: str) -> list[Any]:
    jobs = _jobs(storage, storage.list_jobs_by_batch(batch_id))
    if not jobs:
        raise APIError(
            status_code=404,
            code="BATCH_NOT_FOUND",
            message=f"Batch '{batch_id}' not found",
            hint="Check the batch ID, or use GET /api/v1/batches.",
        )
    return jobs


def _results_or_409(jobs: list[Any], what: str) -> ResultsSet:
    results = from_jobs(jobs)
    if results.is_empty():
        reasons = "; ".join(f"{m.video}: {m.reason}" for m in results.missing)
        raise APIError(
            status_code=409,
            code="NOTHING_TO_EXPORT",
            message=f"{what} has no results to export"
            + (f" ({reasons})" if reasons else ""),
        )
    return results


def _run_name(jobs: list[Any]) -> str:
    run = next((f for f in map(run_folder_of, jobs) if f is not None), None)
    if run is not None:
        return run.name
    return sanitize_component(next((j.batch_name for j in jobs if j.batch_name), "Run"))


def _video_name(job: Any) -> str:
    if run_folder_of(job) is not None and job.output_dir:
        return FilePath(job.output_dir).name
    stem = FilePath(job.video_path).stem if job.video_path else job.job_id
    return sanitize_component(stem, "video")


def _disposition(filename: str) -> str:
    ascii_name = filename.encode("ascii", "replace").decode().replace('"', "")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


async def _file_response(
    write: Callable[[FilePath], ExportSummary],
    suffix: str,
    filename: str,
    media_type: str,
) -> FileResponse:
    """Run `write` into a temporary file in a worker thread and send it."""
    handle, name = tempfile.mkstemp(suffix=suffix, prefix="va-export-")
    os.close(handle)
    path = FilePath(name)
    try:
        summary = await run_in_threadpool(write, path)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return FileResponse(
        path,
        media_type=media_type,
        headers={
            "Content-Disposition": _disposition(filename),
            SUMMARY_HEADER: summary.to_header(),
        },
        background=BackgroundTask(path.unlink, missing_ok=True),
    )


def _check_format(format: str) -> None:
    if format not in TABLE_TYPES:
        raise APIError(
            status_code=422,
            code="INVALID_EXPORT_FORMAT",
            message=f"Unknown format '{format}'",
            hint="Use format=csv or format=parquet",
        )
    if format == "parquet" and not parquet_available():
        raise APIError(
            status_code=422,
            code="PARQUET_NOT_INSTALLED",
            message="Parquet export needs the optional 'parquet' component",
            hint=PARQUET_HINT,
            details={"extra": PARQUET_EXTRA},
        )


async def _table(results: ResultsSet, format: str, name: str) -> FileResponse:
    def write(path: FilePath) -> ExportSummary:
        if format == "parquet":
            return write_parquet(results, path)
        with open(path, "w", encoding="utf-8", newline="") as out:
            return write_csv(results, out)

    return await _file_response(
        write, f".{format}", f"{name}.{format}", TABLE_TYPES[format]
    )


_FORMAT = Query("csv", description="csv (always available) or parquet (optional extra)")
_TABLE_DESCRIPTION = """
One table of every pipeline's results: one row per event and measure, the same
columns for every pipeline (spec 025, `contracts/table.md`). The export summary
(videos and pipelines covered, files skipped and why, videos without results) is
in the `X-VideoAnnotator-Export-Summary` header as JSON.
"""


@router.get(
    "/jobs/{job_id}/export/table",
    summary="Export a job's results as one table",
    description=_TABLE_DESCRIPTION,
    response_class=FileResponse,
)
async def export_job_table(
    job_id: str = Path(..., description="The job's id"),
    format: str = _FORMAT,
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> FileResponse:
    _check_format(format)
    job = _job_or_404(storage, job_id)
    results = _results_or_409([job], f"Job {job_id}")
    return await _table(results, format, _video_name(job))


@router.get(
    "/batches/{batch_id}/export/table",
    summary="Export a run's results as one table",
    description=_TABLE_DESCRIPTION,
    response_class=FileResponse,
)
async def export_batch_table(
    batch_id: str = Path(..., description="The run's batch identifier"),
    format: str = _FORMAT,
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> FileResponse:
    _check_format(format)
    jobs = _batch_or_404(storage, batch_id)
    results = _results_or_409(jobs, "This run")
    return await _table(results, format, _run_name(jobs))


@router.get(
    "/datasets/{dataset_id}/export/table",
    summary="Export every run of a saved dataset as one table",
    description=_TABLE_DESCRIPTION
    + "\nEvery run made from the dataset is included; the `run` column tells them apart.",
    response_class=FileResponse,
)
async def export_dataset_table(
    dataset_id: str = Path(..., description="The saved dataset's id"),
    format: str = _FORMAT,
    storage: StorageBackend = Depends(get_storage),
    db: Session = Depends(get_db),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> FileResponse:
    _check_format(format)
    dataset = SavedDatasetCRUD.get_by_id(db, dataset_id)
    if dataset is None:
        raise APIError(
            status_code=404,
            code="DATASET_NOT_FOUND",
            message=f"Dataset '{dataset_id}' not found",
        )
    jobs = _jobs(storage, storage.list_jobs_by_dataset(dataset_id))
    if not jobs:
        raise APIError(
            status_code=409,
            code="NOTHING_TO_EXPORT",
            message=f"Dataset '{dataset.name}' hasn't been run yet",
        )
    results = _results_or_409(jobs, f"Dataset '{dataset.name}'")
    return await _table(results, format, sanitize_component(dataset.name, "Dataset"))


_METHODS_DESCRIPTION = """
A methods paragraph for a paper (spec 025): each pipeline, the models with their
exact weight revisions, settings that differ from the defaults, VideoAnnotator's
version, and citations, as prose with `[@key]` citations plus BibTeX. Built only
from the provenance recorded in the output files; anything not recorded is said
to be. `format=markdown` returns the same as one Markdown file.
"""


async def _methods(results: ResultsSet, format: str) -> Any:
    if format not in ("json", "markdown"):
        raise APIError(
            status_code=422,
            code="INVALID_EXPORT_FORMAT",
            message=f"Unknown format '{format}'",
            hint="Use format=json or format=markdown",
        )
    described = await run_in_threadpool(describe, results)
    if format == "markdown":
        return Response(
            described.to_markdown(),
            media_type="text/markdown; charset=utf-8",
            headers={"Content-Disposition": _disposition("methods.md")},
        )
    return JSONResponse(described.to_dict())


_METHODS_FORMAT = Query("json", description="json, or markdown for one .md file")


@router.get(
    "/jobs/{job_id}/methods",
    summary="A methods paragraph for a job's results",
    description=_METHODS_DESCRIPTION,
)
async def job_methods(
    job_id: str = Path(..., description="The job's id"),
    format: str = _METHODS_FORMAT,
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> Any:
    job = _job_or_404(storage, job_id)
    return await _methods(_results_or_409([job], f"Job {job_id}"), format)


@router.get(
    "/batches/{batch_id}/methods",
    summary="A methods paragraph for a run's results",
    description=_METHODS_DESCRIPTION
    + "\nVideos processed the same way are described once, with their number.",
)
async def batch_methods(
    batch_id: str = Path(..., description="The run's batch identifier"),
    format: str = _METHODS_FORMAT,
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> Any:
    jobs = _batch_or_404(storage, batch_id)
    return await _methods(_results_or_409(jobs, "This run"), format)


_ELAN_DESCRIPTION = """
ELAN (`.eaf`, EAF 3.0) with one tier per pipeline track (spec 025): speech,
speaker turns, scenes, people and faces in view, VLM labels. Per-detection
measures stay in the table; the summary header's `left_out` says which.
"""


@router.get(
    "/jobs/{job_id}/export/elan",
    summary="Export a job's results as an ELAN file",
    description=_ELAN_DESCRIPTION,
    response_class=FileResponse,
)
async def export_job_elan(
    job_id: str = Path(..., description="The job's id"),
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> FileResponse:
    job = _job_or_404(storage, job_id)
    results = _results_or_409([job], f"Job {job_id}")

    def write(path: FilePath) -> ExportSummary:
        return write_videos(results, lambda _video: path)

    return await _file_response(
        write, ".eaf", f"{_video_name(job)}.eaf", "application/xml"
    )


@router.get(
    "/batches/{batch_id}/export/elan",
    summary="Export a run's results as ELAN files",
    description=_ELAN_DESCRIPTION + "\nOne `.eaf` per video, as a zip.",
    response_class=FileResponse,
)
async def export_batch_elan(
    batch_id: str = Path(..., description="The run's batch identifier"),
    storage: StorageBackend = Depends(get_storage),
    _user: dict[str, Any] | None = Depends(validate_api_key),
) -> FileResponse:
    jobs = _batch_or_404(storage, batch_id)
    results = _results_or_409(jobs, "This run")
    run = _run_name(jobs)

    def write(path: FilePath) -> ExportSummary:
        with tempfile.TemporaryDirectory(prefix="va-elan-") as folder:
            base = FilePath(folder)
            summary = write_videos(
                results,
                lambda video: (
                    base / sanitize_component(video.run or run) / file_name(video)
                ),
            )
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                for eaf in sorted(base.rglob("*.eaf")):
                    archive.write(eaf, eaf.relative_to(base).as_posix())
        return summary

    return await _file_response(write, ".zip", f"{run} ELAN.zip", "application/zip")
