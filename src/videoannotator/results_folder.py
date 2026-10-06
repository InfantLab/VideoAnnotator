"""The results folder: every run's results, by run then video (spec 022).

New jobs write their outputs to `<results root>/<run> (<date>)/<video>/` (the
job's `output_dir`), so a researcher finds them by name instead of behind a
random job ID in a hidden application folder. Each run folder also holds
`run.json`, an index of what ran on which videos.

Every naming and layout rule lives here, so the five ways a run is created
(upload, folder ingest, dataset run, rerun, CLI `process`) can't drift apart.
Two rules are enforced by construction rather than by checks callers might
forget: folders are created exclusively, so existing results are never
overwritten (FR-024), and a video folder never holds a video (FR-020).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from .config_env import host_paths, results_dir

logger = logging.getLogger(__name__)

RUN_RECORD = "run.json"
RUN_RECORD_FORMAT = "videoannotator-run"
MAX_COMPONENT = 120

_INVALID = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

# One lock for run records and batch → run-folder lookups: both are tiny,
# infrequent writes, and one lock can't deadlock against another.
_lock = threading.RLock()
_batch_run_folders: dict[str, Path] = {}


class ResultsFolderError(Exception):
    """The results folder can't be used; names the folder and why."""

    def __init__(self, folder: Path, reason: str):
        super().__init__(f"Can't write results to {display_path(folder)}: {reason}")
        self.folder = folder
        self.reason = reason


def results_root() -> Path:
    """Where every run's results go."""
    return results_dir()


def display_path(path: Path | str) -> str:
    """`path` as the researcher would find it on their own machine.

    Under Docker the server sees `/results/...`; the researcher's file browser
    shows the host folder mounted there.
    """
    text = str(path)
    for container, host in host_paths():
        if text == container or text.startswith(container + "/"):
            return host + text[len(container) :]
    return text


def folder_ref(path: Path | str | None) -> dict[str, str] | None:
    """`{"path", "display_path"}` for API responses, or None."""
    if path is None:
        return None
    return {"path": str(path), "display_path": display_path(path)}


def sanitize_component(name: str, fallback: str = "Run") -> str:
    """`name` made safe as one folder name on every OS.

    Drops characters Windows forbids, trailing dots and spaces (Windows strips
    them, so two names could collide), and avoids reserved device names.
    """
    cleaned = _INVALID.sub("", name).strip().rstrip(". ")
    cleaned = cleaned[:MAX_COMPONENT].rstrip(". ")
    if not cleaned:
        return fallback
    if cleaned.split(".")[0].upper() in _RESERVED:
        cleaned = "_" + cleaned
    return cleaned


def run_name(
    *,
    batch_name: str | None = None,
    dataset_name: str | None = None,
    source_folder: Path | None = None,
    video: Path | None = None,
    now: datetime | None = None,
) -> str:
    """The run's name: the first of these that is set (research.md R2)."""
    for candidate in (
        batch_name,
        dataset_name,
        source_folder.name if source_folder else None,
        video.stem if video else None,
    ):
        if candidate and candidate.strip():
            return candidate.strip()
    return f"Run {(now or datetime.now()).strftime('%Y-%m-%d %H-%M')}"


def check_writable(root: Path | None = None) -> Path:
    """Confirm results can be written under `root` before a run starts (FR-026)."""
    root = root or results_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=root, prefix=".write-check-"):
            pass
    except OSError as e:
        raise ResultsFolderError(root, e.strerror or str(e)) from e
    return root


def results_root_overlaps(configured_roots: list[Path]) -> Path | None:
    """The explicitly configured video folder the results root lies in, if any.

    The default video folder (home) always contains the default results folder,
    and that is fine: results are only ever written into their own run
    folders. A video folder someone named on purpose is different -- writing
    inside it mixes derived results into raw data (FR-021).
    """
    root = results_root()
    for allowed in configured_roots:
        if root == allowed or allowed in root.parents:
            return allowed
    return None


def is_inside_results(path: Path) -> bool:
    """Whether `path` (resolved) lies inside the results root."""
    try:
        return path.resolve().is_relative_to(results_root())
    except OSError:
        return False


@dataclass
class RunFolder:
    """One run's folder, with `run.json` and one folder per video."""

    path: Path
    source_root: Path | None = None

    @classmethod
    def create(
        cls,
        name: str,
        *,
        batch_id: str | None,
        pipelines: list[str] | None,
        config: dict[str, Any] | None,
        source_root: Path | None = None,
        today: date | None = None,
    ) -> RunFolder:
        """Create `<root>/<name> (<date>)`, numbering it if that exists."""
        root = check_writable()
        stamp = (today or date.today()).isoformat()
        base = sanitize_component(name)[: MAX_COMPONENT - len(stamp) - 8].rstrip(". ")
        number = 1
        while True:
            suffix = stamp if number == 1 else f"{stamp} {number}"
            path = root / f"{base} ({suffix})"
            try:
                path.mkdir()
                break
            except FileExistsError:
                number += 1
            except OSError as e:
                raise ResultsFolderError(root, e.strerror or str(e)) from e

        from .version import __version__

        _write_record(
            path,
            {
                "format": RUN_RECORD_FORMAT,
                "format_version": 1,
                "run": {
                    "batch_id": batch_id,
                    "name": name,
                    "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                "videoannotator_version": __version__,
                "pipelines": list(pipelines or []),
                "config": _redacted(config),
                "videos": [],
            },
        )
        return cls(path=path, source_root=source_root)

    @classmethod
    def for_batch(cls, batch_id: str, create: Callable[[], RunFolder]) -> RunFolder:
        """The run folder for `batch_id`, created by `create` only once.

        Uploads arrive one job per request, possibly in parallel, but a batch
        is one run: without this, two first uploads could each make a folder.
        """
        with _lock:
            known = _batch_run_folders.get(batch_id)
            if known is not None and known.is_dir():
                return cls(path=known)
            from .api.database import get_storage_backend

            storage = get_storage_backend()
            for job_id in storage.list_jobs_by_batch(batch_id):
                job = storage.load_job_metadata(job_id)
                folder = Path(job.output_dir).parent if job and job.output_dir else None
                if folder is not None and (folder / RUN_RECORD).is_file():
                    _batch_run_folders[batch_id] = folder
                    return cls(path=folder)
            run = create()
            _batch_run_folders[batch_id] = run.path
            return run

    def add_video(
        self,
        job: Any,
        video: Path,
        *,
        uploaded_name: str | None = None,
        size_bytes: int | None = None,
    ) -> Path:
        """Give `job` its own video folder here and record it in `run.json`.

        Named after the video. A second video with the same name gets its path
        within the source folder (`site_a__child01`), then a number.
        """
        name = Path(uploaded_name).stem if uploaded_name else video.stem
        candidates = [sanitize_component(name, "video")]
        if self.source_root is not None and video.is_relative_to(self.source_root):
            relative = video.relative_to(self.source_root).with_suffix("")
            if len(relative.parts) > 1:
                candidates.append(
                    sanitize_component("__".join(relative.parts), "video")
                )
        candidates += [f"{candidates[-1]} {n}" for n in range(2, 1000)]

        for candidate in candidates:
            folder = self.path / candidate
            try:
                folder.mkdir()
                break
            except FileExistsError:
                continue
            except OSError as e:
                raise ResultsFolderError(self.path, e.strerror or str(e)) from e
        else:  # pragma: no cover - a thousand same-named videos in one run
            raise ResultsFolderError(self.path, f"too many videos named {name}")

        if size_bytes is None and uploaded_name is None:
            try:
                size_bytes = video.stat().st_size
            except OSError:
                size_bytes = None
        source: dict[str, Any] = (
            {"kind": "uploaded", "original_filename": uploaded_name}
            if uploaded_name
            else {"kind": "in_place", "path": str(video), "size_bytes": size_bytes}
        )
        if uploaded_name and size_bytes is not None:
            source["size_bytes"] = size_bytes

        def add(record: dict[str, Any]) -> None:
            record["videos"].append(
                {
                    "job_id": job.job_id,
                    "folder": folder.name,
                    "source": source,
                    "status": "pending",
                }
            )

        _update_record(self.path, add)
        job.output_dir = folder
        return folder


def run_folder_of(job: Any) -> Path | None:
    """The run folder holding `job`'s results, or None for older jobs."""
    if not getattr(job, "output_dir", None):
        return None
    folder = Path(job.output_dir).parent
    return folder if (folder / RUN_RECORD).is_file() else None


def record_job_finished(job: Any) -> None:
    """Update `job`'s entry in its run's `run.json` once it settles.

    Never raises: a run record is an index, and failing to update it must not
    fail the job whose results are already written.
    """
    folder = run_folder_of(job)
    if folder is None:
        return
    output_dir = Path(job.output_dir)
    try:
        files = sorted(p.name for p in output_dir.iterdir() if p.is_file())
    except OSError:
        files = []
    models: dict[str, list[Any]] = {}
    for name, result in (job.pipeline_results or {}).items():
        provenance = getattr(result, "provenance", None) or {}
        if provenance.get("models"):
            models[name] = provenance["models"]
    finished = job.completed_at.astimezone(UTC) if job.completed_at else None

    def update(record: dict[str, Any]) -> None:
        for entry in record["videos"]:
            if entry.get("job_id") == job.job_id:
                entry["status"] = job.status.value
                entry["finished_at"] = (
                    finished.isoformat(timespec="seconds") if finished else None
                )
                entry["files"] = files
                entry["models"] = models
                if job.error_message:
                    entry["error"] = job.error_message
                else:
                    entry.pop("error", None)

    try:
        _update_record(folder, update)
    except Exception as e:
        logger.warning(f"Could not update {folder / RUN_RECORD}: {e}")


def remove_job_results(job: Any) -> None:
    """Delete `job`'s video folder, and its run folder once nothing else is left.

    Only ever inside the results root, and never the source video (FR-030).
    """
    if not getattr(job, "output_dir", None):
        return
    output_dir = Path(job.output_dir)
    if not is_inside_results(output_dir) or output_dir.resolve() == results_root():
        return
    folder = run_folder_of(job)
    shutil.rmtree(output_dir, ignore_errors=True)
    if folder is None:
        return

    def drop(record: dict[str, Any]) -> None:
        record["videos"] = [
            v for v in record["videos"] if v.get("job_id") != job.job_id
        ]

    try:
        _update_record(folder, drop)
        if [p.name for p in folder.iterdir()] == [RUN_RECORD]:
            shutil.rmtree(folder)
            with _lock:
                for batch_id, path in list(_batch_run_folders.items()):
                    if path == folder:
                        del _batch_run_folders[batch_id]
    except OSError as e:
        logger.warning(f"Could not tidy run folder {folder}: {e}")


def read_record(folder: Path) -> dict[str, Any] | None:
    """`folder`'s run record, or None when it has none or it can't be read."""
    try:
        return json.loads((folder / RUN_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _redacted(config: dict[str, Any] | None) -> dict[str, Any]:
    from .provenance import redact

    return redact(dict(config or {}))


def _update_record(folder: Path, change: Callable[[dict[str, Any]], None]) -> None:
    with _lock:
        record = read_record(folder)
        if record is None:
            raise OSError(f"no readable {RUN_RECORD} in {folder}")
        change(record)
        _write_record(folder, record)


def _write_record(folder: Path, record: dict[str, Any]) -> None:
    """Replace `run.json` atomically: readers see the old file or the new one."""
    with _lock:
        fd, temp = tempfile.mkstemp(dir=folder, prefix=".run-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)
            os.replace(temp, folder / RUN_RECORD)
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise
