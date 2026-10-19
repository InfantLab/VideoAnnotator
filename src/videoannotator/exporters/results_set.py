"""Which result files an export reads, by run and video (spec 025).

Built from a folder (the CLI, offline: a video's folder, a run's folder, a
folder of runs, a downloaded zip extracted, a pre-022 job folder) or from jobs
(the API). Both end in the same `VideoResults`, so the CLI and the API export
the same rows for the same files (FR-010).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..provenance import read_record
from ..results_folder import RUN_RECORD
from .readers import FileKind, classify


@dataclass
class ResultFile:
    path: Path
    kind: FileKind
    provenance: dict[str, Any] | None

    @property
    def pipeline(self) -> str:
        recorded = (self.provenance or {}).get("pipeline") or {}
        name = recorded.get("name") if isinstance(recorded, dict) else None
        return name if isinstance(name, str) and name else self.kind.default_pipeline


@dataclass
class SkippedFile:
    run: str
    video: str
    file: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "run": self.run,
            "video": self.video,
            "file": self.file,
            "reason": self.reason,
        }


@dataclass
class MissingVideo:
    run: str
    video: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"run": self.run, "video": self.video, "reason": self.reason}


@dataclass
class VideoResults:
    run: str
    video: str
    files: list[ResultFile] = field(default_factory=list)
    skipped: list[SkippedFile] = field(default_factory=list)
    media_path: str | None = None
    # Where the job read the video, even when it has moved since (notes, spec 028).
    video_path: str | None = None


@dataclass
class ResultsSet:
    videos: list[VideoResults] = field(default_factory=list)
    missing: list[MissingVideo] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def skipped(self) -> list[SkippedFile]:
        return [s for v in self.videos for s in v.skipped]

    def is_empty(self) -> bool:
        return not any(v.files for v in self.videos)


def _kind_order(f: ResultFile) -> int:
    from .readers import FILE_KINDS

    return FILE_KINDS.index(f.kind)


def _read_provenance(path: Path) -> dict[str, Any] | None:
    try:
        record = read_record(path)
    except (OSError, ValueError):
        return None
    return record if isinstance(record, dict) else None


def _files_in(folder: Path) -> Iterable[Path]:
    try:
        return sorted(p for p in folder.iterdir() if p.is_file())
    except OSError:
        return []


def _collect(folder: Path, run: str, video: str) -> VideoResults:
    """Every result file directly in `folder`, all belonging to `video`."""
    results = VideoResults(run=run, video=video)
    for path in _files_in(folder):
        kind, stem = classify(path)
        if kind is not None:
            results.files.append(ResultFile(path, kind, _read_provenance(path)))
        elif stem == "":
            results.skipped.append(
                SkippedFile(run, video, path.name, "no reader for this file")
            )
    results.files.sort(key=_kind_order)
    return results


def _group_by_stem(folder: Path, run: str) -> list[VideoResults]:
    """A flat folder: result files grouped by the video stem in their names."""
    groups: dict[str, VideoResults] = {}
    unknown: list[Path] = []
    for path in _files_in(folder):
        kind, stem = classify(path)
        if kind is not None and stem:
            group = groups.setdefault(stem, VideoResults(run=run, video=stem))
            group.files.append(ResultFile(path, kind, _read_provenance(path)))
        elif stem == "":
            unknown.append(path)
    if unknown:
        owner = (
            next(iter(groups.values()))
            if len(groups) == 1
            else groups.setdefault(
                folder.name, VideoResults(run=run, video=folder.name)
            )
        )
        owner.skipped += [
            SkippedFile(run, owner.video, p.name, "no reader for this file")
            for p in unknown
        ]
    for group in groups.values():
        group.files.sort(key=_kind_order)
    return [g for g in groups.values() if g.files or g.skipped]


def _run_record(folder: Path) -> dict[str, Any]:
    try:
        record = json.loads((folder / RUN_RECORD).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return record if isinstance(record, dict) else {}


def _status_reason(status: str | None, error: str | None) -> str:
    if status == "failed":
        return f"failed: {error}" if error else "failed"
    if status in ("pending", "queued", "running", "retrying"):
        return "still running"
    if status == "cancelled":
        return "cancelled"
    return "no result files found"


def _read_run(folder: Path, results: ResultsSet) -> None:
    record = _run_record(folder)
    entries = {
        e.get("folder"): e
        for e in record.get("videos") or []
        if isinstance(e, dict) and e.get("folder")
    }
    seen = set()
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        seen.add(sub.name)
        video = _collect(sub, folder.name, sub.name)
        entry = entries.get(sub.name) or {}
        source = entry.get("source") or {}
        if isinstance(source, dict) and source.get("path"):
            video.media_path = str(source["path"])
        if video.files or video.skipped:
            results.videos.append(video)
        if not video.files:
            results.missing.append(
                MissingVideo(
                    folder.name,
                    sub.name,
                    _status_reason(entry.get("status"), entry.get("error")),
                )
            )
    for name, entry in entries.items():
        if name not in seen:
            results.missing.append(
                MissingVideo(
                    folder.name,
                    str(name),
                    _status_reason(entry.get("status"), entry.get("error")),
                )
            )


def _walk(folder: Path, results: ResultsSet, root: Path) -> None:
    if (folder / RUN_RECORD).is_file():
        _read_run(folder, results)
        return
    # A loose folder below the one exported is named by its path, so two
    # folders holding the same video's results stay apart.
    run = "" if folder == root else folder.relative_to(root).as_posix()
    results.videos += _group_by_stem(folder, run)
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()):
        _walk(sub, results, root)


def from_folder(path: str | Path) -> ResultsSet:
    """Every result under `path`; raises FileNotFoundError if it isn't a folder."""
    folder = Path(path).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    results = ResultsSet()
    parent = folder.resolve().parent
    if not (folder / RUN_RECORD).is_file() and (parent / RUN_RECORD).is_file():
        # One video's folder inside a run.
        video = _collect(folder, parent.name, folder.name)
        entry = next(
            (
                e
                for e in _run_record(parent).get("videos") or []
                if isinstance(e, dict) and e.get("folder") == folder.name
            ),
            {},
        )
        source = entry.get("source") or {}
        if isinstance(source, dict) and source.get("path"):
            video.media_path = str(source["path"])
        if video.files or video.skipped:
            results.videos.append(video)
    else:
        _walk(folder, results, folder)
    results.videos.sort(key=lambda v: (v.run, v.video))
    return results


def from_jobs(jobs: Iterable[Any]) -> ResultsSet:
    """The results of `jobs` (BatchJob-like objects), as they are on disk."""
    from ..batch.result_files import job_folder
    from ..results_folder import run_folder_of

    results = ResultsSet()
    used: set[tuple[str, str]] = set()
    for job in jobs:
        folder = job_folder(job)
        run_folder = run_folder_of(job)
        video_path = Path(job.video_path) if job.video_path else None
        if run_folder is not None:
            run, video = run_folder.name, folder.name
        else:
            run = job.batch_name or ""
            video = video_path.stem if video_path else job.job_id
            base, number = video, 2
            while (run, video) in used:
                video, number = f"{base} {number}", number + 1
        used.add((run, video))

        status = getattr(job.status, "value", job.status)
        found = _collect(folder, run, video) if folder.is_dir() else None
        if found is None or not found.files:
            results.missing.append(
                MissingVideo(run, video, _status_reason(status, job.error_message))
            )
            if found is not None and found.skipped:
                results.videos.append(found)
            continue
        if video_path is not None:
            found.video_path = str(video_path)
            if video_path.is_file():
                found.media_path = str(video_path)
        results.videos.append(found)
        if status not in ("completed", "failed", "cancelled"):
            results.notes.append(
                f"{video}: still running; pipelines that haven't finished are not included"
            )
    results.videos.sort(key=lambda v: (v.run, v.video))
    return results
