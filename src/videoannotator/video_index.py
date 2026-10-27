"""Videos as groups of paths, their results, and runs (spec 029 R1, R2, R11).

A video is every path the server can tell holds one file: the same resolved
path, hard links of one file (same device and inode, which every rerun of an
upload is), a rerun of a stored upload (so the link survives deleting one
copy), or the same input sha256 recorded in provenance where one of the paths is an
upload (two copies in shared folders stay two videos, as the researcher sees
them). Never the name or the size: two recordings of the same length stay two.

Built from `job_summaries()` (one query) plus one stat per distinct path, and
cached until the jobs change or a few seconds pass.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .storage.base import JobSummary

DONE, RUNNING, WAITING, FAILED = "done", "running", "waiting", "failed"
UPLOADS = "uploads"
STOPPED_REASON = "Stopped before it finished"

_JOB_WAITING = {"pending", "retrying"}


@dataclass
class PipelineState:
    state: str
    run_id: str
    job_id: str
    finished_at: datetime | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "run_id": self.run_id,
            "finished_at": self.finished_at,
            "error": self.error,
        }


@dataclass
class Video:
    key: str
    path: str
    paths: list[str]
    name: str
    folder: str
    available: bool
    size_bytes: int | None
    jobs: list[JobSummary] = field(default_factory=list)  # newest first
    results: dict[str, PipelineState] = field(default_factory=dict)
    last_run_at: datetime | None = None

    @property
    def uploaded(self) -> bool:
        return self.folder == UPLOADS


@dataclass
class Run:
    id: str
    name: str
    jobs: list[JobSummary]
    project_id: str | None
    total: int = 0
    done: int = 0
    failed: int = 0
    running: int = 0
    waiting: int = 0
    cancelled: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    pipelines: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.running > 0 or self.waiting > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "total": self.total,
            "done": self.done,
            "failed": self.failed,
            "running": self.running,
            "waiting": self.waiting,
            "cancelled": self.cancelled,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "pipelines": self.pipelines,
        }


@dataclass(frozen=True)
class _Stat:
    exists: bool
    size: int | None
    inode: tuple[int, int] | None


def _resolve(raw: str) -> str:
    try:
        return str(Path(raw).expanduser().resolve())
    except (OSError, RuntimeError):
        return raw


def _stat(path: str) -> _Stat:
    try:
        st = os.stat(path)
    except OSError:
        return _Stat(False, None, None)
    return _Stat(True, st.st_size, (st.st_dev, st.st_ino) if st.st_ino else None)


def _within(path: str, folder: str | None) -> bool:
    if not folder:
        return False
    return path.startswith(folder.rstrip(os.sep) + os.sep)


def _time(value: datetime | None) -> float:
    return value.timestamp() if value else 0.0


def run_id_of(job: JobSummary) -> str:
    return job.batch_id or job.job_id


def pipeline_state(job: JobSummary, name: str) -> PipelineState | None:
    """Where `name` stands in `job`, or None when it was never asked for."""
    run_id = run_id_of(job)
    result = job.pipelines.get(name)
    status = result.status if result is not None else None
    error = result.error if result is not None else None
    if status is None:
        if name not in (job.selected_pipelines or []):
            return None
        status = {"completed": "failed"}.get(job.status, job.status)
        if job.status == "completed":
            error = "No result was recorded"
        elif job.status == "failed":
            error = "The run failed before this step finished"
    if status == "completed":
        return PipelineState(DONE, run_id, job.job_id, job.completed_at)
    if status == "running":
        return PipelineState(RUNNING, run_id, job.job_id, None)
    if status in _JOB_WAITING:
        return PipelineState(WAITING, run_id, job.job_id, None)
    if status == "cancelled":
        return PipelineState(FAILED, run_id, job.job_id, None, STOPPED_REASON)
    return PipelineState(FAILED, run_id, job.job_id, job.completed_at, error)


def _results(jobs: list[JobSummary]) -> dict[str, PipelineState]:
    """Per pipeline, its newest finished result, else its newest state."""
    names: list[str] = []
    for job in jobs:
        for name in [*job.pipelines, *(job.selected_pipelines or [])]:
            if name not in names:
                names.append(name)
    results = {}
    for name in names:
        states = [s for j in jobs if (s := pipeline_state(j, name)) is not None]
        if not states:
            continue
        done = [s for s in states if s.state == DONE]
        results[name] = done[0] if done else states[0]
    return results


class _UnionFind:
    def __init__(self, parent: dict[str, str] | None = None):
        self.parent: dict[str, str] = dict(parent or {})

    def add(self, item: str) -> None:
        self.parent.setdefault(item, item)

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


class VideoIndex:
    """The videos the server knows, grouped (R1)."""

    def __init__(
        self,
        summaries: list[JobSummary],
        uploads_root: Path | None = None,
        *,
        _resolved: dict[str, str] | None = None,
        _stats: dict[str, _Stat] | None = None,
        _groups: _UnionFind | None = None,
        _by_inode: dict[tuple[int, int], str] | None = None,
    ):
        self.summaries = summaries
        self.uploads_root = str(uploads_root) if uploads_root else None
        self._resolved = _resolved if _resolved is not None else {}
        self._stats = _stats if _stats is not None else {}
        self._groups = _groups or _UnionFind()
        self._by_inode = _by_inode if _by_inode is not None else {}
        self.jobs_by_id = {s.job_id: s for s in summaries}
        self._videos: dict[str, Video] = {}
        self._video_of_path: dict[str, str] = {}

    # -- building ---------------------------------------------------------

    def resolved(self, raw: str) -> str:
        found = self._resolved.get(raw)
        if found is None:
            found = self._resolved[raw] = _resolve(raw)
        return found

    def _add_path(self, path: str) -> None:
        if path in self._stats:
            return
        self._groups.add(path)
        stat = self._stats[path] = _stat(path)
        if stat.inode is not None:
            other = self._by_inode.setdefault(stat.inode, path)
            if other != path:
                self._groups.union(path, other)

    def _join_jobs(self) -> None:
        by_sha: dict[str, list[tuple[str, bool]]] = {}
        for job in self.summaries:
            if not job.video_path:
                continue
            path = self.resolved(job.video_path)
            self._add_path(path)
            uploaded = self._is_upload(job, path)
            for result in job.pipelines.values():
                if result.sha256:
                    by_sha.setdefault(result.sha256, []).append((path, uploaded))
            original = self.jobs_by_id.get(job.rerun_of or "")
            if original is not None and original.video_path:
                original_path = self.resolved(original.video_path)
                if self._stored_in(original, original_path):
                    self._add_path(original_path)
                    self._groups.union(path, original_path)
        # One content in several places: an upload joins every copy, but two
        # files in shared folders stay two videos, as the researcher sees them.
        for found in by_sha.values():
            uploads = [p for p, uploaded in found if uploaded]
            if not uploads:
                continue
            for path, _ in found:
                self._groups.union(path, uploads[0])

    def _is_upload(self, job: JobSummary, path: str) -> bool:
        return _within(path, self.uploads_root) or self._stored_in(job, path)

    def _stored_in(self, job: JobSummary, path: str) -> bool:
        return bool(job.storage_path) and _within(
            path, self.resolved(job.storage_path or "")
        )

    def _group(self) -> None:
        jobs_by_path: dict[str, list[JobSummary]] = {}
        for job in self.summaries:
            if job.video_path:
                jobs_by_path.setdefault(self.resolved(job.video_path), []).append(job)
        members: dict[str, list[str]] = {}
        for path in self._stats:
            members.setdefault(self._groups.find(path), []).append(path)

        self._videos = {}
        self._video_of_path = {}
        for paths in members.values():
            jobs = sorted(
                (j for p in paths for j in jobs_by_path.get(p, [])),
                key=lambda j: _time(j.created_at),
                reverse=True,
            )
            newest: dict[str, float] = {}
            for job in jobs:
                newest.setdefault(
                    self.resolved(job.video_path or ""), _time(job.created_at)
                )
            ordered = sorted(paths, key=lambda p: (-newest.get(p, -1.0), p))
            existing = [p for p in ordered if self._stats[p].exists]
            path = existing[0] if existing else ordered[0]
            key = min(paths)
            stat = self._stats[path]
            self._videos[key] = Video(
                key=key,
                path=path,
                paths=ordered,
                name=Path(path).name,
                folder=self._folder_of(path, jobs),
                available=stat.exists,
                size_bytes=stat.size,
                jobs=jobs,
                results=_results(jobs),
                last_run_at=jobs[0].created_at if jobs else None,
            )
            for p in paths:
                self._video_of_path[p] = key

    def _folder_of(self, path: str, jobs: list[JobSummary]) -> str:
        if _within(path, self.uploads_root):
            return UPLOADS
        own = [j for j in jobs if self.resolved(j.video_path or "") == path]
        if any(self._stored_in(j, path) for j in own):
            return UPLOADS
        return str(Path(path).parent)

    def build(self, extra_paths: Iterable[str] = ()) -> VideoIndex:
        self._join_jobs()
        for raw in extra_paths:
            self._add_path(self.resolved(raw))
        self._group()
        return self

    def extended(self, paths: Iterable[str]) -> VideoIndex:
        """This index with `paths` added (a folder's files, a project's
        entries), leaving this one as it is."""
        new = [self.resolved(p) for p in paths]
        if all(p in self._stats for p in new):
            return self
        index = VideoIndex(
            self.summaries,
            Path(self.uploads_root) if self.uploads_root else None,
            _resolved=self._resolved,
            _stats=dict(self._stats),
            _groups=_UnionFind(self._groups.parent),
            _by_inode=dict(self._by_inode),
        )
        for path in new:
            index._add_path(path)
        index._group()
        return index

    # -- reading ----------------------------------------------------------

    def videos(self) -> list[Video]:
        return sorted(self._videos.values(), key=lambda v: (v.name.lower(), v.path))

    def video(self, key: str) -> Video | None:
        return self._videos.get(key)

    def video_for(self, path: str | Path) -> Video | None:
        key = self._video_of_path.get(self.resolved(str(path)))
        return self._videos.get(key) if key else None

    def key_of(self, path: str | Path) -> str | None:
        return self._video_of_path.get(self.resolved(str(path)))

    def size_of(self, path: str) -> int | None:
        stat = self._stats.get(self.resolved(path))
        return stat.size if stat else None


def build_index(
    summaries: list[JobSummary],
    extra_paths: Iterable[str] = (),
    uploads_root: Path | None = None,
) -> VideoIndex:
    return VideoIndex(summaries, uploads_root).build(extra_paths)


# -- runs (R12, R17) ---------------------------------------------------------


def _run_name(jobs: list[JobSummary]) -> str:
    from types import SimpleNamespace

    from .results_folder import read_record, run_folder_of

    for job in jobs:
        if job.batch_name:
            return job.batch_name
    for job in jobs:
        folder = run_folder_of(SimpleNamespace(output_dir=job.output_dir))
        if folder is not None:
            # The name the run was given, not its folder's: that one carries
            # the date and a number to keep folders apart (UAT #41).
            recorded = ((read_record(folder) or {}).get("run") or {}).get("name")
            return str(recorded) if recorded else folder.name
    video = next((j.video_path for j in jobs if j.video_path), None)
    return f"Run of {Path(video).name}" if video else "Run"


def _run_of(run_id: str, jobs: list[JobSummary]) -> Run:
    jobs = sorted(jobs, key=lambda j: _time(j.created_at))
    run = Run(
        id=run_id,
        name=_run_name(jobs),
        jobs=jobs,
        project_id=next((j.dataset_id for j in jobs if j.dataset_id), None),
        total=len(jobs),
    )
    for job in jobs:
        if job.status == "completed":
            run.done += 1
        elif job.status == "failed":
            run.failed += 1
        elif job.status == "running":
            run.running += 1
        elif job.status == "cancelled":
            run.cancelled += 1
        else:
            run.waiting += 1
        for name in job.selected_pipelines or list(job.pipelines):
            if name not in run.pipelines:
                run.pipelines.append(name)
    run.started_at = jobs[0].created_at if jobs else None
    if not run.active:
        finished = [j.completed_at for j in jobs if j.completed_at]
        run.finished_at = max(finished) if finished else None
    return run


def runs(index: VideoIndex) -> list[Run]:
    """Every run, newest first: a run is the jobs sharing a batch id, or one
    job without one."""
    grouped: dict[str, list[JobSummary]] = {}
    for job in index.summaries:
        grouped.setdefault(run_id_of(job), []).append(job)
    found = [_run_of(run_id, jobs) for run_id, jobs in grouped.items()]
    return sorted(found, key=lambda r: _time(r.started_at), reverse=True)


def run(index: VideoIndex, run_id: str) -> Run | None:
    jobs = [j for j in index.summaries if run_id_of(j) == run_id]
    return _run_of(run_id, jobs) if jobs else None


def run_videos(index: VideoIndex, run_id: str) -> list[Video]:
    keys = {
        index.key_of(j.video_path)
        for j in index.summaries
        if run_id_of(j) == run_id and j.video_path
    }
    return [v for v in index.videos() if v.key in keys]


def latest_pairs(index: VideoIndex, keys: Iterable[str]) -> set[tuple[str, str]]:
    """Per video and pipeline, the newest job whose pipeline completed."""
    pairs = set()
    for key in keys:
        video = index.video(key)
        if video is None:
            continue
        for name, state in video.results.items():
            if state.state == DONE:
                pairs.add((state.job_id, name))
    return pairs


# -- the cache (R11) ---------------------------------------------------------

CACHE_SECONDS = 5.0

_cache_lock = threading.Lock()
_cache: dict[int, tuple[tuple[Any, ...], float, VideoIndex]] = {}


def cached_index(storage: Any, uploads_root: Path | None = None) -> VideoIndex:
    """The index for `storage`, rebuilt when its jobs change or after a few
    seconds (files can move without a job changing)."""
    signature = (storage.jobs_signature(), str(uploads_root))
    now = time.monotonic()
    with _cache_lock:
        cached = _cache.get(id(storage))
        if cached and cached[0] == signature and now - cached[1] < CACHE_SECONDS:
            return cached[2]
    index = build_index(storage.job_summaries(), uploads_root=uploads_root)
    with _cache_lock:
        _cache[id(storage)] = (signature, now, index)
    return index


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
