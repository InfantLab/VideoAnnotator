"""Which videos a project holds (spec 029 R4).

A project is a saved dataset. What it holds is worked out from what the dataset
records:

- a folder, not a selection ("follows the folder"): every video in it,
  including ones added later;
- a folder and a selection: its manifest entries (`relative_path` within the
  folder, `path`, or an older entry's filename); the folder's other videos are
  "new in the folder";
- no folder: entries by `path`, the videos of jobs run from it (`dataset_id`),
  and older entries' stored copies, by filename and size as `/stored-videos`
  finds them. Once videos are added or removed by path it is a selection
  (`server_selection`), and only its entries count.

A video (a group of paths, R1) is in a project when any of its paths is.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .video_index import CACHE_SECONDS, Video, VideoIndex

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm")


def folder_videos(folder: Path | str, recursive: bool = False) -> list[Path]:
    """Video files in `folder`, as ingest lists them; never raises."""
    from .results_folder import results_root

    try:
        root = Path(folder).expanduser().resolve()
        results = results_root()
        entries = root.rglob("*") if recursive else root.iterdir()
        return sorted(
            entry
            for entry in entries
            if entry.suffix.lower() in VIDEO_EXTENSIONS
            and entry.is_file()
            and not entry.is_relative_to(results)
        )
    except OSError:
        return []


def _entries(dataset: Any) -> list[dict[str, Any]]:
    return [e for e in (dataset.video_manifest or []) if isinstance(e, dict)]


def _folder(dataset: Any) -> Path | None:
    if not dataset.server_folder:
        return None
    try:
        return Path(dataset.server_folder).expanduser().resolve()
    except (OSError, RuntimeError):
        return None


def follows_folder(dataset: Any) -> bool:
    return bool(dataset.server_folder) and not dataset.server_selection


class _Folders:
    """Folder listings, read once per catalog."""

    def __init__(self) -> None:
        self._seen: dict[tuple[str, bool], list[Path]] = {}

    def videos(self, folder: Path, recursive: bool) -> list[Path]:
        key = (str(folder), recursive)
        if key not in self._seen:
            self._seen[key] = folder_videos(folder, recursive)
        return self._seen[key]


def member_paths(dataset: Any, folders: _Folders | None = None) -> list[str]:
    """Paths a project names directly, which the index may not know yet."""
    folders = folders or _Folders()
    paths = [e["path"] for e in _entries(dataset) if e.get("path")]
    folder = _folder(dataset)
    if folder is not None:
        recursive = bool(dataset.server_folder_recursive)
        paths += [str(p) for p in folders.videos(folder, recursive)]
    return paths


def _by_name(index: VideoIndex) -> dict[str, list[Video]]:
    found: dict[str, list[Video]] = {}
    for video in index.videos():
        for name in {Path(p).name for p in video.paths}:
            found.setdefault(name, []).append(video)
    return found


def members(
    dataset: Any,
    index: VideoIndex,
    folders: _Folders | None = None,
    by_name: dict[str, list[Video]] | None = None,
) -> set[str]:
    """The keys of the videos `dataset` holds. `index` must know its
    `member_paths`."""
    folders = folders or _Folders()
    keys: set[str] = set()

    def add(path: str | Path) -> None:
        key = index.key_of(path)
        if key is not None:
            keys.add(key)

    entries = _entries(dataset)
    for entry in entries:
        if entry.get("path"):
            add(entry["path"])

    folder = _folder(dataset)
    recursive = bool(dataset.server_folder_recursive)
    if folder is not None:
        listed = folders.videos(folder, recursive)
        if not dataset.server_selection:
            for path in listed:
                add(path)
            return keys
        for entry in entries:
            if entry.get("path"):
                continue
            if entry.get("relative_path"):
                add(folder / entry["relative_path"])
            elif entry.get("filename"):
                for path in listed:
                    if path.name == entry["filename"]:
                        add(path)
        return keys

    if dataset.server_selection:
        return keys
    dataset_id = str(dataset.id)
    for video in index.videos():
        if any(j.dataset_id == dataset_id for j in video.jobs):
            keys.add(video.key)
    by_name = by_name if by_name is not None else _by_name(index)
    for entry in entries:
        if entry.get("path") or not entry.get("filename"):
            continue
        size = entry.get("size_bytes")
        for video in by_name.get(entry["filename"], []):
            if size is None or any(
                index.size_of(p) == size
                for p in video.paths
                if Path(p).name == entry["filename"]
            ):
                keys.add(video.key)
    return keys


def new_in_folder(
    dataset: Any, index: VideoIndex, held: set[str], folders: _Folders | None = None
) -> list[Video]:
    """A selection's folder videos it doesn't hold yet (replaces drift)."""
    folder = _folder(dataset)
    if folder is None or not dataset.server_selection:
        return []
    folders = folders or _Folders()
    found = []
    for path in folders.videos(folder, bool(dataset.server_folder_recursive)):
        video = index.video_for(path)
        if video is not None and video.key not in held and video not in found:
            found.append(video)
    return found


@dataclass
class Catalog:
    """The index with every project's videos, and who holds what."""

    index: VideoIndex
    datasets: list[Any]
    members: dict[str, set[str]] = field(default_factory=dict)
    new: dict[str, list[Video]] = field(default_factory=dict)

    def projects_of(self, key: str) -> list[dict[str, str]]:
        found = [
            {"id": str(d.id), "name": d.name}
            for d in self.datasets
            if key in self.members.get(str(d.id), set())
        ]
        return sorted(found, key=lambda p: p["name"].lower())

    def dataset(self, dataset_id: str) -> Any | None:
        return next((d for d in self.datasets if str(d.id) == dataset_id), None)

    def in_any_project(self) -> set[str]:
        return set().union(*self.members.values()) if self.members else set()


def projects_of(catalog: Catalog) -> dict[str, list[dict[str, str]]]:
    """Each video's projects, `[{id, name}]`, by video key."""
    return {v.key: catalog.projects_of(v.key) for v in catalog.index.videos()}


def build_catalog(index: VideoIndex, datasets: list[Any]) -> Catalog:
    folders = _Folders()
    extra = [p for d in datasets for p in member_paths(d, folders)]
    index = index.extended(extra)
    by_name = _by_name(index)
    catalog = Catalog(index=index, datasets=list(datasets))
    for dataset in datasets:
        held = members(dataset, index, folders, by_name)
        catalog.members[str(dataset.id)] = held
        catalog.new[str(dataset.id)] = new_in_folder(dataset, index, held, folders)
    return catalog


def _datasets_signature(datasets: list[Any]) -> str:
    content = [
        (
            str(d.id),
            d.name,
            d.video_manifest,
            d.server_folder,
            bool(d.server_folder_recursive),
            bool(d.server_selection),
        )
        for d in datasets
    ]
    return hashlib.sha256(json.dumps(content, default=str).encode()).hexdigest()


_lock = threading.Lock()
_cached: dict[str, tuple[int, str, float, Catalog]] = {}


def cached_catalog(index: VideoIndex, datasets: list[Any]) -> Catalog:
    """`build_catalog`, reused while the index and the datasets are unchanged
    and for a few seconds (a folder can gain a file without either changing)."""
    signature = _datasets_signature(datasets)
    now = time.monotonic()
    with _lock:
        cached = _cached.get("catalog")
        if (
            cached is not None
            and cached[0] == id(index)
            and cached[1] == signature
            and now - cached[2] < CACHE_SECONDS
        ):
            return cached[3]
    catalog = build_catalog(index, datasets)
    with _lock:
        _cached["catalog"] = (id(index), signature, now, catalog)
    return catalog


def clear_cache() -> None:
    with _lock:
        _cached.clear()


def catalog_for(storage: Any, db: Any) -> Catalog:
    """Every video the server knows, with its projects (spec 029)."""
    from .database.models import SavedDataset
    from .storage.config import get_storage_root
    from .video_index import cached_index

    index = cached_index(storage, uploads_root=get_storage_root())
    return cached_catalog(index, db.query(SavedDataset).all())


def manifest_entry(path: str | Path, index: VideoIndex | None = None) -> dict[str, Any]:
    """A manifest entry naming a video by its server path (R4)."""
    resolved = Path(path).expanduser().resolve()
    try:
        size = resolved.stat().st_size
    except OSError:
        size = (index.size_of(str(resolved)) if index else None) or 0
    return {"filename": resolved.name, "size_bytes": size, "path": str(resolved)}


def is_video_name(path: str | Path) -> bool:
    return os.path.splitext(str(path))[1].lower() in VIDEO_EXTENSIONS


def _entry_for(video: Video, dataset: Any) -> dict[str, Any]:
    entry = manifest_entry(video.path)
    entry["size_bytes"] = video.size_bytes or entry["size_bytes"]
    folder = _folder(dataset)
    if folder is not None and Path(video.path).is_relative_to(folder):
        entry["relative_path"] = Path(video.path).relative_to(folder).as_posix()
    return entry


def with_videos(
    dataset: Any, paths: list[str], catalog: Catalog
) -> tuple[list[dict[str, Any]], bool] | None:
    """The manifest and selection flag once `paths` are added, or None when
    the project holds them all already. A project that followed its folder,
    or held its uploads by run, becomes a selection of what it held (R4)."""
    held = catalog.members.get(str(dataset.id), set())
    index = catalog.index.extended(paths)
    new = []
    for path in paths:
        video = index.video_for(path)
        if video is not None and video.key not in held and video not in new:
            new.append(video)
    if not new:
        return None
    if dataset.server_selection:
        entries = list(_entries(dataset))
    else:
        entries = [_entry_for(v, dataset) for v in index.videos() if v.key in held]
    return entries + [_entry_for(v, dataset) for v in new], True


def without_videos(
    dataset: Any, paths: list[str], catalog: Catalog
) -> list[dict[str, Any]]:
    """The manifest once the videos at `paths` are removed: one entry per
    video it still holds. Files and results are never touched."""
    held = catalog.members.get(str(dataset.id), set())
    removed = {catalog.index.key_of(p) for p in paths}
    return [
        _entry_for(v, dataset)
        for v in catalog.index.videos()
        if v.key in held and v.key not in removed
    ]
