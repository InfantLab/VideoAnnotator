"""Why a job's video can't be found: not shared any more, or moved (spec 024, R9)."""

import sys
from pathlib import Path

import pytest

from videoannotator.api.v1 import ingest as ingest_module
from videoannotator.results_folder import video_unavailable_reason


@pytest.fixture
def shared(tmp_path, monkeypatch):
    root = tmp_path / "Studies"
    root.mkdir()
    monkeypatch.setattr(ingest_module, "INGEST_ROOTS", str(root))
    return root


def test_a_video_outside_every_share_is_not_shared_any_more(shared, tmp_path):
    video = tmp_path / "Old study" / "Day 1" / "child01.mp4"
    assert video_unavailable_reason(video) == (
        f"{video.parent} isn't shared with VideoAnnotator any more"
    )


def test_a_video_inside_a_share_was_moved_or_deleted(shared):
    assert video_unavailable_reason(shared / "gone.mp4") == (
        "moved or deleted since the job was created"
    )


# Container paths: the server reading them runs on Linux.
posix_only = pytest.mark.skipif(sys.platform == "win32", reason="container paths")


@posix_only
def test_the_folder_is_named_as_the_host_shows_it(shared, monkeypatch):
    monkeypatch.setenv(
        "VIDEOANNOTATOR_HOST_PATHS", "/c/Users/ada/Old=C:\\Users\\ada\\Old"
    )
    assert video_unavailable_reason("/c/Users/ada/Old/Day 1/a.mp4") == (
        "C:\\Users\\ada\\Old\\Day 1 isn't shared with VideoAnnotator any more"
    )


def test_an_uploaded_video_was_never_shared(shared, tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    video = tmp_path / "storage" / "job-1" / "upload.mp4"
    assert video_unavailable_reason(video) == (
        "moved or deleted since the job was created"
    )


@posix_only
def test_relative_to_nothing_shared(monkeypatch):
    monkeypatch.setattr(ingest_module, "in_container", lambda: True)
    monkeypatch.setattr(ingest_module, "INGEST_ROOTS", "")
    assert video_unavailable_reason(Path("/home/ada/Studies/a.mp4")) == (
        "/home/ada/Studies isn't shared with VideoAnnotator any more"
    )
