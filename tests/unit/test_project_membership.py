"""Which videos a project holds (spec 029 R4)."""

from __future__ import annotations

import uuid
from types import SimpleNamespace

from tests.fixtures.video_index import JobsDb, write_video
from videoannotator.projects import build_catalog, manifest_entry
from videoannotator.video_index import build_index


def _dataset(name="QS", **fields):
    defaults = {
        "id": str(uuid.uuid4()),
        "name": name,
        "video_manifest": [],
        "server_folder": None,
        "server_folder_recursive": False,
        "server_selection": False,
    }
    return SimpleNamespace(**(defaults | fields))


def _catalog(jobs: JobsDb, *datasets):
    index = build_index(jobs.storage.job_summaries(), uploads_root=jobs.storage_root)
    return build_catalog(index, list(datasets))


def _names(catalog, dataset) -> set[str]:
    return {catalog.index.video(key).name for key in catalog.members[str(dataset.id)]}


def test_following_a_folder_holds_its_videos_including_new_ones(tmp_path):
    jobs = JobsDb(tmp_path)
    folder = tmp_path / "pk"
    write_video(folder / "p01.mp4")
    write_video(folder / "sub" / "p02.mp4")
    flat = _dataset("flat", server_folder=str(folder))
    deep = _dataset("deep", server_folder=str(folder), server_folder_recursive=True)

    assert _names(_catalog(jobs, flat, deep), flat) == {"p01.mp4"}

    write_video(folder / "p03.mp4")
    catalog = _catalog(jobs, flat, deep)

    assert _names(catalog, flat) == {"p01.mp4", "p03.mp4"}
    assert _names(catalog, deep) == {"p01.mp4", "p02.mp4", "p03.mp4"}
    assert catalog.new[str(flat.id)] == []


def test_a_selection_holds_its_entries_and_offers_the_rest(tmp_path):
    jobs = JobsDb(tmp_path)
    folder = tmp_path / "pk"
    write_video(folder / "p01.mp4")
    write_video(folder / "sub" / "p02.mp4")
    write_video(folder / "p03.mp4")
    selection = _dataset(
        server_folder=str(folder),
        server_folder_recursive=True,
        server_selection=True,
        video_manifest=[
            {"filename": "p01.mp4", "size_bytes": 1},
            {"filename": "p02.mp4", "size_bytes": 1, "relative_path": "sub/p02.mp4"},
        ],
    )

    catalog = _catalog(jobs, selection)

    assert _names(catalog, selection) == {"p01.mp4", "p02.mp4"}
    assert [v.name for v in catalog.new[str(selection.id)]] == ["p03.mp4"]


def test_a_project_spans_two_folders_by_path(tmp_path):
    jobs = JobsDb(tmp_path)
    a = write_video(tmp_path / "site_a" / "p01.mp4")
    b = write_video(tmp_path / "site_b" / "p01.mp4")
    project = _dataset(
        server_selection=True,
        video_manifest=[manifest_entry(a), manifest_entry(b)],
    )

    catalog = _catalog(jobs, project)

    held = catalog.members[str(project.id)]
    assert {catalog.index.video(k).folder for k in held} == {
        str(a.parent),
        str(b.parent),
    }


def test_uploads_by_dataset_id_and_by_stored_copy(tmp_path):
    jobs = JobsDb(tmp_path)
    project = _dataset("uploads")
    jobs.upload("ran_from_it.mp4", dataset_id=str(project.id))
    _, copy = jobs.upload("listed.mp4")
    jobs.upload("other.mp4")
    project.video_manifest = [
        {"filename": "listed.mp4", "size_bytes": copy.stat().st_size}
    ]

    assert _names(_catalog(jobs, project), project) == {
        "ran_from_it.mp4",
        "listed.mp4",
    }


def test_a_stored_copy_of_another_size_is_not_held(tmp_path):
    jobs = JobsDb(tmp_path)
    jobs.upload("listed.mp4")
    project = _dataset(video_manifest=[{"filename": "listed.mp4", "size_bytes": 1}])

    assert _names(_catalog(jobs, project), project) == set()


def test_a_selection_without_a_folder_holds_only_its_entries(tmp_path):
    jobs = JobsDb(tmp_path)
    project = _dataset(server_selection=True)
    jobs.upload("ran_from_it.mp4", dataset_id=str(project.id))

    assert _names(_catalog(jobs, project), project) == set()


def test_a_video_in_two_projects_and_its_rerun(tmp_path):
    jobs = JobsDb(tmp_path)
    first, copy = jobs.upload("clip.mp4")
    _, rerun_copy = jobs.rerun(first)
    qs = _dataset("QS", server_selection=True, video_manifest=[manifest_entry(copy)])
    qs2 = _dataset("QS2", server_selection=True, video_manifest=[manifest_entry(copy)])

    catalog = _catalog(jobs, qs2, qs)

    key = catalog.index.key_of(rerun_copy)
    assert [p["name"] for p in catalog.projects_of(key)] == ["QS", "QS2"]
    assert catalog.in_any_project() == {key}
