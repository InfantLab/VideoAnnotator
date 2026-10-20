"""Datasets as projects (spec 029 R4, R5): videos by path from any folder,
added and removed, current settings, and edits open to any signed-in user."""

from __future__ import annotations

from tests.fixtures.video_index import write_video
from videoannotator import projects

OTHER = {"id": "33333333-3333-3333-3333-333333333333", "username": "ana"}


def _add(api, project, *paths):
    return api.client.post(
        f"/api/v1/datasets/{project['id']}/videos",
        json={"paths": [str(p) for p in paths]},
    )


def _remove(api, project, *paths):
    return api.client.request(
        "DELETE",
        f"/api/v1/datasets/{project['id']}/videos",
        json={"paths": [str(p) for p in paths]},
    )


def _held(api, project):
    return {v["name"] for v in api.videos(project=project["id"])["videos"]}


def test_add_videos_from_two_folders(api):
    a = write_video(api.shared / "site_a" / "p01.mp4")
    b = write_video(api.shared / "site_b" / "p02.mp4")
    project = api.project("QS")

    r = _add(api, project, a, b)

    assert r.status_code == 200, r.text
    body = r.json()
    assert {e["path"] for e in body["video_manifest"]} == {
        str(a.resolve()),
        str(b.resolve()),
    }
    assert body["server_selection"] is True
    assert _held(api, project) == {"p01.mp4", "p02.mp4"}


def test_adding_to_a_folder_project_makes_it_a_selection(api):
    pk = api.shared / "pk"
    write_video(pk / "p01.mp4")
    write_video(pk / "p02.mp4")
    elsewhere = write_video(api.shared / "other" / "x.mp4")
    project = api.project("Follows", server_folder=str(pk))

    body = _add(api, project, elsewhere).json()

    assert body["server_selection"] is True
    assert body["server_folder"] == str(pk)
    assert _held(api, project) == {"p01.mp4", "p02.mp4", "x.mp4"}
    write_video(pk / "p03.mp4")
    projects.clear_cache()  # a folder's new files show within a few seconds
    listed = api.videos(project=project["id"])
    assert {v["name"] for v in listed["new_in_folder"]} == {"p03.mp4"}


def test_adding_a_video_it_holds_changes_nothing(api):
    pk = api.shared / "pk"
    a = write_video(pk / "p01.mp4")
    project = api.project("Follows", server_folder=str(pk))

    body = _add(api, project, a).json()

    assert body["server_selection"] is False
    assert body["video_manifest"] == []


def test_remove_leaves_files_and_results(api):
    a = write_video(api.shared / "p01.mp4")
    b = write_video(api.shared / "p02.mp4")
    api.jobs.make_job(a, pipelines={"face_analysis": ("completed", None, None)})
    project = api.project("QS")
    _add(api, project, a, b)

    r = _remove(api, project, a)

    assert r.status_code == 200, r.text
    assert _held(api, project) == {"p02.mp4"}
    assert a.is_file()
    [video] = [v for v in api.videos()["videos"] if v["name"] == "p01.mp4"]
    assert video["results"]["face_analysis"]["state"] == "done"


def test_removing_from_an_upload_project_held_by_its_runs(api):
    project = api.project("Uploads")
    _, stored = api.jobs.upload("home.mp4", dataset_id=project["id"])
    api.jobs.upload("other.mp4", dataset_id=project["id"])
    assert _held(api, project) == {"home.mp4", "other.mp4"}

    _remove(api, project, stored)

    assert _held(api, project) == {"other.mp4"}


def test_another_user_may_edit_but_not_delete(api):
    a = write_video(api.shared / "p01.mp4")
    project = api.project("QS")
    api.as_user(OTHER)

    assert _add(api, project, a).status_code == 200
    renamed = api.client.put(
        f"/api/v1/datasets/{project['id']}", json={"name": "QS renamed"}
    )
    assert renamed.status_code == 200, renamed.text
    deleted = api.client.delete(f"/api/v1/datasets/{project['id']}")
    assert deleted.status_code == 403


def test_export_and_import_keep_paths_and_settings(api):
    a = write_video(api.shared / "p01.mp4")
    project = api.project("QS")
    _add(api, project, a)
    api.client.put(
        f"/api/v1/datasets/{project['id']}",
        json={"current_settings": {"selected_pipelines": ["face_analysis"]}},
    )
    exported = api.client.get(f"/api/v1/datasets/{project['id']}").json()

    imported = api.client.post(
        "/api/v1/datasets/", json=exported | {"name": "QS imported"}
    )

    assert imported.status_code == 201, imported.text
    assert imported.json()["video_manifest"][0]["path"] == str(a.resolve())
    assert _held(api, imported.json()) == {"p01.mp4"}


def test_a_project_from_a_selection_and_a_clashing_name(api):
    a = write_video(api.shared / "p01.mp4")
    entry = {"filename": "p01.mp4", "size_bytes": a.stat().st_size, "path": str(a)}
    made = api.client.post(
        "/api/v1/datasets/",
        json={"name": "Picked", "video_manifest": [entry], "server_selection": True},
    )
    assert made.status_code == 201, made.text
    assert _held(api, made.json()) == {"p01.mp4"}

    again = api.client.post("/api/v1/datasets/", json={"name": "Picked"})

    assert again.status_code == 409
    assert again.json()["error"]["code"] == "DATASET_NAME_CONFLICT"
