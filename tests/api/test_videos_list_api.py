"""GET /videos: the videos the server knows, per scope (spec 029 R2)."""

from __future__ import annotations

from tests.fixtures.video_index import write_video
from videoannotator.projects import manifest_entry


def _fill(api):
    pk = api.shared / "pk"
    a = write_video(pk / "p01.mp4")
    b = write_video(pk / "p02.mp4")
    write_video(pk / "never_run.mp4")
    api.jobs.make_job(
        a,
        batch_id="b1",
        batch_name="Peekaboo faces",
        pipelines={"face_analysis": ("completed", None, None)},
    )
    api.jobs.make_job(
        b,
        batch_id="b1",
        batch_name="Peekaboo faces",
        pipelines={"face_analysis": ("failed", "No faces", None)},
    )
    _, upload = api.jobs.upload(
        "home.mp4", pipelines={"scene_detection": ("running", None, None)}
    )
    return pk, a, b, upload


def test_all_videos_with_results_and_counts(api):
    pk, a, _, upload = _fill(api)

    body = api.videos()

    assert body["scope"] == {
        "kind": "all",
        "id": None,
        "name": None,
        "path": None,
        "recursive": False,
    }
    by_name = {v["name"]: v for v in body["videos"]}
    assert set(by_name) == {"p01.mp4", "p02.mp4", "home.mp4"}
    p01 = by_name["p01.mp4"]
    assert p01["path"] == str(a.resolve())
    assert p01["folder"] == str(pk.resolve())
    assert p01["available"] is True
    assert p01["results"]["face_analysis"]["state"] == "done"
    assert p01["results"]["face_analysis"]["run_id"] == "b1"
    failed = by_name["p02.mp4"]["results"]["face_analysis"]
    assert (failed["state"], failed["error"]) == ("failed", "No faces")
    assert by_name["home.mp4"]["folder"] == "uploads"
    assert by_name["home.mp4"]["results"]["scene_detection"]["state"] == "running"

    counts = body["counts"]
    assert counts["all"] == 3
    assert counts["not_in_project"] == 3
    folders = {f["path"]: f for f in counts["folders"]}
    assert folders[str(pk.resolve())]["count"] == 2
    assert folders["uploads"] == {
        "path": "uploads",
        "label": "Uploaded videos",
        "count": 1,
        "uploads": True,
        "recursive": False,
    }


def test_a_folder_lists_files_never_run(api):
    pk, *_ = _fill(api)

    body = api.videos(folder=str(pk))

    assert body["scope"]["kind"] == "folder"
    assert body["scope"]["path"] == str(pk.resolve())
    names = {v["name"]: v for v in body["videos"]}
    assert set(names) == {"p01.mp4", "p02.mp4", "never_run.mp4"}
    assert names["never_run.mp4"]["results"] == {}
    assert body["total"] == 3


def test_a_folder_outside_the_shared_folders(api, tmp_path):
    private = tmp_path / "private"
    write_video(private / "x.mp4")

    r = api.client.get("/api/v1/videos", params={"folder": str(private)})

    assert r.status_code == 403
    assert r.json()["error"]["code"] == "PATH_NOT_ALLOWED"


def test_uploads(api):
    _fill(api)

    body = api.videos(folder="uploads")

    assert [v["name"] for v in body["videos"]] == ["home.mp4"]


def test_a_project_its_runs_and_its_new_videos(api):
    pk, a, b, upload = _fill(api)
    project = api.project(
        "Peekaboo study",
        server_folder=str(pk),
        server_selection=True,
        video_manifest=[manifest_entry(a), manifest_entry(upload)],
    )
    api.jobs.make_job(
        a,
        batch_id="b2",
        batch_name="Again",
        dataset_id=project["id"],
        pipelines={"face_analysis": ("running", None, None)},
    )

    body = api.videos(project=project["id"])

    assert body["scope"]["name"] == "Peekaboo study"
    assert {v["name"] for v in body["videos"]} == {"p01.mp4", "home.mp4"}
    assert {v["name"] for v in body["new_in_folder"]} == {"p02.mp4", "never_run.mp4"}
    [run] = body["runs"]
    assert (run["id"], run["name"], run["running"], run["active"]) == (
        "b2",
        "Again",
        1,
        True,
    )
    assert run["project"] == {"id": project["id"], "name": "Peekaboo study"}
    [p01] = [v for v in body["videos"] if v["name"] == "p01.mp4"]
    assert p01["projects"] == [{"id": project["id"], "name": "Peekaboo study"}]
    counts = body["counts"]
    [listed] = counts["projects"]
    assert listed["count"] == 2
    assert counts["not_in_project"] == 2


def test_not_in_a_project(api):
    pk, a, *_ = _fill(api)
    api.project("Just one", server_selection=True, video_manifest=[manifest_entry(a)])

    body = api.videos(not_in_project="true")

    assert {v["name"] for v in body["videos"]} == {"p02.mp4", "home.mp4"}


def test_a_run(api):
    _fill(api)

    body = api.videos(run="b1")

    assert body["scope"] == {
        "kind": "run",
        "id": "b1",
        "name": "Peekaboo faces",
        "path": None,
        "recursive": False,
    }
    assert [v["name"] for v in body["videos"]] == ["p01.mp4", "p02.mp4"]
    [run] = body["runs"]
    # A video whose job finished with a failed pipeline is done; the failure
    # shows on its row.
    assert (run["total"], run["done"], run["failed"], run["active"]) == (2, 2, 0, False)
    assert len(run["job_ids"]) == 2


def test_unknown_project_and_run(api):
    r = api.client.get("/api/v1/videos", params={"project": "nope"})
    assert (r.status_code, r.json()["error"]["code"]) == (404, "DATASET_NOT_FOUND")
    r = api.client.get("/api/v1/videos", params={"run": "nope"})
    assert (r.status_code, r.json()["error"]["code"]) == (404, "RUN_NOT_FOUND")


def test_search_and_paging(api):
    _fill(api)

    found = api.videos(q="P0")
    page = api.videos(limit=1, offset=1)

    assert {v["name"] for v in found["videos"]} == {"p01.mp4", "p02.mp4"}
    assert (page["total"], len(page["videos"])) == (3, 1)


def test_two_projects_with_one_name_carry_their_owners(api):
    _fill(api)
    api.project("QS")
    api.as_user({"id": "22222222-2222-2222-2222-222222222222", "username": "ana"})
    api.project("QS")

    projects = api.videos()["counts"]["projects"]

    assert [p["name"] for p in projects] == ["QS", "QS"]
    assert all("owner_name" in p for p in projects)
