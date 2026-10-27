"""POST /videos/run: one run on any videos the server can read (spec 029 R8)."""

from __future__ import annotations

from pathlib import Path

from tests.fixtures.video_index import write_video
from videoannotator.api.database import get_storage_backend


def _run(api, paths, **body):
    return api.client.post(
        "/api/v1/videos/run",
        json={
            "paths": [str(p) for p in paths],
            "selected_pipelines": ["scene_detection"],
        }
        | body,
    )


def _job(job_id):
    return get_storage_backend().load_job_metadata(job_id)


def test_a_folder_video_is_read_in_place(api):
    video = write_video(api.shared / "pk" / "p01.mp4")

    r = _run(api, [video], name="Peekaboo faces")

    assert r.status_code == 201, r.text
    body = r.json()
    assert body["name"] == "Peekaboo faces"
    [job_id] = body["created"]
    job = _job(job_id)
    assert Path(job.video_path) == video.resolve()
    assert job.batch_id == body["run_id"]
    assert job.batch_name == "Peekaboo faces"
    assert body["results_folder"]["path"]
    assert Path(job.output_dir).parent == Path(body["results_folder"]["path"])


def test_an_upload_is_hard_linked_and_stays_one_video(api):
    first, stored = api.jobs.upload("home.mp4")

    r = _run(api, [stored])

    assert r.status_code == 201, r.text
    job = _job(r.json()["created"][0])
    assert Path(job.video_path) != stored
    assert Path(job.video_path).stat().st_ino == stored.stat().st_ino
    [video] = api.videos()["videos"]
    assert len(video["paths"]) == 2


def test_a_mix_is_one_run_and_unusable_paths_are_skipped(api, tmp_path):
    folder_video = write_video(api.shared / "pk" / "p01.mp4")
    _, stored = api.jobs.upload("home.mp4")
    gone = api.shared / "pk" / "gone.mp4"
    private = write_video(tmp_path / "private" / "x.mp4")

    r = _run(api, [folder_video, stored, gone, private])

    assert r.status_code == 201, r.text
    body = r.json()
    assert len(body["created"]) == 2
    assert {_job(j).batch_id for j in body["created"]} == {body["run_id"]}
    reasons = {s["path"]: s["reason"] for s in body["skipped"]}
    assert "moved or deleted" in reasons[str(gone)]
    assert "isn't in a folder shared" in reasons[str(private)]
    run = api.videos(run=body["run_id"])
    assert run["total"] == 2


def test_nothing_usable(api):
    r = _run(api, [api.shared / "missing.mp4"])

    assert r.status_code == 422
    assert r.json()["error"]["code"] == "NOTHING_TO_RUN"


def test_an_unknown_pipeline_is_refused(api):
    video = write_video(api.shared / "p01.mp4")

    r = _run(api, [video], selected_pipelines=["no_such_pipeline"], config={"x": 1})

    assert r.status_code in (400, 422), r.text


def test_settings_are_remembered_and_videos_join_the_project(api):
    video = write_video(api.shared / "pk" / "p01.mp4")
    other = write_video(api.shared / "elsewhere" / "p02.mp4")
    project = api.project("QS", server_folder=str(api.shared / "pk"))
    config = {"scene_detection": {"threshold": 0.4}}

    r = _run(api, [video, other], project_id=project["id"], config=config)

    assert r.status_code == 201, r.text
    assert (
        r.json()["name"] == "QS · PySceneDetect + CLIP"
    )  # what it ran on, then what it found (UAT #41)
    job = _job(r.json()["created"][0])
    assert job.dataset_id == project["id"]
    held = api.videos(project=project["id"])
    assert {v["name"] for v in held["videos"]} == {"p01.mp4", "p02.mp4"}
    settings = api.client.get(
        "/api/v1/videos/run-settings", params={"project": project["id"]}
    ).json()
    assert settings["selected_pipelines"] == ["scene_detection"]
    assert settings["config"] == config
    assert settings["source"]["kind"] == "project"
    assert settings["source"]["from"]["kind"] == "run"
    last = api.client.get("/api/v1/videos/run-settings").json()
    assert last["source"]["kind"] == "last_used"
    assert last["source"]["name"] == "QS · PySceneDetect + CLIP"


def test_an_unknown_project(api):
    video = write_video(api.shared / "p01.mp4")

    r = _run(api, [video], project_id="nope")

    assert (r.status_code, r.json()["error"]["code"]) == (404, "DATASET_NOT_FOUND")
