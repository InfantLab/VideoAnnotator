"""Videos as groups of paths, their per-pipeline results, and runs (spec 029
R1, R2, R12)."""

from __future__ import annotations

import os

from tests.fixtures.video_index import JobsDb, write_video
from videoannotator import video_index
from videoannotator.video_index import (
    DONE,
    FAILED,
    RUNNING,
    UPLOADS,
    WAITING,
    build_index,
    latest_pairs,
    run_videos,
    runs,
)


def _index(jobs: JobsDb, extra=()):
    return build_index(
        jobs.storage.job_summaries(), extra_paths=extra, uploads_root=jobs.storage_root
    )


# -- grouping (T005) -------------------------------------------------------


def test_an_upload_and_its_hard_linked_rerun_are_one_video(tmp_path):
    jobs = JobsDb(tmp_path)
    first, first_copy = jobs.upload("clip.mp4")
    second, second_copy = jobs.rerun(first)

    index = _index(jobs)

    [video] = index.videos()
    assert video.paths == [str(second_copy.resolve()), str(first_copy.resolve())]
    assert video.path == str(second_copy.resolve())
    assert [j.job_id for j in video.jobs] == [second.job_id, first.job_id]
    assert index.video_for(first_copy) is video
    assert video.folder == UPLOADS
    assert video.name == "clip.mp4"


def test_after_deleting_the_first_copy_the_rerun_chain_still_joins(tmp_path):
    jobs = JobsDb(tmp_path)
    first, first_copy = jobs.upload("clip.mp4")
    _, second_copy = jobs.rerun(first)
    os.unlink(first_copy)

    index = _index(jobs)

    [video] = index.videos()
    assert video.path == str(second_copy.resolve())
    assert video.available
    assert index.video_for(first_copy) is video


def test_two_uploads_with_the_same_recorded_sha256_are_one(tmp_path):
    jobs = JobsDb(tmp_path)
    jobs.upload("a.mp4", pipelines={"face_analysis": ("completed", None, "f00d")})
    jobs.upload("b.mp4", pipelines={"scene_detection": ("completed", None, "f00d")})

    [video] = _index(jobs).videos()

    assert set(video.results) == {"face_analysis", "scene_detection"}


def test_different_files_with_the_same_name_and_size_stay_two(tmp_path):
    jobs = JobsDb(tmp_path)
    one = tmp_path / "site_a" / "p01.mp4"
    two = tmp_path / "site_b" / "p01.mp4"
    one.parent.mkdir()
    two.parent.mkdir()
    one.write_bytes(b"recording 1")
    two.write_bytes(b"recording 2")
    jobs.make_job(one)
    jobs.make_job(two)

    videos = _index(jobs).videos()

    assert len(videos) == 2
    assert {v.folder for v in videos} == {str(one.parent), str(two.parent)}


def test_a_symlink_and_its_target_are_one(tmp_path):
    jobs = JobsDb(tmp_path)
    target = write_video(tmp_path / "data" / "p01.mp4")
    link = tmp_path / "data" / "link.mp4"
    link.symlink_to(target)
    jobs.make_job(target)
    jobs.make_job(link)

    [video] = _index(jobs).videos()

    assert video.paths == [str(target.resolve())]
    assert len(video.jobs) == 2


def test_a_missing_file_stays_listed_and_says_so(tmp_path):
    jobs = JobsDb(tmp_path)
    jobs.make_job(tmp_path / "gone" / "p01.mp4")

    [video] = _index(jobs).videos()

    assert video.available is False
    assert video.size_bytes is None


def test_extra_paths_join_by_inode_and_leave_the_original_alone(tmp_path):
    jobs = JobsDb(tmp_path)
    _, stored = jobs.upload("clip.mp4")
    elsewhere = tmp_path / "data" / "clip_link.mp4"
    elsewhere.parent.mkdir()
    os.link(stored, elsewhere)
    fresh = write_video(tmp_path / "data" / "never_run.mp4")
    index = _index(jobs)

    bigger = index.extended([str(elsewhere), str(fresh)])

    assert len(index.videos()) == 1
    assert len(bigger.videos()) == 2
    assert bigger.video_for(elsewhere) is bigger.video_for(stored)
    assert bigger.video_for(fresh).results == {}


# -- results (T006) --------------------------------------------------------


def test_results_take_the_newest_finished_run_of_each_pipeline(tmp_path):
    jobs = JobsDb(tmp_path)
    video = write_video(tmp_path / "data" / "p01.mp4")
    done = jobs.make_job(
        video, batch_id="b1", pipelines={"face_analysis": ("completed", None, None)}
    )
    jobs.make_job(
        video,
        batch_id="b2",
        pipelines={
            "face_analysis": ("failed", "Out of memory", None),
            "speech_recognition": ("failed", "No audio", None),
        },
    )
    jobs.make_job(video, pipelines={"scene_detection": ("running", None, None)})

    [found] = _index(jobs).videos()

    faces = found.results["face_analysis"]
    assert (faces.state, faces.run_id, faces.job_id) == (DONE, "b1", done.job_id)
    assert faces.finished_at is not None
    speech = found.results["speech_recognition"]
    assert (speech.state, speech.error) == (FAILED, "No audio")
    assert found.results["scene_detection"].state == RUNNING
    assert found.last_run_at == found.jobs[0].created_at


def test_a_waiting_job_shows_its_pipelines_waiting(tmp_path):
    jobs = JobsDb(tmp_path)
    video = write_video(tmp_path / "data" / "p01.mp4")
    job = jobs.make_job(video, save=False)
    job.selected_pipelines = ["face_analysis"]
    jobs.storage.save_job_metadata(job)

    [found] = _index(jobs).videos()

    assert found.results["face_analysis"].state == WAITING


def test_a_cancelled_run_shows_as_stopped(tmp_path):
    jobs = JobsDb(tmp_path)
    video = write_video(tmp_path / "data" / "p01.mp4")
    jobs.make_job(video, pipelines={"face_analysis": ("cancelled", None, None)})

    [found] = _index(jobs).videos()

    faces = found.results["face_analysis"]
    assert (faces.state, faces.error) == (FAILED, video_index.STOPPED_REASON)


def test_latest_pairs_pick_the_newest_completed_job(tmp_path):
    jobs = JobsDb(tmp_path)
    video = write_video(tmp_path / "data" / "p01.mp4")
    jobs.make_job(video, pipelines={"face_analysis": ("completed", None, None)})
    newer = jobs.make_job(video, pipelines={"face_analysis": ("completed", None, None)})
    jobs.make_job(video, pipelines={"face_analysis": ("failed", "x", None)})
    index = _index(jobs)

    assert latest_pairs(index, [v.key for v in index.videos()]) == {
        (newer.job_id, "face_analysis")
    }


def test_the_cache_rebuilds_when_the_jobs_change(tmp_path):
    video_index.clear_cache()
    jobs = JobsDb(tmp_path)
    jobs.make_job(write_video(tmp_path / "a.mp4"))
    first = video_index.cached_index(jobs.storage)
    assert video_index.cached_index(jobs.storage) is first

    jobs.make_job(write_video(tmp_path / "b.mp4"))

    assert len(video_index.cached_index(jobs.storage).videos()) == 2


# -- runs (T009) -----------------------------------------------------------


def test_runs_group_jobs_by_batch_with_counts(tmp_path):
    jobs = JobsDb(tmp_path)
    a = write_video(tmp_path / "data" / "a.mp4")
    b = write_video(tmp_path / "data" / "b.mp4")
    jobs.make_job(
        a,
        batch_id="b1",
        batch_name="Peekaboo faces",
        dataset_id="d1",
        pipelines={"face_analysis": ("completed", None, None)},
    )
    jobs.make_job(
        b, batch_id="b1", pipelines={"face_analysis": ("running", None, None)}
    )
    solo = jobs.make_job(a, pipelines={"scene_detection": ("completed", None, None)})

    index = _index(jobs)
    found = {r.id: r for r in runs(index)}

    batch = found["b1"]
    assert batch.name == "Peekaboo faces"
    assert (batch.total, batch.done, batch.running) == (2, 1, 1)
    assert batch.active
    assert batch.finished_at is None
    assert batch.project_id == "d1"
    assert batch.pipelines == ["face_analysis"]
    single = found[solo.job_id]
    assert single.name == "Run of a.mp4"
    assert not single.active and single.finished_at is not None
    assert [v.name for v in run_videos(index, "b1")] == ["a.mp4", "b.mp4"]
    assert runs(index)[0].id == solo.job_id
