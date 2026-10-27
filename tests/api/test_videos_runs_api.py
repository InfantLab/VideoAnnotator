"""GET /videos/all-runs: every run, those going on first (UAT D2)."""

from __future__ import annotations

from datetime import datetime, timedelta

from tests.fixtures.video_index import write_video
from videoannotator.batch.types import JobStatus


def _runs(api, **params):
    r = api.client.get("/api/v1/videos/all-runs", params=params)
    assert r.status_code == 200, r.text
    return r.json()


def test_no_runs(api):
    assert _runs(api) == {"runs": [], "total": 0}


def test_running_first_then_newest_with_videos_and_why_one_failed(api):
    a = write_video(api.shared / "a.mp4")
    b = write_video(api.shared / "b.mp4")
    now = datetime.now()
    api.jobs.make_job(
        a,
        batch_id="old",
        batch_name="Old",
        created_at=now - timedelta(days=3),
        pipelines={"face_analysis": ("completed", None, None)},
    )
    api.jobs.make_job(
        b,
        batch_id="old",
        batch_name="Old",
        created_at=now - timedelta(days=3),
        status=JobStatus.FAILED,
        pipelines={"face_analysis": ("failed", "No faces model", None)},
    )
    api.jobs.make_job(
        a,
        batch_id="new",
        batch_name="New",
        created_at=now - timedelta(minutes=5),
        pipelines={"scene_detection": ("completed", None, None)},
    )
    api.jobs.make_job(
        b,
        batch_id="going",
        batch_name="Going",
        created_at=now - timedelta(days=1),
        pipelines={"scene_detection": ("running", None, None)},
    )

    body = _runs(api)

    assert body["total"] == 3
    assert [r["id"] for r in body["runs"]] == ["going", "new", "old"]
    old = body["runs"][2]
    assert sorted(old["videos"]) == ["a.mp4", "b.mp4"]
    assert (old["done"], old["failed"], old["failure"]) == (1, 1, "No faces model")
    assert body["runs"][0]["active"] is True
    assert body["runs"][1]["failure"] is None
    assert [r["id"] for r in _runs(api, limit=1, offset=1)["runs"]] == ["new"]


def test_videos_by_their_last_run(api):
    a = write_video(api.shared / "a.mp4")
    b = write_video(api.shared / "b.mp4")
    write_video(api.shared / "never.mp4")
    now = datetime.now()
    api.jobs.make_job(a, created_at=now - timedelta(days=2))
    api.jobs.make_job(b, created_at=now - timedelta(hours=1))

    r = api.client.get(
        "/api/v1/videos",
        params={"sort": "recent", "limit": 5, "folder": str(api.shared)},
    )

    assert r.status_code == 200, r.text
    assert [v["name"] for v in r.json()["videos"]] == ["b.mp4", "a.mp4"]
