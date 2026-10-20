"""GET /videos/activity: runs going on, and those just finished (spec 029 R17)."""

from __future__ import annotations

from datetime import datetime, timedelta

from tests.fixtures.video_index import write_video
from videoannotator.batch.types import JobStatus


def _activity(api):
    r = api.client.get("/api/v1/videos/activity")
    assert r.status_code == 200, r.text
    return r.json()


def test_nothing_going_on(api):
    assert _activity(api) == {"active": [], "recent": []}


def test_an_active_run_and_a_recent_one(api):
    a = write_video(api.shared / "a.mp4")
    b = write_video(api.shared / "b.mp4")
    project = api.project("QS")
    api.jobs.make_job(
        a,
        batch_id="b1",
        batch_name="Faces",
        dataset_id=project["id"],
        pipelines={"face_analysis": ("running", None, None)},
    )
    api.jobs.make_job(a, batch_id="b1", batch_name="Faces")
    now = datetime.now()
    api.jobs.make_job(
        b,
        batch_id="b2",
        batch_name="Done",
        created_at=now - timedelta(minutes=5),
        pipelines={"scene_detection": ("completed", None, None)},
    )
    api.jobs.make_job(
        b,
        batch_id="b3",
        batch_name="Yesterday",
        created_at=now - timedelta(days=2),
        pipelines={"scene_detection": ("completed", None, None)},
    )
    api.jobs.make_job(
        b,
        batch_id="b4",
        batch_name="Stopped",
        created_at=now - timedelta(minutes=1),
        status=JobStatus.CANCELLED,
    )

    body = _activity(api)

    [active] = body["active"]
    assert (active["id"], active["total"], active["running"], active["waiting"]) == (
        "b1",
        2,
        1,
        1,
    )
    assert active["project"] == {"id": project["id"], "name": "QS"}
    assert [r["id"] for r in body["recent"]] == ["b2"]
    assert body["recent"][0]["stopped"] is False
