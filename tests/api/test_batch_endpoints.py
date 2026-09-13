"""API tests for spec 008: submission batch tagging, aggregate batch status,
batch-retry, and real SSE job-status-changed events.

Follows tests/integration/test_job_cancellation.py's pattern: the session-scoped
`test_storage_env` fixture (tests/conftest.py) already isolates storage to a
temp dir for the whole test session, so a bare `TestClient(app)` + per-test
`reset_storage_backend()` is enough -- no separate temp_db machinery needed.
"""

import asyncio
import io
import json
import time
import uuid
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.v1 import events as events_module
from videoannotator.batch.types import JobStatus

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_db():
    """Reset storage before each test."""
    reset_storage_backend()
    yield
    reset_storage_backend()


def _submit_job(
    batch_id: str | None = None,
    dataset_id: str | None = None,
    batch_name: str | None = None,
) -> dict:
    data = {}
    if batch_id is not None:
        data["batch_id"] = batch_id
    if dataset_id is not None:
        data["dataset_id"] = dataset_id
    if batch_name is not None:
        data["batch_name"] = batch_name
    response = client.post(
        "/api/v1/jobs/",
        files={"video": ("test.mp4", io.BytesIO(b"fake video content"), "video/mp4")},
        data=data,
    )
    assert response.status_code == 201, response.text
    return response.json()


def _set_job_status(
    job_id: str,
    status: JobStatus,
    started_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> None:
    storage = get_storage_backend()
    job = storage.load_job_metadata(job_id)
    job.status = status
    if started_at is not None:
        job.started_at = started_at
    if completed_at is not None:
        job.completed_at = completed_at
    storage.save_job_metadata(job)


class TestBatchSubmissionTagging:
    def test_job_response_includes_batch_and_dataset_id(self):
        batch_id = str(uuid.uuid4())
        job = _submit_job(batch_id=batch_id, dataset_id="ds-123")
        assert job["batch_id"] == batch_id
        assert job["dataset_id"] == "ds-123"

    def test_omitting_batch_id_behaves_as_standalone_job(self):
        """API Contract: omitting batch_id behaves exactly as a standalone job."""
        job = _submit_job()
        assert job["batch_id"] is None
        assert job["dataset_id"] is None

    def test_status_endpoint_also_reports_batch_and_dataset_id(self):
        batch_id = str(uuid.uuid4())
        job = _submit_job(batch_id=batch_id)
        resp = client.get(f"/api/v1/jobs/{job['id']}")
        assert resp.json()["batch_id"] == batch_id


class TestBatchSummary:
    def test_counts_match_true_job_states(self):
        """US1 acceptance scenario 1 / SC-001."""
        batch_id = str(uuid.uuid4())
        job_ids = [_submit_job(batch_id=batch_id)["id"] for _ in range(3)]

        _set_job_status(job_ids[0], JobStatus.RUNNING, started_at=datetime.now())
        _set_job_status(
            job_ids[1],
            JobStatus.COMPLETED,
            started_at=datetime.now() - timedelta(seconds=10),
            completed_at=datetime.now(),
        )
        # job_ids[2] stays pending

        resp = client.get(f"/api/v1/batches/{batch_id}")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 3
        assert body["by_status"] == {
            "pending": 1,
            "running": 1,
            "completed": 1,
            "failed": 0,
            "cancelled": 0,
        }

    def test_eta_present_once_a_job_has_completed(self):
        """US1 acceptance scenario 2."""
        batch_id = str(uuid.uuid4())
        job_ids = [_submit_job(batch_id=batch_id)["id"] for _ in range(2)]
        _set_job_status(
            job_ids[0],
            JobStatus.COMPLETED,
            started_at=datetime.now() - timedelta(seconds=10),
            completed_at=datetime.now(),
        )
        # job_ids[1] stays pending

        resp = client.get(f"/api/v1/batches/{batch_id}")
        body = resp.json()
        assert body["estimated_seconds_remaining"] is not None
        assert body["estimated_seconds_remaining"] > 0

    def test_no_completed_jobs_yet_gives_null_eta(self):
        batch_id = str(uuid.uuid4())
        _submit_job(batch_id=batch_id)
        resp = client.get(f"/api/v1/batches/{batch_id}")
        assert resp.json()["estimated_seconds_remaining"] is None

    def test_fully_terminal_batch_has_no_eta(self):
        """US1 acceptance scenario 3."""
        batch_id = str(uuid.uuid4())
        job_ids = [_submit_job(batch_id=batch_id)["id"] for _ in range(2)]
        _set_job_status(
            job_ids[0],
            JobStatus.COMPLETED,
            started_at=datetime.now() - timedelta(seconds=5),
            completed_at=datetime.now(),
        )
        _set_job_status(job_ids[1], JobStatus.FAILED)

        resp = client.get(f"/api/v1/batches/{batch_id}")
        body = resp.json()
        assert body["completion_percentage"] == 100.0
        assert body["estimated_seconds_remaining"] is None

    def test_unknown_batch_id_returns_empty_summary_not_404(self):
        """Key Entities: a batch isn't a resource that can be "not found" --
        it's whatever jobs currently carry the identifier."""
        resp = client.get(f"/api/v1/batches/{uuid.uuid4()}")
        assert resp.status_code == 200
        assert resp.json()["total"] == 0

    def test_deleting_referenced_dataset_does_not_break_batch_summary(self):
        """FR-008/SC-005: dataset_id is a loose reference; a stale/deleted
        one must not affect batch summary."""
        batch_id = str(uuid.uuid4())
        _submit_job(batch_id=batch_id, dataset_id="a-dataset-id-that-does-not-exist")
        resp = client.get(f"/api/v1/batches/{batch_id}")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


class TestBatchRetry:
    def test_only_failed_jobs_are_retried(self):
        """US2 acceptance scenario 1 / SC-003."""
        batch_id = str(uuid.uuid4())
        failed_job = _submit_job(batch_id=batch_id)["id"]
        completed_job = _submit_job(batch_id=batch_id)["id"]
        _set_job_status(failed_job, JobStatus.FAILED)
        _set_job_status(
            completed_job,
            JobStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
        )

        resp = client.post(f"/api/v1/batches/{batch_id}/retry")
        assert resp.status_code == 200
        body = resp.json()
        assert body["retried"] == [failed_job]
        skipped_ids = [s["job_id"] for s in body["skipped"]]
        assert completed_job in skipped_ids

        # The failed job actually reset to pending, exactly as spec 006's
        # single-job retry would do.
        assert client.get(f"/api/v1/jobs/{failed_job}").json()["status"] == "pending"
        # The completed job is untouched.
        assert (
            client.get(f"/api/v1/jobs/{completed_job}").json()["status"] == "completed"
        )

    def test_running_job_in_batch_is_left_alone_not_errored(self):
        """US2 acceptance scenario 2."""
        batch_id = str(uuid.uuid4())
        running_job = _submit_job(batch_id=batch_id)["id"]
        _set_job_status(running_job, JobStatus.RUNNING, started_at=datetime.now())

        resp = client.post(f"/api/v1/batches/{batch_id}/retry")
        assert resp.status_code == 200
        body = resp.json()
        assert body["retried"] == []
        assert any(s["job_id"] == running_job for s in body["skipped"])
        # Untouched -- still running, not bumped back to pending.
        assert client.get(f"/api/v1/jobs/{running_job}").json()["status"] == "running"

    def test_nothing_retryable_reports_cleanly_not_an_error(self):
        """US2 acceptance scenario 3."""
        batch_id = str(uuid.uuid4())
        _submit_job(batch_id=batch_id)  # stays pending

        resp = client.post(f"/api/v1/batches/{batch_id}/retry")
        assert resp.status_code == 200
        body = resp.json()
        assert body["retried"] == []
        assert len(body["skipped"]) == 1


def _list_batches(per_page: int = 100) -> list[dict]:
    """All batch summaries. Storage is shared across this file's tests, so
    callers must assert on the batches they created, not on the whole list."""
    response = client.get(f"/api/v1/batches/?per_page={per_page}")
    assert response.status_code == 200, response.text
    return response.json()["batches"]


def _batch_ids() -> list[str]:
    return [batch["batch_id"] for batch in _list_batches()]


class TestBatchNaming:
    def test_batch_name_is_reported_on_the_job_and_the_batch(self):
        batch_id = str(uuid.uuid4())
        job = _submit_job(batch_id=batch_id, batch_name="Irene corpus run 3")
        assert job["batch_name"] == "Irene corpus run 3"

        summary = client.get(f"/api/v1/batches/{batch_id}").json()
        assert summary["batch_name"] == "Irene corpus run 3"

    def test_batch_name_without_batch_id_is_ignored(self):
        """A name only means something as a label for a batch."""
        job = _submit_job(batch_name="orphan label")
        assert job["batch_id"] is None
        assert job["batch_name"] is None

    def test_unnamed_batch_reports_a_null_name_not_an_error(self):
        batch_id = str(uuid.uuid4())
        _submit_job(batch_id=batch_id)
        summary = client.get(f"/api/v1/batches/{batch_id}").json()
        assert summary["batch_name"] is None

    def test_name_survives_when_only_some_member_jobs_carry_it(self):
        """Partially-tagged batch still reports its name rather than none."""
        batch_id = str(uuid.uuid4())
        _submit_job(batch_id=batch_id)
        _submit_job(batch_id=batch_id, batch_name="named later")
        summary = client.get(f"/api/v1/batches/{batch_id}").json()
        assert summary["batch_name"] == "named later"


class TestBatchListing:
    def test_lists_every_batch_with_its_aggregate(self):
        first = str(uuid.uuid4())
        second = str(uuid.uuid4())
        _submit_job(batch_id=first, batch_name="first batch")
        _submit_job(batch_id=second, batch_name="second batch")
        _submit_job(batch_id=second, batch_name="second batch")

        by_id = {batch["batch_id"]: batch for batch in _list_batches()}
        assert by_id[first]["total"] == 1
        assert by_id[second]["total"] == 2
        assert by_id[second]["batch_name"] == "second batch"
        assert by_id[second]["by_status"]["pending"] == 2

    def test_standalone_jobs_are_not_listed_as_batches(self):
        before = set(_batch_ids())
        _submit_job()
        _submit_job()
        assert set(_batch_ids()) == before

    def test_listing_is_always_a_list_never_an_error(self):
        response = client.get("/api/v1/batches/")
        assert response.status_code == 200
        assert isinstance(response.json()["batches"], list)

    def test_newest_batch_is_listed_before_an_older_one(self):
        older = str(uuid.uuid4())
        newer = str(uuid.uuid4())
        _submit_job(batch_id=older)
        _submit_job(batch_id=newer)
        listed = _batch_ids()
        assert listed.index(newer) < listed.index(older)

    def test_pagination_splits_the_listing_without_losing_batches(self):
        for _ in range(3):
            _submit_job(batch_id=str(uuid.uuid4()))

        body = client.get("/api/v1/batches/?page=1&per_page=2").json()
        assert body["page"] == 1
        assert body["per_page"] == 2
        assert len(body["batches"]) == 2
        assert body["total"] == len(_batch_ids())

        page_two = client.get("/api/v1/batches/?page=2&per_page=2").json()
        first_ids = {batch["batch_id"] for batch in body["batches"]}
        second_ids = {batch["batch_id"] for batch in page_two["batches"]}
        assert first_ids.isdisjoint(second_ids)

    def test_summary_carries_dataset_and_submission_time(self):
        batch_id = str(uuid.uuid4())
        _submit_job(batch_id=batch_id, dataset_id="ds-42")
        batch = next(b for b in _list_batches() if b["batch_id"] == batch_id)
        assert batch["dataset_id"] == "ds-42"
        assert batch["created_at"] is not None


class TestJobsFilteredByBatch:
    def test_batch_id_filter_returns_only_that_batch(self):
        mine = str(uuid.uuid4())
        theirs = str(uuid.uuid4())
        a = _submit_job(batch_id=mine)
        b = _submit_job(batch_id=mine)
        _submit_job(batch_id=theirs)
        _submit_job()

        body = client.get(f"/api/v1/jobs/?batch_id={mine}").json()
        assert body["total"] == 2
        assert {job["id"] for job in body["jobs"]} == {a["id"], b["id"]}

    def test_filter_composes_with_status_filter(self):
        batch_id = str(uuid.uuid4())
        pending = _submit_job(batch_id=batch_id)
        done = _submit_job(batch_id=batch_id)
        _set_job_status(done["id"], JobStatus.COMPLETED)

        body = client.get(
            f"/api/v1/jobs/?batch_id={batch_id}&status_filter=pending"
        ).json()
        assert [job["id"] for job in body["jobs"]] == [pending["id"]]

    def test_unknown_batch_id_returns_no_jobs_not_an_error(self):
        _submit_job(batch_id=str(uuid.uuid4()))
        body = client.get(f"/api/v1/jobs/?batch_id={uuid.uuid4()}").json()
        assert body["total"] == 0
        assert body["jobs"] == []

    def test_omitting_the_filter_still_returns_everything(self):
        before = client.get("/api/v1/jobs/").json()["total"]
        _submit_job(batch_id=str(uuid.uuid4()))
        _submit_job()
        assert client.get("/api/v1/jobs/").json()["total"] == before + 2


class TestUnbatchedJobsFilter:
    def test_returns_only_jobs_belonging_to_no_batch(self):
        _submit_job(batch_id=str(uuid.uuid4()))
        standalone = _submit_job()

        body = client.get("/api/v1/jobs/?unbatched_only=true&per_page=100").json()
        ids = {job["id"] for job in body["jobs"]}
        assert standalone["id"] in ids
        assert all(job["batch_id"] is None for job in body["jobs"])

    def test_survives_a_page_full_of_batched_jobs(self):
        """The reason this filter exists: filtering a paginated list
        client-side hides ungrouped jobs whenever batched ones fill the page."""
        batch_id = str(uuid.uuid4())
        for _ in range(12):
            _submit_job(batch_id=batch_id)
        standalone = _submit_job()

        body = client.get("/api/v1/jobs/?unbatched_only=true&per_page=10").json()
        assert standalone["id"] in {job["id"] for job in body["jobs"]}

    def test_batch_id_takes_precedence_when_both_are_given(self):
        batch_id = str(uuid.uuid4())
        job = _submit_job(batch_id=batch_id)
        _submit_job()

        body = client.get(
            f"/api/v1/jobs/?batch_id={batch_id}&unbatched_only=true&per_page=100"
        ).json()
        assert [j["id"] for j in body["jobs"]] == [job["id"]]

    def test_composes_with_status_filter(self):
        pending = _submit_job()
        done = _submit_job()
        _set_job_status(done["id"], JobStatus.COMPLETED)

        body = client.get(
            "/api/v1/jobs/?unbatched_only=true&status_filter=pending&per_page=100"
        ).json()
        ids = {job["id"] for job in body["jobs"]}
        assert pending["id"] in ids
        assert done["id"] not in ids

    def test_off_by_default(self):
        """Without the flag, batched jobs are included -- asserted on totals
        rather than page contents, since storage is shared across this file's
        tests and a specific job may not be on the first page."""
        batch_id = str(uuid.uuid4())
        _submit_job(batch_id=batch_id)

        everything = client.get("/api/v1/jobs/?per_page=1").json()["total"]
        unbatched = client.get("/api/v1/jobs/?unbatched_only=true&per_page=1").json()[
            "total"
        ]
        assert everything > unbatched


class TestBatchCancel:
    def test_cancels_every_pending_job_in_the_batch(self):
        batch_id = str(uuid.uuid4())
        a = _submit_job(batch_id=batch_id)
        b = _submit_job(batch_id=batch_id)

        body = client.post(f"/api/v1/batches/{batch_id}/cancel").json()
        assert set(body["cancelled"]) == {a["id"], b["id"]}
        assert body["skipped"] == []

        summary = client.get(f"/api/v1/batches/{batch_id}").json()
        assert summary["by_status"]["cancelled"] == 2

    def test_completed_job_is_skipped_with_a_reason_not_errored(self):
        batch_id = str(uuid.uuid4())
        running = _submit_job(batch_id=batch_id)
        finished = _submit_job(batch_id=batch_id)
        _set_job_status(finished["id"], JobStatus.COMPLETED)

        resp = client.post(f"/api/v1/batches/{batch_id}/cancel")
        assert resp.status_code == 200
        body = resp.json()
        assert body["cancelled"] == [running["id"]]
        assert [s["job_id"] for s in body["skipped"]] == [finished["id"]]
        assert "completed" in body["skipped"][0]["reason"]

    def test_running_job_is_cancelled(self):
        batch_id = str(uuid.uuid4())
        job = _submit_job(batch_id=batch_id)
        _set_job_status(job["id"], JobStatus.RUNNING)

        body = client.post(f"/api/v1/batches/{batch_id}/cancel").json()
        assert body["cancelled"] == [job["id"]]

    def test_cancelling_twice_is_idempotent(self):
        batch_id = str(uuid.uuid4())
        job = _submit_job(batch_id=batch_id)

        client.post(f"/api/v1/batches/{batch_id}/cancel")
        second = client.post(f"/api/v1/batches/{batch_id}/cancel")
        assert second.status_code == 200
        assert second.json()["cancelled"] == [job["id"]]
        assert second.json()["skipped"] == []

    def test_unknown_batch_cancels_nothing_without_erroring(self):
        resp = client.post(f"/api/v1/batches/{uuid.uuid4()}/cancel")
        assert resp.status_code == 200
        assert resp.json()["cancelled"] == []

    def test_cancelled_batch_is_then_retryable(self):
        """Cancel-all then retry-all is the 'stop, fix, rerun' loop."""
        batch_id = str(uuid.uuid4())
        _submit_job(batch_id=batch_id)
        _submit_job(batch_id=batch_id)

        client.post(f"/api/v1/batches/{batch_id}/cancel")
        retried = client.post(f"/api/v1/batches/{batch_id}/retry").json()
        assert len(retried["retried"]) == 2

        summary = client.get(f"/api/v1/batches/{batch_id}").json()
        assert summary["by_status"]["pending"] == 2


class TestSSEJobStatusChanged:
    """Exercises `event_stream()` (the async generator behind the /stream
    endpoint) directly rather than through TestClient's HTTP streaming --
    httpx/Starlette's synchronous test transport does not reliably surface
    partial SSE chunks from a long-lived generator with an internal sleep
    loop, which made an end-to-end HTTP version of this test hang
    indefinitely rather than fail fast. Testing the generator itself still
    genuinely covers FR-006's actual logic (poll, diff, emit) without
    depending on that transport quirk.
    """

    @pytest.mark.asyncio
    async def test_status_change_is_pushed_without_polling(self, monkeypatch):
        """US3 acceptance scenario 1 / SC-004: a subscribed client observes
        a job status change via the stream, without a separate poll."""
        monkeypatch.setattr(events_module, "_POLL_INTERVAL_SECONDS", 0.01)

        job = _submit_job()
        job_id = job["id"]

        gen = events_module.event_stream()
        try:
            first = await asyncio.wait_for(anext(gen), timeout=2.0)
            assert json.loads(first.removeprefix("data: "))["type"] == "connected"

            # Change status *after* connecting, so this is a genuine push,
            # not something already true before the subscription existed.
            _set_job_status(job_id, JobStatus.RUNNING, started_at=datetime.now())

            matching = None
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                raw = await asyncio.wait_for(anext(gen), timeout=2.0)
                if not raw.startswith("data: "):
                    continue
                event = json.loads(raw.removeprefix("data: "))
                if (
                    event.get("type") == "job_status_changed"
                    and event.get("job_id") == job_id
                ):
                    matching = event
                    break

            assert matching is not None, "no job_status_changed event observed"
            assert matching["status"] == "running"
            assert matching["batch_id"] is None
        finally:
            await gen.aclose()


def test_events_router_is_registered():
    """Sanity check that the events router is actually mounted at the
    expected path (regression guard for the /events prefix wiring in
    api/v1/__init__.py). Uses the static OpenAPI schema rather than an
    actual HTTP connection: opening a real connection to a StreamingResponse
    whose generator sleeps in a loop deadlocks Starlette's TestClient in
    this environment (its sync/background-thread transport doesn't return
    control once the handshake is waiting on the first chunk) -- a
    transport-level quirk, not something the endpoint's own logic (already
    covered directly, above) can be blamed for."""
    schema = app.openapi()
    assert "/api/v1/events/stream" in schema["paths"]
