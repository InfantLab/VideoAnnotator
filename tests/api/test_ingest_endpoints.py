"""API tests for server-side folder ingest.

Creating jobs from videos already on the server's disk is the alternative to
uploading a corpus one multipart request at a time. Because it reads the
filesystem on a caller's instruction, the guards matter as much as the happy
path, so both are covered here.

Follows tests/api/test_batch_endpoints.py: the session-scoped `test_storage_env`
fixture (tests/conftest.py) isolates storage, so a bare TestClient plus a
per-test `reset_storage_backend()` is enough.
"""

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.api.main import app
from videoannotator.api.middleware.auth import validate_required_api_key
from videoannotator.api.v1 import ingest as ingest_module

client = TestClient(app)

ADMIN_USER = {"id": "admin-1", "username": "root", "is_admin": True}
NON_ADMIN_USER = {"id": "user-1", "username": "irene", "is_admin": False}


@pytest.fixture(autouse=True)
def reset_db():
    reset_storage_backend()
    yield
    reset_storage_backend()


@pytest.fixture(autouse=True)
def as_admin():
    """Ingest is admin-only; most tests are about what it does once past that."""
    app.dependency_overrides[validate_required_api_key] = lambda: ADMIN_USER
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A folder of videos the server is allowed to read."""
    root = tmp_path / "research"
    corpus_dir = root / "corpus"
    corpus_dir.mkdir(parents=True)

    for i in range(1, 13):
        (corpus_dir / f"dyad_{i:02d}.mp4").write_bytes(b"fake video bytes")
    (corpus_dir / "notes.txt").write_text("not a video")
    (corpus_dir / "empty.mp4").write_bytes(b"")

    nested = corpus_dir / "session_two"
    nested.mkdir()
    (nested / "dyad_13.mp4").write_bytes(b"fake video bytes")

    monkeypatch.setattr(ingest_module, "INGEST_ROOTS", str(root))
    return corpus_dir


def _ingest(path, **kwargs):
    body = {"path": str(path), **kwargs}
    return client.post("/api/v1/ingest", json=body)


class TestIngestCreatesJobsWithoutUploading:
    def test_one_call_turns_a_folder_into_a_batch(self, corpus):
        response = _ingest(corpus, selected_pipelines=["stub_pipeline"])
        assert response.status_code == 201, response.text

        body = response.json()
        # 12 real videos; notes.txt isn't a video, empty.mp4 is skipped.
        assert body["total"] == 12
        assert len(body["created"]) == 12
        assert [s["filename"] for s in body["skipped"]] == ["empty.mp4"]
        assert "empty" in body["skipped"][0]["reason"]

    def test_jobs_reference_the_original_files_and_are_not_copied(self, corpus):
        body = _ingest(corpus, selected_pipelines=["stub_pipeline"]).json()

        storage = get_storage_backend()
        job = storage.load_job_metadata(body["created"][0])
        assert job.video_path.parent == corpus
        assert job.video_path.exists()
        # Outputs still get their own directory, separate from the source.
        assert job.storage_path is not None
        assert Path(job.storage_path) != corpus

    def test_deleting_an_ingested_job_leaves_the_original_video_alone(self, corpus):
        body = _ingest(corpus, selected_pipelines=["stub_pipeline"]).json()
        storage = get_storage_backend()
        job = storage.load_job_metadata(body["created"][0])
        original = job.video_path
        assert original.exists()

        assert client.delete(f"/api/v1/jobs/{body['created'][0]}").status_code == 204
        assert original.exists(), "ingest must never delete a researcher's video"

    def test_all_jobs_share_one_batch_the_summary_can_report(self, corpus):
        body = _ingest(corpus, selected_pipelines=["stub_pipeline"]).json()

        summary = client.get(f"/api/v1/batches/{body['batch_id']}").json()
        assert summary["total"] == 12
        assert summary["by_status"]["pending"] == 12
        # The folder names the run when the caller doesn't.
        assert summary["batch_name"] == "corpus"

    def test_caller_can_name_the_batch_and_supply_its_id(self, corpus):
        batch_id = str(uuid.uuid4())
        body = _ingest(
            corpus,
            selected_pipelines=["stub_pipeline"],
            batch_id=batch_id,
            batch_name="Irene corpus",
        ).json()

        assert body["batch_id"] == batch_id
        assert (
            client.get(f"/api/v1/batches/{batch_id}").json()["batch_name"]
            == "Irene corpus"
        )

    def test_ingested_jobs_are_ordinary_jobs(self, corpus):
        body = _ingest(corpus, selected_pipelines=["stub_pipeline"]).json()
        job = client.get(f"/api/v1/jobs/{body['created'][0]}").json()

        assert job["status"] == "pending"
        assert job["selected_pipelines"] == ["stub_pipeline"]
        assert job["batch_id"] == body["batch_id"]
        assert job["video_filename"].endswith(".mp4")

    def test_recursive_reaches_subfolders_and_default_does_not(self, corpus):
        shallow = _ingest(corpus, selected_pipelines=["stub_pipeline"]).json()
        assert shallow["total"] == 12

        deep = _ingest(
            corpus, selected_pipelines=["stub_pipeline"], recursive=True
        ).json()
        assert deep["total"] == 13

    def test_config_is_carried_onto_every_job(self, corpus):
        """One config, shared by every video in the run -- the whole point of
        submitting a corpus as a batch."""
        config = {"vlm_annotation": {"sampling_mode": "single_frame"}}
        body = _ingest(
            corpus, selected_pipelines=["vlm_annotation"], config=config
        ).json()

        assert body["total"] == 12
        for job_id in body["created"]:
            job = client.get(f"/api/v1/jobs/{job_id}").json()
            assert job["config"] == config


class TestIngestRejectsUnusableRequests:
    def test_empty_folder_is_a_clear_error_not_an_empty_batch(self, corpus, tmp_path):
        empty = corpus / "nothing_here"
        empty.mkdir()

        response = _ingest(empty, selected_pipelines=["stub_pipeline"])
        assert response.status_code == 422
        assert "No videos found" in response.text

    def test_a_file_is_not_a_folder(self, corpus):
        response = _ingest(corpus / "dyad_01.mp4", selected_pipelines=["stub_pipeline"])
        assert response.status_code == 422
        assert "Not a folder" in response.text

    def test_missing_folder_is_reported_as_missing(self, corpus):
        response = _ingest(
            corpus / "no_such_folder", selected_pipelines=["stub_pipeline"]
        )
        assert response.status_code == 404

    def test_unavailable_pipeline_fails_before_any_job_is_created(self, corpus):
        """Discovering the pipeline isn't installed after creating 12 jobs
        would be worse than useless."""
        before = len(get_storage_backend().list_jobs())
        # Forced unavailable, so the test doesn't depend on which extras this
        # environment happens to have installed.
        with patch("videoannotator.api.v1.jobs.extras_available", return_value=False):
            response = _ingest(corpus, selected_pipelines=["speaker_diarization"])
        assert response.status_code == 422
        assert len(get_storage_backend().list_jobs()) == before


class TestIngestPathGuards:
    def test_path_outside_the_allowed_roots_is_refused(self, corpus, tmp_path):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "secret.mp4").write_bytes(b"fake video bytes")

        response = _ingest(outside, selected_pipelines=["stub_pipeline"])
        assert response.status_code == 403
        assert "outside the folders" in response.text
        # The error says how to allow it rather than leaving the user stuck.
        assert "VIDEOANNOTATOR_INGEST_ROOTS" in response.text

    def test_traversal_out_of_a_root_is_refused(self, corpus, tmp_path):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        (outside / "secret.mp4").write_bytes(b"fake video bytes")

        # Resolution happens before the check, so '..' cannot escape.
        traversal = corpus / ".." / ".." / "elsewhere"
        response = _ingest(traversal, selected_pipelines=["stub_pipeline"])
        assert response.status_code == 403

    def test_root_itself_is_allowed(self, corpus, monkeypatch):
        root = corpus.parent
        (root / "loose.mp4").write_bytes(b"fake video bytes")

        response = _ingest(root, selected_pipelines=["stub_pipeline"])
        assert response.status_code == 201

    def test_defaults_to_the_home_directory_when_unconfigured(self, monkeypatch):
        monkeypatch.setattr(ingest_module, "INGEST_ROOTS", "")
        assert ingest_module.allowed_roots() == [Path.home().resolve()]

    def test_multiple_roots_can_be_configured(self, monkeypatch, tmp_path):
        import os

        a, b = tmp_path / "a", tmp_path / "b"
        a.mkdir()
        b.mkdir()
        monkeypatch.setattr(ingest_module, "INGEST_ROOTS", f"{a}{os.pathsep}{b}")

        roots = ingest_module.allowed_roots()
        assert a.resolve() in roots and b.resolve() in roots


class TestIngestBrowse:
    def test_browsing_without_a_path_lists_the_roots(self, corpus):
        body = client.get("/api/v1/ingest/browse").json()
        assert body["path"] is None
        assert body["roots"] == [str(corpus.parent)]
        assert [d["path"] for d in body["directories"]] == [str(corpus.parent)]

    def test_browsing_a_folder_lists_videos_and_subfolders(self, corpus):
        body = client.get("/api/v1/ingest/browse", params={"path": str(corpus)}).json()

        assert body["path"] == str(corpus)
        assert body["video_count"] == 13  # 12 + the empty one, which is still a video
        assert "notes.txt" not in [v["name"] for v in body["videos"]]

        subdirs = {d["name"]: d for d in body["directories"]}
        assert "session_two" in subdirs
        assert subdirs["session_two"]["video_count"] == 1

    def test_a_folder_reports_a_browsable_parent(self, corpus):
        body = client.get("/api/v1/ingest/browse", params={"path": str(corpus)}).json()
        assert body["parent"] == str(corpus.parent)

    def test_a_root_reports_no_parent_to_escape_through(self, corpus):
        body = client.get(
            "/api/v1/ingest/browse", params={"path": str(corpus.parent)}
        ).json()
        assert body["parent"] is None

    def test_browsing_outside_the_roots_is_refused(self, corpus, tmp_path):
        outside = tmp_path / "elsewhere"
        outside.mkdir()
        response = client.get("/api/v1/ingest/browse", params={"path": str(outside)})
        assert response.status_code == 403


class TestIngestIsAdminOnly:
    def test_a_non_admin_is_refused(self, corpus):
        """Same bar as extras-install: turning server files into jobs is a
        privileged operation."""
        app.dependency_overrides[validate_required_api_key] = lambda: NON_ADMIN_USER
        response = _ingest(corpus, selected_pipelines=["stub_pipeline"])
        assert response.status_code == 403
        assert "Administrator" in response.text

    def test_a_non_admin_cannot_browse_the_filesystem_either(self, corpus):
        app.dependency_overrides[validate_required_api_key] = lambda: NON_ADMIN_USER
        response = client.get("/api/v1/ingest/browse")
        assert response.status_code == 403

    def test_no_jobs_are_created_when_refused(self, corpus):
        before = len(get_storage_backend().list_jobs())
        app.dependency_overrides[validate_required_api_key] = lambda: NON_ADMIN_USER
        _ingest(corpus, selected_pipelines=["stub_pipeline"])
        assert len(get_storage_backend().list_jobs()) == before


class TestIngestIsLocalOnly:
    def test_a_remote_caller_is_refused(self, corpus):
        """The feature exists for 'the server is on my machine'; a remote
        caller has no business naming local paths."""
        from fastapi import Request

        original = ingest_module.require_local_caller

        def fake_remote(request: Request) -> None:
            # Simulate the client address the guard actually inspects.
            request.scope["client"] = ("203.0.113.7", 51234)
            return original(request)

        ingest_module.require_local_caller = fake_remote
        try:
            response = _ingest(corpus, selected_pipelines=["stub_pipeline"])
            assert response.status_code == 403
            assert "same machine" in response.text
        finally:
            ingest_module.require_local_caller = original

    def test_loopback_addresses_are_accepted(self):
        from fastapi import Request

        for host in ("127.0.0.1", "::1", "testclient"):
            scope = {"type": "http", "client": (host, 1234), "headers": []}
            # Does not raise.
            ingest_module.require_local_caller(Request(scope))

    def test_a_routable_address_is_rejected(self):
        from fastapi import Request

        from videoannotator.api.errors import APIError

        scope = {"type": "http", "client": ("203.0.113.7", 1234), "headers": []}
        with pytest.raises(APIError):
            ingest_module.require_local_caller(Request(scope))
