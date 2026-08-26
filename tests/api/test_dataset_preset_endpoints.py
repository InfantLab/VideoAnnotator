"""API tests for the saved-dataset and saved-preset endpoints (spec 007).

Covers User Story 1 (save/list a dataset), User Story 2 (save/apply a preset,
including a verbatim multi-line prompt), User Story 3 (shared visibility,
owner-only mutation), and the FR-006/FR-007/FR-009 edge cases (export/import
round-trip, name-collision rejection, unavailable-pipeline flagging).
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from videoannotator.api.main import create_app
from videoannotator.api.middleware.auth import validate_api_key

OWNER_USER = {"id": "owner-1", "username": "alice", "is_admin": False}
OTHER_USER = {"id": "other-1", "username": "bob", "is_admin": False}
ADMIN_USER = {"id": "admin-1", "username": "root", "is_admin": True}


@pytest.fixture
def temp_db(monkeypatch, tmp_path):
    """Point the app's SQLAlchemy engine at a fresh, empty temp SQLite file."""
    import videoannotator.database.database as db_module

    db_path = tmp_path / "test_datasets_presets.db"
    new_engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    monkeypatch.setattr(db_module, "engine", new_engine)
    monkeypatch.setattr(
        db_module,
        "SessionLocal",
        sessionmaker(autocommit=False, autoflush=False, bind=new_engine),
    )
    db_module.Base.metadata.create_all(bind=new_engine)
    yield new_engine


@pytest.fixture
def client(temp_db):
    app = create_app()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _as(client: TestClient, user: dict) -> TestClient:
    """Switch the shared TestClient's authenticated identity in place.

    `owner_client`/`other_client`/`admin_client` all wrap the *same*
    underlying app instance (one per test, from the `client` fixture), so
    this must be called again immediately before each call made "as" a
    different user within a single test -- fixtures resolved once at the
    start of the test would all just overwrite the same override, with
    whichever resolves last silently winning for every call in the test.
    """
    client.app.dependency_overrides[validate_api_key] = lambda: user
    return client


@pytest.fixture
def owner_client(client):
    return _as(client, OWNER_USER)


@pytest.fixture
def other_client(client):
    return _as(client, OTHER_USER)


@pytest.fixture
def admin_client(client):
    return _as(client, ADMIN_USER)


class TestSavedDatasets:
    def test_save_then_list_returns_same_manifest(self, owner_client):
        """US1 acceptance scenario 1/2: save a named dataset, list it back."""
        manifest = [
            {"filename": "vid001.mp4", "size_bytes": 100},
            {"filename": "vid002.mp4", "size_bytes": 200},
        ]
        create_resp = owner_client.post(
            "/api/v1/datasets/",
            json={
                "name": "Corpus A",
                "description": "test set",
                "video_manifest": manifest,
            },
        )
        assert create_resp.status_code == 201
        created = create_resp.json()
        assert created["name"] == "Corpus A"
        assert [
            {"filename": e["filename"], "size_bytes": e["size_bytes"]}
            for e in created["video_manifest"]
        ] == manifest

        list_resp = owner_client.get("/api/v1/datasets/")
        assert list_resp.status_code == 200
        body = list_resp.json()
        assert body["total"] == 1
        assert body["datasets"][0]["id"] == created["id"]

    def test_duplicate_name_for_same_owner_is_rejected(self, owner_client):
        """FR-007 / edge case: names unique per owner."""
        owner_client.post(
            "/api/v1/datasets/", json={"name": "Dup", "video_manifest": []}
        )
        resp = owner_client.post(
            "/api/v1/datasets/", json={"name": "Dup", "video_manifest": []}
        )
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "DATASET_NAME_CONFLICT"

    def test_same_name_different_owners_does_not_collide(self, client):
        """Edge case: names unique per owner, not globally."""
        r1 = _as(client, OWNER_USER).post(
            "/api/v1/datasets/", json={"name": "Shared Name", "video_manifest": []}
        )
        r2 = _as(client, OTHER_USER).post(
            "/api/v1/datasets/", json={"name": "Shared Name", "video_manifest": []}
        )
        assert r1.status_code == 201
        assert r2.status_code == 201

    def test_second_user_sees_it_but_cannot_modify_or_delete(self, client):
        """US3: shared-read visibility, owner-only mutation."""
        created = (
            _as(client, OWNER_USER)
            .post("/api/v1/datasets/", json={"name": "Team Set", "video_manifest": []})
            .json()
        )
        dataset_id = created["id"]

        _as(client, OTHER_USER)
        list_resp = client.get("/api/v1/datasets/")
        assert any(d["id"] == dataset_id for d in list_resp.json()["datasets"])

        update_resp = client.put(
            f"/api/v1/datasets/{dataset_id}", json={"name": "Hijacked"}
        )
        assert update_resp.status_code == 403

        delete_resp = client.delete(f"/api/v1/datasets/{dataset_id}")
        assert delete_resp.status_code == 403

        still_there = _as(client, OWNER_USER).get(f"/api/v1/datasets/{dataset_id}")
        assert still_there.json()["name"] == "Team Set"

    def test_admin_can_modify_another_users_dataset(self, client):
        created = (
            _as(client, OWNER_USER)
            .post(
                "/api/v1/datasets/", json={"name": "Admin Target", "video_manifest": []}
            )
            .json()
        )
        resp = _as(client, ADMIN_USER).put(
            f"/api/v1/datasets/{created['id']}", json={"description": "edited by admin"}
        )
        assert resp.status_code == 200
        assert resp.json()["description"] == "edited by admin"

    def test_empty_manifest_is_a_valid_listable_dataset(self, owner_client):
        """Edge case: a dataset with all videos removed remains valid, not an error."""
        created = owner_client.post(
            "/api/v1/datasets/", json={"name": "Now Empty", "video_manifest": []}
        )
        assert created.status_code == 201
        assert created.json()["video_manifest"] == []

    def test_export_then_import_round_trip_by_different_user(self, client):
        """FR-006/SC-004: GET output is directly POST-able by another user,
        producing an equivalent, immediately usable saved item with a fresh
        id and the importing user as owner."""
        manifest = [{"filename": "a.mp4", "size_bytes": 1}]
        original = (
            _as(client, OWNER_USER)
            .post(
                "/api/v1/datasets/",
                json={
                    "name": "Exportable",
                    "description": "d",
                    "video_manifest": manifest,
                },
            )
            .json()
        )

        exported = client.get(f"/api/v1/datasets/{original['id']}").json()
        imported = _as(client, OTHER_USER).post("/api/v1/datasets/", json=exported)

        assert imported.status_code == 201
        imported_body = imported.json()
        assert imported_body["id"] != original["id"]
        assert imported_body["owner_user_id"] == OTHER_USER["id"]
        assert imported_body["name"] == "Exportable"
        assert [
            {"filename": e["filename"], "size_bytes": e["size_bytes"]}
            for e in imported_body["video_manifest"]
        ] == manifest

    def test_delete_does_not_error_and_is_gone_from_list(self, owner_client):
        """FR-008 (dataset side): deleting removes it cleanly; nothing here
        references a job, so there is nothing to corrupt or orphan."""
        created = owner_client.post(
            "/api/v1/datasets/", json={"name": "To Delete", "video_manifest": []}
        ).json()
        resp = owner_client.delete(f"/api/v1/datasets/{created['id']}")
        assert resp.status_code == 204
        assert owner_client.get(f"/api/v1/datasets/{created['id']}").status_code == 404


class TestSavedPresets:
    def test_save_then_retrieve_reproduces_multiline_prompt_verbatim(
        self, owner_client
    ):
        """US2 acceptance scenario 1/2: exact config, including a multi-line
        prompt field, reproduced verbatim (SC-002)."""
        prompt = "Line one.\nLine two, with detail.\nLine three: be precise."
        config = {"vlm_annotation": {"model": "qwen3.5:9b", "prompt": prompt}}
        created = owner_client.post(
            "/api/v1/presets/",
            json={
                "name": "Prompt Preset",
                "selected_pipelines": ["vlm_annotation"],
                "config": config,
            },
        ).json()

        retrieved = owner_client.get(f"/api/v1/presets/{created['id']}").json()
        assert retrieved["config"] == config
        assert retrieved["config"]["vlm_annotation"]["prompt"] == prompt
        assert retrieved["selected_pipelines"] == ["vlm_annotation"]

    def test_duplicate_name_for_same_owner_is_rejected(self, owner_client):
        owner_client.post("/api/v1/presets/", json={"name": "Dup"})
        resp = owner_client.post("/api/v1/presets/", json={"name": "Dup"})
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "PRESET_NAME_CONFLICT"

    def test_second_user_sees_it_but_cannot_modify_or_delete(self, client):
        created = (
            _as(client, OWNER_USER)
            .post("/api/v1/presets/", json={"name": "Team Preset"})
            .json()
        )
        preset_id = created["id"]

        _as(client, OTHER_USER)
        list_resp = client.get("/api/v1/presets/")
        assert any(p["id"] == preset_id for p in list_resp.json()["presets"])

        assert (
            client.put(f"/api/v1/presets/{preset_id}", json={"name": "x"}).status_code
            == 403
        )
        assert client.delete(f"/api/v1/presets/{preset_id}").status_code == 403

    def test_unavailable_referenced_pipeline_is_flagged_not_a_failure(
        self, owner_client
    ):
        """Edge case + FR-009: retrieval succeeds even when a referenced
        pipeline is no longer available, flagging it instead of failing."""
        fake_registry = SimpleNamespace(
            load=lambda: None,
            get=lambda name: (
                SimpleNamespace(requires_extras=[])
                if name == "available_pipeline"
                else None
            ),
        )
        created = owner_client.post(
            "/api/v1/presets/",
            json={
                "name": "Mixed Availability",
                "selected_pipelines": ["available_pipeline", "missing_pipeline"],
            },
        ).json()

        with (
            patch(
                "videoannotator.api.v1.presets.get_registry", return_value=fake_registry
            ),
            patch("videoannotator.api.v1.presets.extras_available", return_value=True),
        ):
            resp = owner_client.get(f"/api/v1/presets/{created['id']}")

        assert resp.status_code == 200
        assert resp.json()["unavailable_pipelines"] == ["missing_pipeline"]

    def test_export_then_import_round_trip_by_different_user(self, client):
        """FR-006/SC-004, preset side."""
        config = {"vlm_annotation": {"prompt": "reuse me"}}
        original = (
            _as(client, OWNER_USER)
            .post(
                "/api/v1/presets/",
                json={
                    "name": "Exportable Preset",
                    "selected_pipelines": ["vlm_annotation"],
                    "config": config,
                    "tags": {"model": "x"},
                },
            )
            .json()
        )

        exported = client.get(f"/api/v1/presets/{original['id']}").json()
        imported = _as(client, OTHER_USER).post("/api/v1/presets/", json=exported)

        assert imported.status_code == 201
        imported_body = imported.json()
        assert imported_body["id"] != original["id"]
        assert imported_body["owner_user_id"] == OTHER_USER["id"]
        assert imported_body["config"] == config
        assert imported_body["tags"] == {"model": "x"}

    def test_delete_does_not_error_and_is_gone_from_list(self, owner_client):
        created = owner_client.post(
            "/api/v1/presets/", json={"name": "To Delete"}
        ).json()
        resp = owner_client.delete(f"/api/v1/presets/{created['id']}")
        assert resp.status_code == 204
        assert owner_client.get(f"/api/v1/presets/{created['id']}").status_code == 404


class TestOwnerIdentityRequired:
    def test_anonymous_create_is_rejected_with_clear_error(self, client):
        """No auth dependency override => validate_api_key returns None in
        dev/no-auth mode; saving still requires a resolvable owner."""
        client.app.dependency_overrides[validate_api_key] = lambda: None
        resp = client.post(
            "/api/v1/datasets/", json={"name": "Anon", "video_manifest": []}
        )
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "OWNER_IDENTITY_REQUIRED"
