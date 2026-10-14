"""Removed and deprecated pipelines (spec 014).

- Removed (laion_voice, face_laion_clip): every entry point explains the removal
  instead of "not found" or a traceback.
- Deprecated (audio_processing): still runs, isn't listed, warns.
- Family short names resolve to the family's declared default, whatever is installed.
"""

import io
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import reset_storage_backend, set_database_path
from videoannotator.api.main import create_app
from videoannotator.batch.job_execution import _unavailable_reason
from videoannotator.registry.pipeline_loader import (
    REMOVED_PIPELINES,
    deprecation_message,
    removed_pipeline_message,
)
from videoannotator.registry.pipeline_registry import get_registry


@pytest.fixture
def client():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".db") as f:
        db_path = Path(f.name)
    set_database_path(db_path)
    reset_storage_backend()
    yield TestClient(create_app())
    reset_storage_backend()
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def video():
    from tests.fixtures.synthetic_video import synthetic_video_bytes_avi

    return io.BytesIO(synthetic_video_bytes_avi())


@pytest.mark.parametrize("name", sorted(REMOVED_PIPELINES))
def test_api_rejects_removed_pipeline_with_reason(client, video, name):
    response = client.post(
        "/api/v1/jobs/",
        files={"video": ("test.avi", video, "video/avi")},
        data={"selected_pipelines": name},
    )
    assert response.status_code == 422
    text = response.text
    assert name in text
    assert "removed in v1.6.0" in text


@pytest.mark.parametrize("name", sorted(REMOVED_PIPELINES))
def test_batch_path_explains_removal(name):
    assert _unavailable_reason(name) == removed_pipeline_message(name)


def test_unknown_pipeline_is_not_reported_as_removed():
    assert removed_pipeline_message("no_such_pipeline") is None


def test_deprecated_pipeline_is_hidden_but_loadable():
    registry = get_registry()
    registry.load(force=True)
    listed = {m.name for m in registry.list()}
    everything = {m.name for m in registry.list(include_deprecated=True)}
    assert "audio_processing" not in listed
    assert "audio_processing" in everything
    meta = registry.get("audio_processing")
    message = deprecation_message(meta)
    assert message is not None
    assert "v1.7.0" in message
    assert "speech_recognition" in message
    assert "speaker_diarization" in message


def test_submitting_deprecated_pipeline_warns(client, video, monkeypatch):
    import videoannotator.api.v1.jobs as jobs_module

    monkeypatch.setattr(jobs_module, "extras_available", lambda extras: True)
    response = client.post(
        "/api/v1/jobs/",
        files={"video": ("test.avi", video, "video/avi")},
        data={"selected_pipelines": "audio_processing"},
    )
    assert response.status_code in (200, 201), response.text
    warnings = response.json()["warnings"]
    assert any("deprecated" in w and "v1.7.0" in w for w in warnings)


def test_each_family_declares_at_most_one_default():
    registry = get_registry()
    registry.load(force=True)
    defaults: dict[str, list[str]] = {}
    for meta in registry.list(include_deprecated=True):
        if meta.family_default and meta.pipeline_family:
            defaults.setdefault(meta.pipeline_family, []).append(meta.name)
    assert all(len(names) == 1 for names in defaults.values()), defaults


@pytest.mark.parametrize(
    "family,expected", [("audio", "audio_processing"), ("face", "face_analysis")]
)
def test_family_short_name_resolves_to_declared_default(family, expected):
    registry = get_registry()
    registry.load(force=True)
    meta = registry.get(expected)
    assert meta is not None and meta.family_default
    assert meta.pipeline_family == family
