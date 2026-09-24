"""API tests for pipeline readiness and the extras-group listing
(specs/011-pipeline-readiness contract §1-2)."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from videoannotator.api import readiness
from videoannotator.api.main import app

client = TestClient(app)

KNOWN_STATES = {
    "ready",
    "not_installed",
    "installing",
    "restart_required",
    "needs_setup",
}


@pytest.fixture(autouse=True)
def _offline_ollama():
    status = {
        "ollama_reachable": False,
        "base_url": "http://127.0.0.1:11434",
        "models": [],
    }
    with patch.object(readiness, "_ollama_status", return_value=status):
        yield


def test_every_pipeline_carries_a_readiness_object():
    body = client.get("/api/v1/pipelines?include_unavailable=true").json()
    assert body["pipelines"]
    for p in body["pipelines"]:
        r = p["readiness"]
        assert r["state"] in KNOWN_STATES
        assert bool(r["blockers"]) == (r["state"] == "needs_setup")
        if r["state"] == "ready":
            assert p["available"] is True


def test_default_listing_is_unchanged_apart_from_the_new_field():
    everything = client.get("/api/v1/pipelines?include_unavailable=true").json()
    default = client.get("/api/v1/pipelines").json()
    expected = [p["name"] for p in everything["pipelines"] if p["available"]]
    assert [p["name"] for p in default["pipelines"]] == expected
    assert set(default) == {"pipelines", "total", "restart_required"}


def test_unreachable_ollama_blocks_vlm_annotation():
    r = client.get("/api/v1/pipelines/vlm_annotation").json()["readiness"]
    if r["state"] == "not_installed":
        pytest.skip("llm extra not installed in this environment")
    assert r["state"] == "needs_setup"
    assert r["blockers"][0]["name"] == "ollama"


def test_extras_listing_is_its_own_route_not_a_pipeline_name():
    resp = client.get("/api/v1/pipelines/extras")
    assert resp.status_code == 200
    groups = resp.json()["extras"]
    names = [g["name"] for g in groups]
    assert "audio" in names
    assert "all" not in names and "dev" not in names
    audio = groups[names.index("audio")]
    assert {"speaker_diarization", "speech_recognition"} <= set(audio["pipelines"])
    assert isinstance(audio["approx_download_mb"], int)
