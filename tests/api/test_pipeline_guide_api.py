"""The pipeline guide on the API (spec 026, contracts/api.md)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.main import create_app
from videoannotator.api.v1 import pipelines as pipelines_api
from videoannotator.registry.pipeline_registry import PipelineRegistry

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "registry"

# The fields a PipelineInfo carried before spec 026; none may go.
EXISTING_KEYS = {
    "name",
    "display_name",
    "description",
    "enabled",
    "pipeline_family",
    "variant",
    "tasks",
    "modalities",
    "capabilities",
    "backends",
    "stability",
    "outputs",
    "config_schema",
    "examples",
    "available",
    "install_hint",
    "readiness",
}
GUIDE_KEYS = {"title", "summary", "homepage", "benchmarks", "quality_note", "citations"}


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def fixture_registry(monkeypatch):
    def use(name: str):
        registry = PipelineRegistry(metadata_dir=FIXTURES / name)
        registry.load(force=True)
        monkeypatch.setattr(pipelines_api, "get_registry", lambda: registry)

    return use


def test_builtin_carries_existing_and_guide_keys(client):
    detail = client.get("/api/v1/pipelines/face_analysis").json()
    assert detail.keys() >= EXISTING_KEYS
    assert detail.keys() >= GUIDE_KEYS
    field = detail["config_schema"]["detect_emotions"]
    assert {"type", "default", "description"} <= field.keys()
    assert {"group", "label", "unit", "min", "max", "applies_when"} <= field.keys()


def test_list_and_detail_agree_on_guide(client):
    listed = client.get("/api/v1/pipelines?include_unavailable=true").json()[
        "pipelines"
    ]
    for item in listed:
        detail = client.get(f"/api/v1/pipelines/{item['name']}").json()
        for key in GUIDE_KEYS:
            assert item[key] == detail[key], (item["name"], key)


def test_full_guide_fixture(client, fixture_registry):
    fixture_registry("full_guide")
    d = client.get("/api/v1/pipelines/full_guide").json()
    assert d["title"] == "Things: what they are"
    assert d["homepage"] == "https://example.org/method-x"
    assert d["benchmarks"] == [
        {
            "name": "Things benchmark (hard)",
            "measure": "AP",
            "figure": "91.4%",
            "source": "https://example.org/paper",
            "note": "large backbone",
        }
    ]
    slow = d["config_schema"]["slow_only"]
    assert slow["group"] == "technical"
    assert slow["applies_when"] == {"backend": ["slow"]}


def test_metadata_without_guide_gives_nulls(client, fixture_registry):
    fixture_registry("no_guide")
    d = client.get("/api/v1/pipelines/no_guide").json()
    assert d["title"] is None
    assert d["summary"] is None
    assert d["benchmarks"] is None
    assert d["citations"] == []
    assert d["config_schema"]["threshold"]["group"] == "technical"


def test_schema_route_carries_groups_and_labels(client, fixture_registry):
    fixture_registry("full_guide")
    d = client.get("/api/v1/pipelines/full_guide/schema").json()
    assert d["pipeline"]["title"] == "Things: what they are"
    params = {p["name"]: p for p in d["parameters"]}
    most = params["most_things"]
    assert (most["group"], most["label"], most["unit"], most["min"], most["max"]) == (
        "behavioural",
        "Most things per frame",
        "things",
        1,
        100,
    )
    assert params["slow_only"]["applies_when"] == {"backend": ["slow"]}
    assert params["odd_group"]["label"] == "Odd Group"


def test_viewer_schema_fixture_is_current(client):
    """The viewer's contract fixture is today's schema route output."""
    import json

    fixture = (
        Path(__file__).resolve().parents[2]
        / "viewer/src/test/contract/pipeline-schema-face_analysis.json"
    )
    live = client.get("/api/v1/pipelines/face_analysis/schema").json()
    assert json.loads(fixture.read_text()) == live, (
        "Regenerate viewer/src/test/contract/pipeline-schema-face_analysis.json "
        "from GET /api/v1/pipelines/face_analysis/schema"
    )
