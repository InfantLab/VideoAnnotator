from fastapi.testclient import TestClient

from videoannotator.api.main import create_app

app = create_app()


def test_pipelines_list_includes_new_taxonomy_fields():
    client = TestClient(app)
    resp = client.get("/api/v1/pipelines/")
    assert resp.status_code == 200
    data = resp.json()
    assert "pipelines" in data
    assert data["total"] == len(data["pipelines"]) > 0
    sample = data["pipelines"][0]
    # New fields should exist (may be None or list)
    assert "display_name" in sample
    assert "tasks" in sample
    assert "modalities" in sample
    assert "capabilities" in sample
    assert "backends" in sample
    assert "outputs" in sample


def test_single_pipeline_detail_matches_list():
    client = TestClient(app)
    resp = client.get("/api/v1/pipelines/")
    name = resp.json()["pipelines"][0]["name"]
    detail = client.get(f"/api/v1/pipelines/{name}")
    assert detail.status_code == 200
    d = detail.json()
    assert d["name"] == name
    # Ensure display_name and tasks consistent type
    assert isinstance(d.get("display_name"), (str, type(None)))
    assert isinstance(d.get("tasks", []), list)
