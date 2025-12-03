from fastapi.testclient import TestClient

from videoannotator.api.main import app

client = TestClient(app)


def test_pipeline_not_found_error_envelope():
    resp = client.get("/api/v1/pipelines/does_not_exist")
    assert resp.status_code == 404
    data = resp.json()
    assert "error" in data
    assert data["error"].get("code") == "PIPELINE_NOT_FOUND"
    assert "hint" in data["error"]
