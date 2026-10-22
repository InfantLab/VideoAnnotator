"""/api/v1/model-servers (spec 033 contracts/api.md) against a fake server."""

import json

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.fake_openai_server import FakeOpenAIServer
from videoannotator import secrets_store
from videoannotator.api.main import app
from videoannotator.database import database as db_module
from videoannotator.database.models import ModelServer, ServerSetting
from videoannotator.pipelines.vlm_annotation import openai_client

client = TestClient(app)

LAB = "http://127.0.0.1:8000/v1"
KEY = "sk-never-shown-0123456789"


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    db_module.Base.metadata.create_all(bind=db_module.engine)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    monkeypatch.delenv("VIDEOANNOTATOR_REMOTE_MODEL_SERVERS", raising=False)
    secrets_store.model_server_keys_path().unlink(missing_ok=True)
    with db_module.SessionLocal() as db:
        db.query(ModelServer).delete()
        db.query(ServerSetting).delete()
        db.commit()
    yield
    secrets_store.model_server_keys_path().unlink(missing_ok=True)
    secrets_store.KEY_ENV_NAMES.clear()


@pytest.fixture
def fake(monkeypatch):
    server = FakeOpenAIServer("llama_cpp")
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
    return server


def _add(**kwargs):
    body = {"name": "workstation", "kind": "openai_compatible", "base_url": LAB}
    body.update(kwargs)
    return client.post("/api/v1/model-servers", json=body)


def test_lists_the_default_first():
    body = client.get("/api/v1/model-servers").json()
    default = body["servers"][0]
    assert default["id"] == "default"
    assert default["kind"] == "ollama"
    assert default["endpoint"]["class"] == "local"
    assert default["removable"] is False
    assert body["remote"]["allowed"] is False
    assert body["remote"]["fixed_by_environment"] is False
    assert body["remote"]["warning"]
    assert body["keys_file"].endswith("model_server_keys.json")


def test_add_patch_delete():
    r = _add(api_key=KEY)
    assert r.status_code == 201, r.text
    server = r.json()
    assert server["has_key"] is True
    assert server["key_source"] == "stored"
    assert server["endpoint"]["sendable"] is True

    assert _add().status_code == 409
    assert _add().json()["error"]["code"] == "SERVER_EXISTS"

    r = client.patch(
        f"/api/v1/model-servers/{server['id']}",
        json={"name": "lab", "clear_key": True, "api_key_env": "LAB_KEY"},
    )
    assert r.status_code == 200, r.text
    assert (r.json()["name"], r.json()["has_key"], r.json()["api_key_env"]) == (
        "lab",
        False,
        "LAB_KEY",
    )
    r = client.patch(f"/api/v1/model-servers/{server['id']}", json={"name": "lab2"})
    assert r.json()["api_key_env"] == "LAB_KEY"

    assert client.delete(f"/api/v1/model-servers/{server['id']}").status_code == 204
    names = [s["name"] for s in client.get("/api/v1/model-servers").json()["servers"]]
    assert names == ["This computer (Ollama)"]
    assert client.delete("/api/v1/model-servers/default").status_code == 422


def test_bad_env_name():
    r = _add(api_key_env="not a name")
    assert (r.status_code, r.json()["error"]["code"]) == (422, "INVALID_ENV_NAME")


class TestCheck:
    def _check(self, **kwargs):
        body = {"kind": "openai_compatible", "base_url": LAB}
        body.update(kwargs)
        r = client.post("/api/v1/model-servers/check", json=body)
        assert r.status_code == 200, r.text
        return r.json()

    def test_models(self, fake):
        body = self._check()
        assert body["outcome"] == "models"
        assert body["message"] == "Reachable. 1 model."
        assert body["models"] == [fake.model]
        assert body["endpoint"]["class"] == "local"
        assert body["video"] is None

    def test_unreachable(self, fake):
        fake.down = True
        body = self._check()
        assert body["outcome"] == "unreachable"
        assert body["hint"]

    def test_without_v1_suggests_it(self, fake):
        body = self._check(base_url="http://127.0.0.1:8000")
        assert body["outcome"] == "wrong_kind"
        assert body["suggest_base_url"] == "http://127.0.0.1:8000/v1"

    def test_ollama_kind_at_an_openai_server(self, fake):
        body = self._check(kind="ollama", base_url="http://127.0.0.1:8000")
        assert body["outcome"] == "wrong_kind"
        assert body["suggest_kind"] == "openai_compatible"

    def test_ollama_v1_suggests_the_ollama_kind(self, monkeypatch):
        server = FakeOpenAIServer("ollama_v1", ollama_native=True)
        monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
        body = self._check(base_url="http://127.0.0.1:11434/v1")
        assert body["outcome"] == "models"
        assert body["suggest_kind"] == "ollama"

    def test_no_models(self, fake):
        fake.recorded["models"]["data"] = []
        assert self._check()["outcome"] == "no_models"

    def test_key_refused(self, fake):
        fake.refuse_key = True
        assert self._check(api_key=KEY)["outcome"] == "key_refused"

    def test_a_listed_servers_key_is_used(self, fake):
        fake.key = KEY
        server = _add(api_key=KEY).json()
        assert self._check(id=server["id"])["outcome"] == "models"
        assert self._check()["outcome"] == "models"
        models_requests = [r for r in fake.requests if r.path == "/v1/models"]
        assert models_requests[-1].authorized

    def test_video_probe_is_recorded(self, monkeypatch):
        server = FakeOpenAIServer("vllm")
        monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
        _add()
        assert self._check(model=server.model)["video"] is True
        server.reject_video = True
        assert self._check(model="text-only")["video"] is False
        listed = client.get("/api/v1/model-servers").json()["servers"][1]
        assert listed["video_models"] == {server.model: True, "text-only": False}


def test_a_key_never_appears_in_any_response(fake, monkeypatch):
    fake.key = KEY
    monkeypatch.setenv("LAB_ENV_KEY", "sk-env-key-abcdefghij")
    responses = [_add(api_key=KEY)]
    server = responses[0].json()
    responses += [
        client.get("/api/v1/model-servers"),
        client.post(
            "/api/v1/model-servers/check",
            json={"kind": "openai_compatible", "base_url": LAB, "api_key": KEY},
        ),
        client.patch(
            f"/api/v1/model-servers/{server['id']}",
            json={"api_key": KEY, "api_key_env": "LAB_ENV_KEY"},
        ),
        client.get("/api/v1/model-servers"),
        client.get("/api/v1/model-servers/settings"),
    ]
    for r in responses:
        assert r.status_code < 400, r.text
        assert KEY not in r.text
        assert "sk-env-key-abcdefghij" not in r.text
    listed = json.loads(responses[-2].text)["servers"][1]
    assert listed["key_source"] == "env"
