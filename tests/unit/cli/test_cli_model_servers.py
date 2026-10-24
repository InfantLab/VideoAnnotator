"""`videoannotator model-servers ...` and run configs from a shell (spec 033 Story 5)."""

import json
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from tests.fixtures.fake_openai_server import FakeOpenAIServer
from videoannotator import secrets_store
from videoannotator.cli import app
from videoannotator.database import database as db_module
from videoannotator.database.models import ModelServer
from videoannotator.pipelines.vlm_annotation import openai_client

runner = CliRunner()
LAB = "http://127.0.0.1:8080/v1"


@pytest.fixture(autouse=True)
def fake(monkeypatch):
    db_module.Base.metadata.create_all(bind=db_module.engine)
    with db_module.SessionLocal() as db:
        db.query(ModelServer).delete()
        db.commit()
    secrets_store.model_server_keys_path().unlink(missing_ok=True)
    server = FakeOpenAIServer("llama_cpp")
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
    yield server
    secrets_store.model_server_keys_path().unlink(missing_ok=True)
    secrets_store.KEY_ENV_NAMES.clear()


def _check(*args):
    return runner.invoke(app, ["model-servers", "check", LAB, "--json", *args])


def test_check_prints_json_and_exits_0(fake):
    result = _check()
    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["outcome"] == "models"
    assert body["models"] == [fake.model]
    assert body["endpoint"]["class"] == "local"


@pytest.mark.parametrize(
    ("switch", "url", "code"),
    [
        ("down", LAB, 2),
        (None, "http://127.0.0.1:8080", 3),
        ("empty", LAB, 4),
        ("refuse_key", LAB, 5),
    ],
)
def test_check_exit_codes(fake, switch, url, code):
    if switch == "empty":
        fake.recorded["models"]["data"] = []
    elif switch:
        setattr(fake, switch, True)
    result = runner.invoke(app, ["model-servers", "check", url, "--json"])
    assert result.exit_code == code, result.output
    assert json.loads(result.output)["outcome"]


def test_add_with_a_key_variable_stores_only_its_name(fake, monkeypatch):
    monkeypatch.setenv("WORKSTATION_KEY", "sk-from-the-shell-12345678")
    result = runner.invoke(
        app,
        ["model-servers", "add", "workstation", LAB, "--key-env", "WORKSTATION_KEY"],
    )
    assert result.exit_code == 0, result.output
    with db_module.SessionLocal() as db:
        row = db.query(ModelServer).one()
        assert (row.name, row.api_key_env) == ("workstation", "WORKSTATION_KEY")
    assert not secrets_store.model_server_keys_path().exists()
    listed = json.loads(runner.invoke(app, ["model-servers", "list", "--json"]).output)
    assert listed["servers"][1]["key_source"] == "env"
    assert "sk-from-the-shell" not in json.dumps(listed)


def test_add_with_no_key_never_prompts(fake):
    with patch("typer.prompt") as prompt:
        result = runner.invoke(app, ["model-servers", "add", "ws", LAB, "--no-key"])
    assert result.exit_code == 0, result.output
    prompt.assert_not_called()


def test_there_is_no_key_flag():
    result = runner.invoke(
        app, ["model-servers", "add", "ws", LAB, "--api-key", "sk-x"]
    )
    assert result.exit_code != 0


def test_list_and_remove(fake):
    runner.invoke(app, ["model-servers", "add", "ws", LAB, "--no-key"])
    text = runner.invoke(app, ["model-servers", "list"]).output
    assert "This computer (Ollama)" in text and "ws" in text
    assert runner.invoke(app, ["model-servers", "remove", "ws"]).exit_code == 0
    assert runner.invoke(app, ["model-servers", "remove", "ws"]).exit_code == 1


def test_a_remote_server_is_refused_while_switched_off():
    result = runner.invoke(
        app, ["model-servers", "add", "hosted", "https://203.0.113.9/v1", "--no-key"]
    )
    assert result.exit_code == 1
    assert "switched off" in result.output


def _submit(tmp_path, config_text, suffix):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"x")
    config = tmp_path / f"shaped{suffix}"
    config.write_text(config_text)
    response = MagicMock(status_code=201)
    response.json.return_value = {"id": "j1", "status": "pending"}
    with patch("requests.post", return_value=response) as post:
        result = runner.invoke(
            app,
            [
                "job",
                "submit",
                str(video),
                "--pipelines",
                "vlm_annotation",
                "--config",
                str(config),
            ],
        )
    return result, post


def test_job_submit_takes_a_yaml_config_with_backend_and_shape(tmp_path):
    result, post = _submit(
        tmp_path,
        """vlm_annotation:
  backend: openai_compatible
  base_url: http://127.0.0.1:8080/v1
  model: gemma-3-4b
  answer_shape:
    - {name: touch, kind: yes_no}
""",
        ".yaml",
    )
    assert result.exit_code == 0, result.output
    sent = json.loads(post.call_args.kwargs["data"]["config"])
    assert sent["vlm_annotation"]["backend"] == "openai_compatible"
    assert sent["vlm_annotation"]["answer_shape"] == [
        {"name": "touch", "kind": "yes_no"}
    ]


def test_job_submit_refuses_a_key_in_the_config_before_sending(tmp_path):
    result, post = _submit(
        tmp_path,
        json.dumps({"vlm_annotation": {"api_key": "sk-in-a-file-123"}}),
        ".json",
    )
    assert result.exit_code == 1
    assert "Settings → Model servers" in result.output
    post.assert_not_called()
