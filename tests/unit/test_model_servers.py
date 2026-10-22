"""The model server list and where keys come from (spec 033 R4)."""

import logging
import stat

import pytest

from videoannotator import model_servers, secrets_store
from videoannotator.database import database as db_module
from videoannotator.database.models import ModelServer
from videoannotator.model_servers import ModelServerError

LAB = "http://10.0.4.12:8000/v1"
KEY = "sk-lab-gpu-0123456789"


@pytest.fixture
def db(monkeypatch):
    db_module.Base.metadata.create_all(bind=db_module.engine)
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    path = secrets_store.model_server_keys_path()
    path.unlink(missing_ok=True)
    with db_module.SessionLocal() as session:
        session.query(ModelServer).delete()
        session.commit()
        yield session
        session.query(ModelServer).delete()
        session.commit()
    path.unlink(missing_ok=True)
    secrets_store.KEY_ENV_NAMES.clear()


def test_the_default_comes_first_and_cant_be_removed(db):
    model_servers.add(db, "lab-gpu", "openai_compatible", LAB)
    servers = model_servers.list_servers(db)
    assert [s["id"] for s in servers][0] == "default"
    assert servers[0]["name"] == "This computer (Ollama)"
    assert servers[0]["removable"] is False
    assert servers[1]["name"] == "lab-gpu"
    with pytest.raises(ModelServerError) as e:
        model_servers.remove(db, "default")
    assert e.value.code == "DEFAULT_SERVER"


def test_add_normalises_and_refuses_duplicates(db):
    server = model_servers.add(
        db, "lab-gpu", "openai_compatible", "HTTP://10.0.4.12:8000/v1/"
    )
    assert server["base_url"] == LAB
    with pytest.raises(ModelServerError) as e:
        model_servers.add(db, "again", "openai_compatible", LAB)
    assert (e.value.status, e.value.code) == (409, "SERVER_EXISTS")
    with pytest.raises(ModelServerError) as e:
        model_servers.add(db, "ollama again", "ollama", "http://127.0.0.1:11434")
    assert e.value.code == "SERVER_EXISTS"


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"kind": "gemini"}, "INVALID_KIND"),
        ({"base_url": "10.0.4.12:8000"}, "INVALID_BASE_URL"),
        ({"name": " "}, "INVALID_NAME"),
        ({"api_key_env": "lab-key"}, "INVALID_ENV_NAME"),
    ],
)
def test_bad_input_is_refused(db, kwargs, code):
    args = {"name": "lab", "kind": "openai_compatible", "base_url": LAB, **kwargs}
    with pytest.raises(ModelServerError) as e:
        model_servers.add(db, **args)
    assert e.value.code == code


def test_key_order(db, monkeypatch):
    server = model_servers.add(
        db, "lab", "openai_compatible", LAB, api_key=KEY, api_key_env="LAB_GPU_KEY"
    )
    monkeypatch.delenv("LAB_GPU_KEY", raising=False)
    assert model_servers.key_for(db, LAB) == KEY
    assert model_servers.key_source(server) == "stored"
    monkeypatch.setenv("LAB_GPU_KEY", "")
    assert model_servers.key_for(db, LAB) == KEY
    monkeypatch.setenv("LAB_GPU_KEY", "sk-from-env-987654")
    assert model_servers.key_for(db, LAB) == "sk-from-env-987654"
    assert model_servers.key_source(server) == "env"
    model_servers.update(db, server["id"], clear_key=True)
    monkeypatch.delenv("LAB_GPU_KEY")
    assert model_servers.key_for(db, LAB) is None
    assert model_servers.key_source(server) is None


def test_unknown_address_has_no_key(db):
    assert model_servers.key_for(db, "http://127.0.0.1:9999/v1") is None


def test_key_kept_in_a_private_file_not_the_database(db):
    model_servers.add(db, "lab", "openai_compatible", LAB, api_key=KEY)
    path = secrets_store.model_server_keys_path()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert KEY in path.read_text()
    row = db.query(ModelServer).one()
    assert KEY not in repr(vars(row))


def test_removing_a_server_removes_its_key(db):
    server = model_servers.add(db, "lab", "openai_compatible", LAB, api_key=KEY)
    model_servers.remove(db, server["id"])
    assert secrets_store.read_model_server_key(LAB) is None
    assert model_servers.list_servers(db)[1:] == []


def test_the_filter_redacts_a_key_logged_by_accident(db, caplog, monkeypatch):
    model_servers.add(db, "lab", "openai_compatible", LAB, api_key=KEY)
    model_servers.add(
        db,
        "hosted",
        "openai_compatible",
        "http://10.0.4.13/v1",
        api_key_env="HOSTED_KEY",
    )
    monkeypatch.setenv("HOSTED_KEY", "sk-hosted-abcdefgh")
    logger = logging.getLogger("videoannotator.test033")
    caplog.handler.addFilter(secrets_store.KeyRedactingFilter())
    with caplog.at_level(logging.INFO):
        logger.info("calling with %s", KEY)
        logger.info("env key sk-hosted-abcdefgh here")
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert KEY not in text
    assert "sk-hosted-abcdefgh" not in text
    assert text.count("<redacted>") == 2


def test_capabilities_are_recorded_on_the_row(db):
    model_servers.add(db, "lab", "openai_compatible", LAB)
    model_servers.record_capabilities(db, LAB, structured_output=False)
    model_servers.record_capabilities(db, LAB, video=("qwen", True))
    server = model_servers.server_for_url(db, LAB)
    assert server["structured_output"] is False
    assert server["video_models"] == {"qwen": True}
