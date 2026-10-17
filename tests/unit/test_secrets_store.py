"""The saved Hugging Face token and where the server gets its token (spec 031)."""

from __future__ import annotations

import stat

import pytest

from videoannotator import config_env, secrets_store


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_DB_PATH", str(tmp_path / "videoannotator.db"))
    for name in ("HUGGINGFACE_TOKEN", "HF_AUTH_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_save_read_remove(tmp_path):
    assert secrets_store.read_saved_huggingface_token() is None
    path = secrets_store.save_huggingface_token("  hf_abc  ")
    assert path == tmp_path / "huggingface_token"
    assert secrets_store.read_saved_huggingface_token() == "hf_abc"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert secrets_store.remove_saved_huggingface_token() is True
    assert secrets_store.read_saved_huggingface_token() is None
    assert secrets_store.remove_saved_huggingface_token() is False


def test_empty_token_is_refused():
    with pytest.raises(ValueError):
        secrets_store.save_huggingface_token("   ")


def test_empty_file_is_not_a_token(tmp_path):
    (tmp_path / "huggingface_token").write_text("\n")
    assert secrets_store.read_saved_huggingface_token() is None


def test_saved_token_is_used_when_the_environment_has_none():
    assert config_env.huggingface_token() is None
    assert config_env.huggingface_token_source() == (None, None)
    secrets_store.save_huggingface_token("hf_saved")
    assert config_env.huggingface_token() == "hf_saved"
    assert config_env.huggingface_token_source() == ("saved", None)


def test_environment_wins(monkeypatch):
    secrets_store.save_huggingface_token("hf_saved")
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert config_env.huggingface_token() == "hf_env"
    assert config_env.huggingface_token_source() == ("environment", "HF_TOKEN")
