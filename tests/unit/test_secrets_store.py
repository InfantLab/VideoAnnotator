"""The saved Hugging Face token and where the server gets its token (spec 031)."""

from __future__ import annotations

import stat

import pytest

from videoannotator import config_env, secrets_store

SAVED = "hf_savedSAVEDsaved1234567"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_DB_PATH", str(tmp_path / "videoannotator.db"))
    for name in ("HUGGINGFACE_TOKEN", "HF_AUTH_TOKEN", "HF_TOKEN"):
        monkeypatch.delenv(name, raising=False)


def test_save_read_remove(tmp_path):
    assert secrets_store.read_saved_huggingface_token() is None
    path = secrets_store.save_huggingface_token(f"  {SAVED}  ")
    assert path == tmp_path / "huggingface_token"
    assert secrets_store.read_saved_huggingface_token() == SAVED
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
    secrets_store.save_huggingface_token(SAVED)
    assert config_env.huggingface_token() == SAVED
    assert config_env.huggingface_token_source() == ("saved", None)


def test_environment_wins(monkeypatch):
    secrets_store.save_huggingface_token(SAVED)
    monkeypatch.setenv("HF_TOKEN", "hf_env")
    assert config_env.huggingface_token() == "hf_env"
    assert config_env.huggingface_token_source() == ("environment", "HF_TOKEN")


def test_a_bracketed_paste_is_cleaned():
    """What a hidden prompt received from a terminal paste on 2026-10-18."""
    pasted = f"\x1b[200~{SAVED}\x1b[201~ \n"
    assert secrets_store.clean_huggingface_token(pasted) == SAVED
    secrets_store.save_huggingface_token(pasted)
    assert secrets_store.read_saved_huggingface_token() == SAVED


@pytest.mark.parametrize("bad", ["hf_short", "not-a-token-at-all-really", "hf_abc!def"])
def test_something_else_is_refused_without_repeating_it(bad):
    problem = secrets_store.token_shape_problem(bad)
    assert problem and bad not in problem
    with pytest.raises(ValueError):
        secrets_store.save_huggingface_token(bad)


def test_mask_shows_only_the_ends():
    assert secrets_store.mask_huggingface_token(SAVED) == "hf_sav…4567"
