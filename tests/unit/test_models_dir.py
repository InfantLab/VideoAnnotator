"""One models directory (spec 016)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from videoannotator import models_dir as md

_LIBRARY_VARIABLES = ("HF_HUB_CACHE", "TORCH_HOME", "PYANNOTE_CACHE", "DEEPFACE_HOME")


def _env_after_import(env: dict[str, str]) -> dict[str, str | None]:
    """What the package sets at import, in a clean interpreter."""
    names = [md.MODELS_DIR_ENV, "HF_HOME", *_LIBRARY_VARIABLES]
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, os, videoannotator; "
            f"print(json.dumps({{n: os.environ.get(n) for n in {names!r}}}))",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return json.loads(out.splitlines()[-1])


def _clean_env(**extra: str) -> dict[str, str]:
    drop = {md.MODELS_DIR_ENV, "HF_HOME", *_LIBRARY_VARIABLES}
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env.update(extra)
    return env


def test_every_library_cache_points_into_the_models_dir(tmp_path):
    seen = _env_after_import(_clean_env(VIDEOANNOTATOR_MODELS_DIR=str(tmp_path)))
    assert seen[md.MODELS_DIR_ENV] == str(tmp_path.resolve())
    assert seen["HF_HUB_CACHE"] == str(tmp_path.resolve() / "huggingface" / "hub")
    assert seen["TORCH_HOME"] == str(tmp_path.resolve() / "torch")
    assert seen["PYANNOTE_CACHE"] == str(tmp_path.resolve() / "pyannote")
    assert seen["DEEPFACE_HOME"] == str(tmp_path.resolve() / "deepface")


def test_hf_home_is_never_set(tmp_path):
    """HF_HOME holds the `huggingface-cli login` token; moving it logs users out."""
    seen = _env_after_import(_clean_env(VIDEOANNOTATOR_MODELS_DIR=str(tmp_path)))
    assert seen["HF_HOME"] is None


def test_user_settings_win(tmp_path):
    seen = _env_after_import(
        _clean_env(
            VIDEOANNOTATOR_MODELS_DIR=str(tmp_path),
            TORCH_HOME="/elsewhere/torch",
            HF_HOME="/elsewhere/hf",
        )
    )
    assert seen["TORCH_HOME"] == "/elsewhere/torch"
    # A user-set HF_HOME already places Hugging Face's cache.
    assert seen["HF_HUB_CACHE"] is None
    assert seen["HF_HOME"] == "/elsewhere/hf"


def test_default_is_a_per_user_data_directory(monkeypatch):
    monkeypatch.delenv(md.MODELS_DIR_ENV, raising=False)
    root = md.models_dir()
    assert root.is_absolute()
    assert root.parts[-2:] == ("videoannotator", "models")


@pytest.mark.parametrize(
    "configured,expected_tail",
    [
        ("models/yolo/yolo11n-pose.pt", ("yolo", "yolo11n-pose.pt")),
        ("yolo11n-pose.pt", ("yolo", "yolo11n-pose.pt")),
    ],
)
def test_yolo_defaults_resolve_into_the_models_dir(
    tmp_path, monkeypatch, configured, expected_tail
):
    monkeypatch.setenv(md.MODELS_DIR_ENV, str(tmp_path / "models-root"))
    monkeypatch.chdir(tmp_path)  # no ./models here
    resolved = Path(md.resolve_yolo_model(configured))
    assert resolved == (tmp_path / "models-root").resolve().joinpath(*expected_tail)
    assert resolved.parent.is_dir()


def test_explicit_existing_yolo_path_is_kept(tmp_path, monkeypatch):
    monkeypatch.setenv(md.MODELS_DIR_ENV, str(tmp_path / "models-root"))
    custom = tmp_path / "my-model.pt"
    custom.write_bytes(b"")
    assert md.resolve_yolo_model(str(custom)) == str(custom)


def test_legacy_locations_found_only_outside_the_models_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / ".cache"))
    monkeypatch.setenv(md.MODELS_DIR_ENV, str(tmp_path / "new-models"))
    monkeypatch.chdir(tmp_path)
    whisper_old = tmp_path / ".cache" / "whisper"
    whisper_old.mkdir(parents=True)
    (whisper_old / "base.pt").write_bytes(b"x")
    assert whisper_old in md.legacy_locations()

    # The models dir itself is never reported as a legacy location.
    monkeypatch.setenv(md.MODELS_DIR_ENV, str(tmp_path / "models"))
    (tmp_path / "models" / "whisper").mkdir(parents=True)
    (tmp_path / "models" / "whisper" / "base.pt").write_bytes(b"x")
    assert (tmp_path / "models").resolve() not in md.legacy_locations()


def test_diagnose_reports_directory_and_sizes(tmp_path, monkeypatch):
    from videoannotator.diagnostics.models import diagnose_models

    monkeypatch.setenv(md.MODELS_DIR_ENV, str(tmp_path))
    (tmp_path / "yolo").mkdir()
    (tmp_path / "yolo" / "model.pt").write_bytes(b"x" * 1000)
    result = diagnose_models()
    assert result["models_dir"] == str(tmp_path.resolve())
    assert result["sources"]["yolo"] == 1000
    assert result["total_bytes"] == 1000
