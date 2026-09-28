"""SpeechPipeline's patchable `whisper` symbol must reach the real package:
it used to be a stub that always raised, so speech recognition never loaded
a model outside tests."""

import sys
from types import SimpleNamespace

import pytest

pytest.importorskip("librosa", reason="requires the `audio` extra")

from videoannotator.pipelines.audio_processing import speech_pipeline


def test_load_model_forwards_to_openai_whisper(monkeypatch):
    fake = SimpleNamespace(load_model=lambda name, **kw: ("model", name, kw))
    monkeypatch.setitem(sys.modules, "whisper", fake)
    assert speech_pipeline.whisper.load_model("base", device="cpu") == (
        "model",
        "base",
        {"device": "cpu"},
    )


def test_missing_package_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "whisper", None)
    with pytest.raises(ImportError, match="pip install openai-whisper"):
        speech_pipeline.whisper.load_model("base")
