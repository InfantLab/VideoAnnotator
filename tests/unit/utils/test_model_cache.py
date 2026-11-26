"""Models kept loaded between videos (UAT #72)."""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from videoannotator.utils import model_cache


@pytest.fixture
def kept(monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_MODEL_IDLE_SECONDS", "300")
    model_cache.clear()
    yield
    model_cache.clear()


def test_a_second_video_reuses_the_model(kept):
    load = MagicMock(side_effect=lambda: object())
    first = model_cache.get(("m", 1), load)
    second = model_cache.get(("m", 1), load)
    assert second is first
    load.assert_called_once()


def test_different_settings_load_their_own(kept):
    a = model_cache.get(("m", "small"), object)
    b = model_cache.get(("m", "large"), object)
    assert a.value is not b.value


def test_a_failed_load_is_tried_again(kept):
    load = MagicMock(side_effect=[RuntimeError("no network"), "model"])
    with pytest.raises(RuntimeError):
        model_cache.get("m", load)
    assert model_cache.get("m", load).value == "model"


def test_two_videos_at_once_load_once(kept):
    started = threading.Event()
    calls = []

    def slow_load():
        calls.append(1)
        started.set()
        time.sleep(0.2)
        return object()

    results = []
    threads = [
        threading.Thread(target=lambda: results.append(model_cache.get("m", slow_load)))
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert results[0] is results[1]


def test_idle_models_are_released_but_not_while_running(kept):
    entry = model_cache.get("m", object)
    later = time.monotonic() + 301
    with model_cache.turn(entry):
        held = threading.Thread(target=lambda: model_cache.release_idle(later))
        held.start()
        held.join()
        assert model_cache.loaded() == ["m"]
    assert model_cache.release_idle(later) == 1
    assert model_cache.loaded() == []


def test_zero_loads_per_video(monkeypatch):
    monkeypatch.setenv("VIDEOANNOTATOR_MODEL_IDLE_SECONDS", "0")
    load = MagicMock(side_effect=lambda: object())
    model_cache.get("m", load)
    model_cache.get("m", load)
    assert load.call_count == 2
    assert model_cache.loaded() == []


def test_diarization_loads_once_for_two_videos(kept, monkeypatch):
    from videoannotator.pipelines.audio_processing import diarization_pipeline as dp

    monkeypatch.setattr(dp, "PYANNOTE_AVAILABLE", True)
    monkeypatch.setattr(dp, "huggingface_token", lambda: "token")
    fake = MagicMock()
    with patch.object(dp, "PyAnnotePipeline", fake, create=True):
        for _ in range(2):
            pipeline = dp.DiarizationPipeline()
            pipeline.initialize()
            assert [m.name for m in pipeline.provenance_models()]
            pipeline.cleanup()
    fake.from_pretrained.assert_called_once()


def test_speech_keeps_provenance_when_reusing(kept):
    from videoannotator.pipelines.audio_processing.speech_pipeline import (
        SpeechPipeline,
    )
    from videoannotator.provenance import ModelRef

    loads = []

    def fake_load(self):
        loads.append(1)
        self.whisper_model = MagicMock()
        self._model_refs.append(
            ModelRef("Systran/faster-whisper-base", "huggingface", "abc", "git")
        )

    with patch.object(SpeechPipeline, "_load_whisper_model", fake_load):
        records = []
        for _ in range(2):
            pipeline = SpeechPipeline({"whisper_model": "base", "device": "cpu"})
            pipeline.initialize()
            records.append([m.name for m in pipeline.provenance_models()])
            assert pipeline.whisper_model is not None
            pipeline.cleanup()
    assert loads == [1]
    assert records[0] == records[1] == ["Systran/faster-whisper-base"]
