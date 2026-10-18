"""Speech recognition on the faster-whisper runtime (spec 030).

faster-whisper is mocked: these check how our settings reach it, how its
segments become the result the WebVTT is written from, and what provenance
records. The real model runs in tests/integration/test_output_baseline.py.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from videoannotator.pipelines.audio_processing import whisper_base_pipeline
from videoannotator.pipelines.audio_processing.speech_pipeline import (
    SpeechPipeline,
    _suppress_tokens,
)

REPO = "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
SNAPSHOT = "/models/huggingface/hub/models--x/snapshots/abc123"


def _segment(i, start, end, text, words=()):
    return SimpleNamespace(
        id=i,
        start=start,
        end=end,
        text=text,
        tokens=[1, 2],
        temperature=0.0,
        avg_logprob=-0.2,
        compression_ratio=1.1,
        no_speech_prob=0.01,
        words=[
            SimpleNamespace(word=w, start=s, end=e, probability=p)
            for w, s, e, p in words
        ]
        or None,
    )


@pytest.fixture
def faster_whisper(monkeypatch):
    """A stand-in faster_whisper package; returns the WhisperModel mock."""
    model = MagicMock()
    model.model.compute_type = "float32"
    model.transcribe.return_value = (iter([]), SimpleNamespace(language="en"))
    whisper_model = MagicMock(return_value=model)
    download = MagicMock(return_value=SNAPSHOT)

    package = types.ModuleType("faster_whisper")
    package.WhisperModel = whisper_model
    utils = types.ModuleType("faster_whisper.utils")
    utils.download_model = download
    utils._MODELS = {"base": "Systran/faster-whisper-base", "large-v3-turbo": REPO}
    monkeypatch.setitem(sys.modules, "faster_whisper", package)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", utils)
    return SimpleNamespace(WhisperModel=whisper_model, model=model, download=download)


@pytest.fixture
def cpu():
    with patch("torch.cuda.is_available", return_value=False):
        yield


def test_provenance_names_the_repository_snapshot_and_precision(faster_whisper, cpu):
    pipeline = SpeechPipeline({"whisper_model": "large-v3-turbo"})
    pipeline.initialize()

    faster_whisper.download.assert_called_once_with("large-v3-turbo")
    faster_whisper.WhisperModel.assert_called_once_with(
        SNAPSHOT, device="cpu", compute_type="default"
    )
    [ref] = pipeline.provenance_models()
    assert (ref.name, ref.source, ref.revision, ref.revision_kind) == (
        REPO,
        "huggingface",
        "abc123",
        "git",
    )
    assert ref.quantization == "float32"
    assert ref.to_dict()["quantization"] == "float32"


def test_use_fp16_false_means_float32(faster_whisper, cpu):
    SpeechPipeline({"use_fp16": False, "whisper_model": "base"}).initialize()
    assert faster_whisper.WhisperModel.call_args.kwargs["compute_type"] == "float32"


def test_compute_type_setting_is_passed_through(faster_whisper, cpu):
    SpeechPipeline({"compute_type": "int8", "whisper_model": "base"}).initialize()
    assert faster_whisper.WhisperModel.call_args.kwargs["compute_type"] == "int8"


def test_unknown_model_name_says_what_is_available(faster_whisper, cpu):
    with pytest.raises(ValueError, match="Unknown Whisper model 'huge'"):
        SpeechPipeline({"whisper_model": "huge"}).initialize()


def test_translate_with_turbo_is_refused(faster_whisper, cpu):
    with pytest.raises(ValueError, match="large-v3 or medium"):
        SpeechPipeline(
            {"task": "translate", "whisper_model": "large-v3-turbo"}
        ).initialize()
    faster_whisper.WhisperModel.assert_not_called()


def test_translate_with_a_model_that_translates_runs(faster_whisper, cpu):
    pipeline = SpeechPipeline({"task": "translate", "whisper_model": "large-v3"})
    with patch.object(whisper_base_pipeline, "model_repository", return_value="r/x"):
        pipeline.initialize()
    assert pipeline.whisper_model is faster_whisper.model


@patch("torch.cuda.is_available", return_value=True)
def test_missing_cudnn_at_warm_up_falls_back_to_cpu(_cuda, faster_whisper):
    gpu_model = MagicMock()
    gpu_model.transcribe.side_effect = RuntimeError(
        "Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9}"
    )
    faster_whisper.WhisperModel.side_effect = [gpu_model, faster_whisper.model]

    pipeline = SpeechPipeline({"whisper_model": "base"})
    pipeline.initialize()

    assert pipeline.fell_back_to_cpu
    assert pipeline.device.type == "cpu"
    assert faster_whisper.WhisperModel.call_args.kwargs["device"] == "cpu"
    assert len(pipeline.provenance_models()) == 1


def test_settings_reach_transcribe(faster_whisper, cpu):
    pipeline = SpeechPipeline(
        {
            "whisper_model": "base",
            "language": "en",
            "beam_size": 3,
            "temperature": 0.2,
            "word_timestamps": False,
            "suppress_tokens": "-1, 50363",
        }
    )
    pipeline.initialize()
    faster_whisper.model.transcribe.reset_mock()
    pipeline.transcribe_audio(np.zeros(16000, dtype=np.float32))

    options = faster_whisper.model.transcribe.call_args.kwargs
    assert options["language"] == "en"
    assert options["beam_size"] == 3
    assert options["temperature"] == 0.2
    assert options["word_timestamps"] is False
    assert options["suppress_tokens"] == [-1, 50363]
    assert options["vad_filter"] is False


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, None), ("", None), ("-1", [-1]), ("1,2", [1, 2]), ([3, "4"], [3, 4])],
)
def test_suppress_tokens_parsing(value, expected):
    assert _suppress_tokens(value) == expected


def test_segments_become_the_result_the_webvtt_is_written_from(faster_whisper, cpu):
    faster_whisper.model.transcribe.side_effect = None
    pipeline = SpeechPipeline({"whisper_model": "base"})
    pipeline.initialize()
    faster_whisper.model.transcribe.return_value = (
        iter(
            [
                _segment(
                    0, 0.0, 1.32, " Ready, baby girl?", [(" Ready,", 0.0, 0.4, 0.9)]
                ),
                _segment(1, 2.0, 3.16, " Good morning, girl."),
            ]
        ),
        SimpleNamespace(language="en"),
    )

    result = pipeline.transcribe_audio(np.zeros(16000, dtype=np.float32))

    assert result["transcript"] == "Ready, baby girl? Good morning, girl."
    assert result["language"] == "en"
    assert result["words"] == [
        {"word": "Ready,", "start": 0.0, "end": 0.4, "confidence": 0.9}
    ]
    segments = result["metadata"]["segments"]
    assert [(s["start"], s["end"], s["text"]) for s in segments] == [
        (0.0, 1.32, "Ready, baby girl?"),
        (2.0, 3.16, "Good morning, girl."),
    ]
    assert result["metadata"]["num_segments"] == 2
    assert result["metadata"]["end_time"] == 3.16


def test_a_decoding_error_fails_the_pipeline(faster_whisper, cpu):
    pipeline = SpeechPipeline({"whisper_model": "base"})
    pipeline.initialize()

    def broken():
        yield _segment(0, 0.0, 1.0, "hi")
        raise RuntimeError("CUDA error: out of memory")

    faster_whisper.model.transcribe.return_value = (broken(), SimpleNamespace())
    with pytest.raises(RuntimeError, match="Speech recognition failed"):
        pipeline.transcribe_audio(np.zeros(16000, dtype=np.float32))


def test_without_faster_whisper_the_model_is_missing_not_crashing(monkeypatch, cpu):
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    pipeline = SpeechPipeline()
    pipeline.initialize()
    assert pipeline.is_initialized
    assert pipeline.whisper_model is None
    with pytest.raises(RuntimeError, match="not initialized"):
        pipeline.transcribe_audio(np.zeros(16000, dtype=np.float32))
