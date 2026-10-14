"""A pipeline that fails must raise, not return an empty result.

The job runner marks a pipeline as completed whenever `process()` returns, so
a swallowed error shows up as "completed" with no annotations (found
2026-10-01: a Triton cache error left speech_recognition "completed" with an
empty transcript).
"""

import logging
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from videoannotator.pipelines.base_pipeline import FrameFailures


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(30):
        writer.write(np.full((48, 64, 3), i * 8, dtype=np.uint8))
    writer.release()
    return path


class TestFrameFailures:
    def test_every_frame_failing_raises_with_first_error(self):
        failures = FrameFailures(logging.getLogger("test"))
        failures.failed_on(0, ValueError("boom"))
        failures.failed_on(5, ValueError("again"))
        with pytest.raises(RuntimeError, match=r"All 2 .*frame 0: boom"):
            failures.check()

    def test_some_frames_failing_only_warns(self, caplog):
        failures = FrameFailures(logging.getLogger("test"))
        failures.succeeded()
        failures.failed_on(5, ValueError("boom"))
        with caplog.at_level(logging.WARNING):
            failures.check()
        assert "1 of 2 sampled frames failed" in caplog.text

    def test_no_frames_is_not_a_failure(self):
        FrameFailures(logging.getLogger("test")).check()


def test_speech_recognition_raises_when_transcription_fails(tmp_path):
    from videoannotator.pipelines.audio_processing.speech_pipeline import (
        SpeechPipeline,
    )

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    pipeline = SpeechPipeline({})
    pipeline.is_initialized = True
    pipeline.model_type = "standard"
    pipeline.whisper_model = MagicMock()
    pipeline.whisper_model.transcribe.side_effect = RuntimeError("Triton cache")

    with (
        patch.object(
            pipeline,
            "extract_audio_from_video",
            return_value=(np.zeros(16000, dtype=np.float32), 16000),
        ),
        pytest.raises(RuntimeError, match="Triton cache"),
    ):
        pipeline.process(video)


def test_scene_detection_raises_instead_of_inventing_one_scene(tiny_video):
    from videoannotator.pipelines.scene_detection import scene_pipeline

    if not scene_pipeline.SCENEDETECT_AVAILABLE:
        pytest.skip("PySceneDetect not installed")
    pipeline = scene_pipeline.SceneDetectionPipeline({})
    with (
        patch.object(scene_pipeline, "detect", side_effect=OSError("bad codec")),
        pytest.raises(RuntimeError, match="bad codec"),
    ):
        pipeline._detect_scene_boundaries(str(tiny_video), 0.0, None)


def test_face_analysis_raises_when_every_frame_fails(tiny_video):
    from videoannotator.pipelines.face_analysis.face_pipeline import (
        FaceAnalysisPipeline,
    )

    pipeline = FaceAnalysisPipeline({"detection_backend": "opencv"})
    with (
        patch.object(
            pipeline, "_detect_faces_in_frame", side_effect=RuntimeError("cuDNN")
        ),
        pytest.raises(RuntimeError, match=r"All \d+ sampled frames failed.*cuDNN"),
    ):
        pipeline.process(str(tiny_video), pps=5)


def _vlm_pipeline(**config):
    from videoannotator.pipelines.vlm_annotation.ollama_client import VLMCallResult
    from videoannotator.pipelines.vlm_annotation.vlm_pipeline import (
        VLMAnnotationPipeline,
    )

    pipeline = VLMAnnotationPipeline(
        {"base_url": "http://127.0.0.1:11434", "frame_interval_sec": 1.0, **config}
    )
    pipeline.is_initialized = True
    pipeline._client = MagicMock()
    pipeline._client.chat.return_value = VLMCallResult(
        raw_text="", thinking="", total_time=0, load_time=0, prompt_tokens=0,
        resp_tokens=0, tokens_per_sec=0, error="model not found",
    )  # fmt: skip
    return pipeline


def test_vlm_raises_when_no_sample_point_gets_an_answer(tiny_video):
    pipeline = _vlm_pipeline(abort_after_consecutive_failures=99)
    with pytest.raises(RuntimeError, match=r"No sample point.*model not found"):
        pipeline.process(str(tiny_video))


def test_vlm_raises_when_it_aborts(tiny_video, tmp_path):
    pipeline = _vlm_pipeline(abort_after_consecutive_failures=2)
    with pytest.raises(RuntimeError, match="aborted after 2 consecutive call"):
        pipeline.process(str(tiny_video), output_dir=str(tmp_path / "out"))
    assert pipeline._client.chat.call_count == 2
    # What it got is still written out, for diagnosis.
    assert list((tmp_path / "out").iterdir())
