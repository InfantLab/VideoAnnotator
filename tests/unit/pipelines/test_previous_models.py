"""Spec 030 User Story 2: a model the defaults moved away from is still a setting.

Only diarization and pose: the previous speech runtime was removed (Q3).
"""

from unittest.mock import Mock, patch

import pytest

from videoannotator.pipelines.audio_processing.diarization_pipeline import (
    DiarizationPipeline,
)
from videoannotator.pipelines.person_tracking.person_pipeline import (
    PersonTrackingPipeline,
)

DIARIZATION = "videoannotator.pipelines.audio_processing.diarization_pipeline"
PERSON = "videoannotator.pipelines.person_tracking.person_pipeline"


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({}, "pyannote/speaker-diarization-community-1"),
        (
            {"model": "pyannote/speaker-diarization-3.1"},
            "pyannote/speaker-diarization-3.1",
        ),
    ],
)
def test_diarization_loads_and_records_the_configured_model(config, expected):
    with (
        patch(f"{DIARIZATION}.PYANNOTE_AVAILABLE", True),
        patch(f"{DIARIZATION}.PyAnnotePipeline", create=True) as pyannote,
        patch(f"{DIARIZATION}.huggingface_token", return_value="hf_test"),
    ):
        pipeline = DiarizationPipeline(config)
        pipeline.initialize()

    pyannote.from_pretrained.assert_called_once_with(expected, token="hf_test")
    assert [ref.name for ref in pipeline.provenance_models()] == [expected]


def test_previous_pose_model_is_loaded_and_recorded(tmp_path):
    weights = tmp_path / "yolo11n-pose.pt"
    weights.write_bytes(b"weights")
    pipeline = PersonTrackingPipeline({"model": "yolo11n-pose.pt"})
    with (
        patch(f"{PERSON}.YOLO_AVAILABLE", True),
        patch(f"{PERSON}.YOLO", Mock(), create=True),
        patch(f"{PERSON}.resolve_yolo_model", return_value=str(weights)) as resolve,
        patch(f"{PERSON}.log_model_download", return_value=Mock(ckpt_path=None)),
    ):
        pipeline._initialize_model()

    resolve.assert_called_once_with("yolo11n-pose.pt")
    [ref] = pipeline.provenance_models()
    assert ref.name == "yolo11n-pose.pt"
    assert ref.revision_kind == "sha256"


def test_a_v150_config_without_model_settings_gets_the_new_defaults():
    diarization = DiarizationPipeline({"min_speakers": 2, "max_speakers": 4})
    person = PersonTrackingPipeline({"conf_threshold": 0.5})
    assert diarization.config["model"] == "pyannote/speaker-diarization-community-1"
    assert person.config["model"] == PersonTrackingPipeline().config["model"]
