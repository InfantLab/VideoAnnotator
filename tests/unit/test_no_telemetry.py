"""No telemetry from VideoAnnotator or the libraries it drives (constitution I)."""

import os

import pytest


def test_library_telemetry_off_by_default():
    import videoannotator  # noqa: F401

    assert os.environ.get("PYANNOTE_METRICS_ENABLED") == "0"
    assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"


def test_ultralytics_events_off_once_person_tracking_is_imported():
    pytest.importorskip("ultralytics")
    from ultralytics.utils.events import events

    import videoannotator.pipelines.person_tracking.person_pipeline  # noqa: F401

    assert events.enabled is False


def test_onnxruntime_telemetry_off_once_audio_pipelines_are_imported():
    onnxruntime = pytest.importorskip("onnxruntime")
    from unittest.mock import patch

    import videoannotator.pipelines.audio_processing as audio

    with patch.object(onnxruntime, "disable_telemetry_events") as disable:
        import importlib

        importlib.reload(audio)
    disable.assert_called_once()
