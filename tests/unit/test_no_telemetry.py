"""No telemetry from VideoAnnotator or the libraries it drives (constitution I)."""

import os

import pytest


def test_library_telemetry_off_by_default():
    import videoannotator  # noqa: F401

    assert os.environ.get("PYANNOTE_METRICS_ENABLED") == "0"
    assert os.environ.get("HF_HUB_DISABLE_TELEMETRY") == "1"


def test_ultralytics_events_off_once_person_tracking_is_imported():
    pytest.importorskip("ultralytics")
    from ultralytics.hub.utils import events

    import videoannotator.pipelines.person_tracking.person_pipeline  # noqa: F401

    assert events.enabled is False
