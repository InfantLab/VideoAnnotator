"""Each pipeline starts from the same torch settings and can't leak its own.

OpenFace 3 turned on `cudnn.benchmark` for the rest of the process, so later
pipelines in the job gave different results than they did alone.
"""

from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest

from videoannotator.batch.job_execution import run_job_pipelines
from videoannotator.batch.types import BatchJob, JobStatus
from videoannotator.storage.base import StorageBackend
from videoannotator.utils.torch_settings import (
    apply_torch_settings,
    restored_torch_settings,
)

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def torch_defaults():
    with restored_torch_settings():
        apply_torch_settings(deterministic=False)
        yield


def test_restored_settings_undo_changes_made_inside():
    with restored_torch_settings():
        torch.backends.cudnn.benchmark = True
        torch.autograd.set_detect_anomaly(True)
        torch.use_deterministic_algorithms(True)
    assert torch.backends.cudnn.benchmark is False
    assert torch.is_anomaly_enabled() is False
    assert torch.are_deterministic_algorithms_enabled() is False


def test_deterministic_settings(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    settings = apply_torch_settings(deterministic=True)
    assert settings["cudnn_benchmark"] is False
    assert settings["cudnn_deterministic"] is True
    assert settings["deterministic_algorithms"] is True
    assert torch.is_deterministic_algorithms_warn_only_enabled()
    assert settings["cublas_workspace_config"] == ":4096:8"


def _job(tmp_path: Path, pipelines: list[str], **config) -> BatchJob:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    return BatchJob(
        video_path=video,
        output_dir=tmp_path / "out",
        selected_pipelines=pipelines,
        config=config,
    )


def _storage() -> MagicMock:
    storage = MagicMock(spec=StorageBackend)
    storage.annotation_exists.return_value = False
    storage.load_job_metadata.return_value = None
    storage.save_annotations.return_value = "fake://annotations"
    return storage


class _Recorder:
    """A fake pipeline that records the settings it ran with."""

    seen: ClassVar[dict[str, dict]] = {}
    name = "recorder"
    benchmark_on_init = False

    def __init__(self, config):
        pass

    def initialize(self):
        if self.benchmark_on_init:
            torch.backends.cudnn.benchmark = True

    def process(self, **kwargs):
        _Recorder.seen[self.name] = {
            "benchmark": torch.backends.cudnn.benchmark,
            "deterministic": torch.backends.cudnn.deterministic,
        }
        return []

    def cleanup(self):
        pass


class _Leaky(_Recorder):
    name = "leaky"
    benchmark_on_init = True


class _Next(_Recorder):
    name = "next"


def test_a_pipeline_cannot_change_settings_for_itself_or_the_next(tmp_path):
    _Recorder.seen = {}
    job = run_job_pipelines(
        _job(tmp_path, ["leaky", "next"]),
        _storage(),
        {"leaky": _Leaky, "next": _Next},
    )
    assert job.status == JobStatus.COMPLETED
    assert _Recorder.seen["leaky"]["benchmark"] is False
    assert _Recorder.seen["next"]["benchmark"] is False
    assert torch.backends.cudnn.benchmark is False


def test_deterministic_job_setting_reaches_every_pipeline(tmp_path):
    _Recorder.seen = {}
    run_job_pipelines(
        _job(tmp_path, ["leaky", "next"], deterministic=True),
        _storage(),
        {"leaky": _Leaky, "next": _Next},
    )
    assert _Recorder.seen["leaky"] == {"benchmark": False, "deterministic": True}
    assert _Recorder.seen["next"] == {"benchmark": False, "deterministic": True}
    assert torch.backends.cudnn.deterministic is False, "restored after the job"
