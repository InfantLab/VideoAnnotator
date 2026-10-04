"""`videoannotator process`: one job, run here, through the shared job path.

It was a stub that printed "not yet implemented" until v1.6.0.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from videoannotator.api.database import get_storage_backend, reset_storage_backend
from videoannotator.cli import app
from videoannotator.storage.manager import get_storage_provider

runner = CliRunner()


class FakeScenePipeline:
    fail = False

    def __init__(self, config):
        self.config = config

    def initialize(self):
        pass

    def cleanup(self):
        pass

    def process(self, video_path, start_time, end_time, pps, output_dir):
        if self.fail:
            raise RuntimeError("Scene detection failed: bad codec")
        stem = Path(video_path).stem
        (Path(output_dir) / f"{stem}_scene_detection.json").write_text("{}")
        return [{"id": 1}]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    # The provider is cached with its root; don't leak this one to later tests.
    get_storage_provider.cache_clear()
    reset_storage_backend()
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video")
    FakeScenePipeline.fail = False
    with patch("videoannotator.batch.local_job.get_pipeline_loader") as loader:
        loader.return_value.load_all_pipelines.return_value = {
            "scene_detection": FakeScenePipeline
        }
        yield video, tmp_path
    reset_storage_backend()
    get_storage_provider.cache_clear()


def _job_id(output: str) -> str:
    line = next(line for line in output.splitlines() if line.startswith("[INFO] Job "))
    return line.split()[2].rstrip(":")


def test_runs_the_job_and_records_it(env):
    video, tmp_path = env
    result = runner.invoke(
        app,
        ["process", str(video), "--pipelines", "scene_detection",
         "--output", str(tmp_path / "out")],
    )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert "[OK] scene_detection" in result.output
    assert (tmp_path / "out" / "clip_scene_detection.json").is_file()

    job = get_storage_backend().load_job_metadata(_job_id(result.output))
    assert job.status.value == "completed"
    assert job.video_path.parent == job.storage_path
    assert video.is_file(), "the original video must stay where it was"


def test_a_failed_pipeline_exits_1_with_its_error(env):
    video, _ = env
    FakeScenePipeline.fail = True
    result = runner.invoke(
        app, ["process", str(video), "--pipelines", "scene_detection"]
    )
    assert result.exit_code == 1
    assert "[FAILED] scene_detection: Scene detection failed: bad codec" in (
        result.output
    )


def test_unknown_pipeline_is_rejected_before_any_work(env):
    video, tmp_path = env
    result = runner.invoke(app, ["process", str(video), "--pipelines", "scenes"])
    assert result.exit_code == 1
    assert "Unknown pipeline 'scenes'" in result.output
    assert "scene_detection" in result.output
    assert not (tmp_path / "storage").exists() or not any(
        (tmp_path / "storage").iterdir()
    )


def test_config_file_reaches_the_pipeline(env):
    video, tmp_path = env
    config = tmp_path / "settings.yaml"
    config.write_text("scene_detection:\n  threshold: 12.5\n")
    seen = {}

    def capture(self, config):
        seen.update(config)

    with patch.object(FakeScenePipeline, "__init__", capture):
        result = runner.invoke(
            app,
            ["process", str(video), "--pipelines", "scene_detection",
             "--config", str(config)],
        )  # fmt: skip
    assert result.exit_code == 0, result.output
    assert seen == {"threshold": 12.5}
