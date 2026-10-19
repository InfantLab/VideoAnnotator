"""A VLM run writes its answers as they come and can stop keeping them (spec 027)."""

import json
import threading
from pathlib import Path

import pytest

from tests.fixtures.synthetic_video import SyntheticVideoSpec, write_synthetic_video_avi
from videoannotator.pipelines.vlm_annotation import vlm_pipeline
from videoannotator.pipelines.vlm_annotation.ollama_client import VLMCallResult


class FakeClient:
    block_on_call: int | None = None
    blocked = threading.Event()
    release = threading.Event()

    def __init__(self, base_url, timeout):
        self.calls: list[int] = []

    def preflight(self, model):
        pass

    def model_details(self, model):
        return {"digest": "d" * 64, "quantization": "Q4_K_M"}

    def chat(self, *, images, **kwargs):
        self.calls.append(len(images))
        if FakeClient.block_on_call == len(self.calls):
            FakeClient.blocked.set()
            FakeClient.release.wait(10)
        return VLMCallResult("TOUCH", "", 0.1, 0.0, 1, 1, 1.0)


@pytest.fixture(autouse=True)
def fake_client(monkeypatch):
    FakeClient.block_on_call = None
    FakeClient.blocked = threading.Event()
    FakeClient.release = threading.Event()
    monkeypatch.setattr(vlm_pipeline, "OllamaVLMClient", FakeClient)
    # Every rewrite is due: the once-a-second throttle would otherwise make
    # what's on disk depend on how fast the test machine is.
    clock = iter(range(0, 10_000, 5))
    monkeypatch.setattr(vlm_pipeline.time, "monotonic", lambda: float(next(clock)))
    yield
    FakeClient.release.set()


@pytest.fixture
def video(tmp_path) -> str:
    path = tmp_path / "clip.avi"
    write_synthetic_video_avi(path, SyntheticVideoSpec(fps=10, num_frames=100))
    return str(path)


def _pipeline(**config):
    p = vlm_pipeline.VLMAnnotationPipeline(
        {"base_url": "http://127.0.0.1:11434", **config}
    )
    p.initialize()
    return p


def _results(out: Path) -> dict:
    return json.loads((out / "clip_vlm_annotation.json").read_text())


def test_the_file_has_answers_while_the_run_goes(video, tmp_path):
    out = tmp_path / "out"
    FakeClient.block_on_call = 4
    p = _pipeline(frame_interval_sec=1.0)
    runner = threading.Thread(
        target=p.process, args=(video,), kwargs={"output_dir": str(out)}
    )
    runner.start()
    try:
        assert FakeClient.blocked.wait(10)
        data = _results(out)
        assert data["vlm_progress"] == {
            "planned": 11,
            "done": 3,
            "finished": False,
            "stopped_early": False,
        }
        assert len(data["annotations"]) == 3
        assert not list(out.glob("*.tmp"))
    finally:
        FakeClient.release.set()
        runner.join(10)
    final = _results(out)
    assert final["vlm_progress"]["finished"] is True
    assert len(final["annotations"]) == 10


def test_stop_returns_normally_keeping_the_answers(video, tmp_path):
    out = tmp_path / "out"
    p = _pipeline(frame_interval_sec=1.0)
    p.set_stop_check(lambda: len(p._client.calls) >= 5)
    annotations = p.process(video, output_dir=str(out))
    assert len(annotations) == 5
    data = _results(out)
    assert data["vlm_progress"]["stopped_early"] is True
    assert data["vlm_progress"]["done"] == 5
    assert len(data["annotations"]) == 5
    assert p.run_summary() == {
        "samples_planned": 11,
        "samples_answered": 5,
        "stopped_early": True,
    }


def test_stopped_before_any_answer_is_not_a_failure(video, tmp_path):
    p = _pipeline(frame_interval_sec=1.0)
    p.set_stop_check(lambda: True)
    assert p.process(video, output_dir=str(tmp_path)) == []
    assert p.run_summary()["stopped_early"] is True


def test_an_unreadable_frame_is_listed_not_answered(video, tmp_path):
    # The grid's last point is t = 10.0 s: frame 100 of a 100-frame video.
    p = _pipeline(frame_interval_sec=1.0)
    annotations = p.process(video, output_dir=str(tmp_path))
    data = _results(tmp_path)
    assert data["vlm_skipped"] == [{"t": 10.0, "reason": "no readable frame"}]
    assert [a["timestamp_sec"] for a in annotations] == [float(t) for t in range(10)]
    assert {a["status"] for a in annotations} == {"answered"}
    assert annotations[3]["frame_times"] == [3.0]


def test_range_and_moments_from_config(video, tmp_path):
    p = _pipeline(frame_interval_sec=2.0, start_sec=3.0, end_sec=8.0)
    assert [a["timestamp_sec"] for a in p.process(video)] == [3.0, 5.0, 7.0]
    p = _pipeline(moments_sec=[7.5, 1.0, 1.0, 42.0])
    assert [a["timestamp_sec"] for a in p.process(video)] == [1.0, 7.5]


def test_burst_spacing(video):
    p = _pipeline(
        sampling_mode="frame_burst",
        burst_offsets=[-1, 0, 1],
        burst_spacing_sec=0.5,
        moments_sec=[5.0],
    )
    [a] = p.process(video)
    assert a["context_frame_numbers"] == [45, 50, 55]
    assert a["frame_times"] == [4.5, 5.0, 5.5]


def test_old_config_means_what_it_did(video):
    """No new keys: every interval from 0, burst frames one interval apart."""
    p = _pipeline(sampling_mode="frame_burst", frame_interval_sec=2.0)
    annotations = p.process(video)
    assert [a["timestamp_sec"] for a in annotations] == [0, 2, 4, 6, 8, 10]
    assert annotations[2]["context_frame_offsets"] == [-2, -1, 0, 1, 2]
    assert annotations[2]["context_frame_numbers"] == [0, 20, 40, 60, 80]
    assert annotations[0]["context_frame_offsets"] == [0, 1, 2]
    # t = 10 s is the video's end: frame 100 doesn't exist, so it isn't sent.
    assert annotations[5]["context_frame_numbers"] == [60, 80]
