"""sampling_mode: clip (spec 033 R12)."""

import json

import pytest

from tests.fixtures.fake_openai_server import FakeOpenAIServer
from tests.fixtures.synthetic_video import SyntheticVideoSpec, write_synthetic_video_avi
from videoannotator.pipelines.vlm_annotation import openai_client, vlm_pipeline

OPENAI = "http://127.0.0.1:8000/v1"


@pytest.fixture(autouse=True)
def server(monkeypatch):
    fake = FakeOpenAIServer("vllm")
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", fake.transport)
    monkeypatch.setattr(vlm_pipeline, "_server_key", lambda url: None)
    return fake


@pytest.fixture
def video(tmp_path) -> str:
    path = tmp_path / "clip.avi"
    write_synthetic_video_avi(path, SyntheticVideoSpec(fps=10, num_frames=60))
    return str(path)


def _pipeline(server, **config):
    return vlm_pipeline.VLMAnnotationPipeline(
        {
            "backend": "openai_compatible",
            "base_url": OPENAI,
            "model": server.model,
            "sampling_mode": "clip",
            "clip_length_sec": 2.0,
            "clip_fps": 2.0,
            "frame_interval_sec": 2.5,
            **config,
        }
    )


def test_each_sample_is_one_clip(server, video, tmp_path):
    p = _pipeline(server)
    p.initialize()
    p.process(video, output_dir=str(tmp_path))
    data = json.loads((tmp_path / "clip_vlm_annotation.json").read_text())
    first, second = data["annotations"][:2]
    assert (first["clip_start_sec"], first["clip_end_sec"], first["clip_fps"]) == (
        0.0,
        1.0,
        2.0,
    )
    assert first["context_frame_numbers"] == [0, 5]
    assert (second["clip_start_sec"], second["clip_end_sec"]) == (1.5, 3.5)
    assert second["frame_times"] == [1.5, 2.0, 2.5, 3.0]
    assert second["sampling_mode"] == "clip"
    probe, *calls = server.chat_requests
    assert probe.body["max_tokens"] == 1
    parts = calls[1].body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text", "video_url"]
    assert len(parts[1]["video_url"]["url"].split(",")) == 1 + 4
    assert calls[1].body["media_io_kwargs"] == {"video": {"fps": 2.0}}


def test_a_model_without_video_is_refused_before_any_frame(server, video):
    server.reject_video = True
    with pytest.raises(vlm_pipeline.ConfigProblem) as e:
        _pipeline(server).initialize()
    assert e.value.code == "CLIP_NOT_SUPPORTED"
    assert len(server.chat_requests) == 1


def test_ollama_with_clip_is_refused():
    with pytest.raises(vlm_pipeline.ConfigProblem) as e:
        vlm_pipeline.VLMAnnotationPipeline({"sampling_mode": "clip"}).initialize()
    assert e.value.code == "CLIP_NOT_SUPPORTED"
