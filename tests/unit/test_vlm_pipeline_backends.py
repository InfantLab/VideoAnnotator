"""The VLM pipeline on either backend (spec 033 R3, R7)."""

import json
from typing import ClassVar

import pytest

from tests.fixtures.fake_openai_server import FakeOpenAIServer
from tests.fixtures.synthetic_video import SyntheticVideoSpec, write_synthetic_video_avi
from videoannotator.pipelines.vlm_annotation import openai_client, vlm_pipeline
from videoannotator.pipelines.vlm_annotation.ollama_client import VLMCallResult
from videoannotator.pipelines.vlm_annotation.openai_client import (
    OpenAICompatibleVLMClient,
)

OPENAI = "http://127.0.0.1:8000/v1"


class FakeOllama:
    built: ClassVar[list["FakeOllama"]] = []

    def __init__(self, base_url, timeout):
        FakeOllama.built.append(self)

    def preflight(self, model):
        pass

    def model_details(self, model):
        return {"digest": "d" * 64, "quantization": "Q4_K_M"}

    def chat(self, **kwargs):
        return VLMCallResult("TOUCH", "", 0.1, 0.05, 1, 1, 10.0)


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeOllama.built = []
    monkeypatch.setattr(vlm_pipeline, "OllamaVLMClient", FakeOllama)
    server = FakeOpenAIServer("vllm")
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
    monkeypatch.setattr(vlm_pipeline, "_server_key", lambda base_url: None)
    return server


@pytest.fixture
def video(tmp_path) -> str:
    path = tmp_path / "clip.avi"
    write_synthetic_video_avi(path, SyntheticVideoSpec(fps=10, num_frames=50))
    return str(path)


def _run(video, out, **config):
    p = vlm_pipeline.VLMAnnotationPipeline({"frame_interval_sec": 2.0, **config})
    p.initialize()
    annotations = p.process(video, output_dir=str(out))
    return p, annotations, json.loads((out / "clip_vlm_annotation.json").read_text())


def test_no_backend_is_ollama_as_before(video, tmp_path):
    p, annotations, data = _run(video, tmp_path, base_url="http://127.0.0.1:11434")
    assert len(FakeOllama.built) == 1
    assert {a["backend"] for a in annotations} == {"ollama"}
    first = data["annotations"][0]
    assert first["label"] == "TOUCH"
    assert first["load_time"] == 0.05
    assert first["tokens_per_sec"] == 10.0
    for key in ("fields", "fit", "fit_problem"):
        assert key not in first
    assert "vlm_answer_shape" not in data
    assert p.provenance_vlm()["backend"] == "ollama"
    assert [m.to_dict() for m in p.provenance_models()] == [
        {
            "name": "qwen3.5:9b",
            "source": "ollama",
            "revision": "d" * 64,
            "revision_kind": "ollama-digest",
        }
    ]


def test_openai_compatible_writes_the_same_structure(video, tmp_path, fakes):
    _, ollama_annotations, _ = _run(
        video, tmp_path / "a", base_url="http://127.0.0.1:11434"
    )
    p, annotations, data = _run(
        video,
        tmp_path / "b",
        backend="openai_compatible",
        base_url=OPENAI,
        model=fakes.model,
    )
    assert not FakeOllama.built[1:]
    assert len(annotations) == len(ollama_annotations)
    assert set(annotations[0]) == set(ollama_annotations[0])
    first = data["annotations"][0]
    assert first["backend"] == "openai_compatible"
    assert first["label"] == "TOUCH"
    assert first["load_time"] is None
    assert first["tokens_per_sec"] is None
    vlm = p.provenance_vlm()
    assert vlm["backend"] == "openai_compatible"
    assert vlm["base_url"] == OPENAI
    assert vlm["model_digest"] is None
    (ref,) = [m.to_dict() for m in p.provenance_models()]
    assert ref["source"] == "openai_compatible"
    assert ref["revision_kind"] == "unknown"
    assert ref["revision_note"] == "the server reported no digest"


def test_a_server_fingerprint_is_the_revision(video, tmp_path, monkeypatch):
    server = FakeOpenAIServer("llama_cpp")
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
    p, _, _ = _run(
        video,
        tmp_path,
        backend="openai_compatible",
        base_url=OPENAI,
        model=server.model,
    )
    (ref,) = [m.to_dict() for m in p.provenance_models()]
    assert (ref["revision"], ref["revision_kind"]) == (
        "b6500-1a2b3c4d",
        "server-fingerprint",
    )


def test_the_key_comes_from_the_server_list(video, tmp_path, monkeypatch, fakes):
    fakes.key = "sk-from-settings-12345"
    monkeypatch.setattr(vlm_pipeline, "_server_key", lambda url: fakes.key)
    _run(
        video, tmp_path, backend="openai_compatible", base_url=OPENAI, model=fakes.model
    )
    assert all(r.authorized for r in fakes.requests)


def test_a_refused_key_stops_the_run_and_keeps_what_it_has(video, tmp_path, fakes):
    p = vlm_pipeline.VLMAnnotationPipeline(
        {
            "backend": "openai_compatible",
            "base_url": OPENAI,
            "model": fakes.model,
            "frame_interval_sec": 1.0,
        }
    )
    p.initialize()
    original = OpenAICompatibleVLMClient.chat
    calls = []

    def refuse_after_two(self, **kwargs):
        calls.append(1)
        fakes.refuse_key = len(calls) > 2
        return original(self, **kwargs)

    p._client.chat = refuse_after_two.__get__(p._client)
    with pytest.raises(RuntimeError, match="refused the key"):
        p.process(video, output_dir=str(tmp_path))
    assert len(calls) == 3
    data = json.loads((tmp_path / "clip_vlm_annotation.json").read_text())
    assert len(data["annotations"]) == 2


@pytest.mark.parametrize(
    ("config", "code"),
    [
        ({"backend": "openai_compatible", "base_url": ""}, "MISSING_BASE_URL"),
        ({"api_key": "sk-123"}, "KEY_IN_CONFIG"),
        ({"options": {"auth_token": "abc"}}, "KEY_IN_CONFIG"),
        ({"backend": "bard"}, "INVALID_BACKEND"),
        ({"sampling_mode": "clip"}, "CLIP_NOT_SUPPORTED"),
        ({"answer_shape": []}, "INVALID_ANSWER_SHAPE"),
    ],
)
def test_configs_that_cant_run_are_refused(config, code):
    p = vlm_pipeline.VLMAnnotationPipeline(config)
    with pytest.raises(vlm_pipeline.ConfigProblem) as e:
        p.initialize()
    assert e.value.code == code
    assert not FakeOllama.built


def test_openai_compatible_without_an_address_says_so():
    with pytest.raises(ValueError, match=r"Say which model server to use\."):
        vlm_pipeline.VLMAnnotationPipeline(
            {"backend": "openai_compatible"}
        ).initialize()
