"""Shaped questions in the VLM pipeline, on both backends (spec 033 R8)."""

import json
from typing import ClassVar

import pytest

from tests.fixtures.fake_openai_server import FakeOpenAIServer
from tests.fixtures.synthetic_video import SyntheticVideoSpec, write_synthetic_video_avi
from videoannotator.pipelines.vlm_annotation import (
    answer_shape,
    openai_client,
    vlm_pipeline,
)
from videoannotator.pipelines.vlm_annotation.ollama_client import VLMCallResult

OPENAI = "http://127.0.0.1:8000/v1"
SHAPE = [
    {"name": "touch", "kind": "yes_no"},
    {
        "name": "initiator",
        "kind": "choice",
        "values": ["mother", "infant", "both", "none"],
    },
]


class FakeOllama:
    replies: ClassVar[list[str]] = []
    calls: ClassVar[list[dict]] = []

    def __init__(self, base_url, timeout):
        pass

    def preflight(self, model):
        pass

    def model_details(self, model):
        return {"digest": "d" * 64, "quantization": None}

    def chat(self, **kwargs):
        FakeOllama.calls.append(kwargs)
        reply = FakeOllama.replies.pop(0) if FakeOllama.replies else "TOUCH"
        return VLMCallResult(
            reply, "", 0.1, 0.0, 1, 1, 1.0,
            structured_output="enforced" if kwargs.get("schema") else None,
        )  # fmt: skip


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    FakeOllama.replies = []
    FakeOllama.calls = []
    monkeypatch.setattr(vlm_pipeline, "OllamaVLMClient", FakeOllama)
    server = FakeOpenAIServer("llama_cpp")
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
    monkeypatch.setattr(vlm_pipeline, "_server_key", lambda url: None)
    monkeypatch.setattr(vlm_pipeline, "_server_capabilities", lambda url: None)
    recorded: list = []
    monkeypatch.setattr(
        vlm_pipeline,
        "_record_structured_output",
        lambda url, enforced: recorded.append((url, enforced)),
    )
    server.recorded_capabilities = recorded
    return server


@pytest.fixture
def video(tmp_path) -> str:
    path = tmp_path / "clip.avi"
    write_synthetic_video_avi(path, SyntheticVideoSpec(fps=10, num_frames=30))
    return str(path)


def _run(video, out, **config):
    p = vlm_pipeline.VLMAnnotationPipeline({"frame_interval_sec": 1.0, **config})
    p.initialize()
    p.process(video, output_dir=str(out))
    data = json.loads((out / "clip_vlm_annotation.json").read_text())
    return p, data


def test_ollama_gets_the_schema_as_format_and_answers_are_checked(video, tmp_path):
    FakeOllama.replies = [
        '{"touch": "yes", "initiator": "mother"}',
        '{"touch": "yes", "initiator": "mom"}',
        "TOUCH",
    ]
    p, data = _run(
        video, tmp_path, answer_shape=SHAPE, base_url="http://127.0.0.1:11434"
    )
    schema = FakeOllama.calls[0]["schema"]
    assert schema == answer_shape.to_json_schema(answer_shape.validate_shape(SHAPE))
    first, second, third = data["annotations"]
    assert (first["label"], first["fields"], first["fit"]) == (
        "yes",
        {"touch": "yes", "initiator": "mother"},
        True,
    )
    assert first["fit_problem"] is None
    assert (second["label"], second["fit"]) == ("DIDN'T FIT", False)
    assert (
        second["fit_problem"]
        == "initiator: 'mom' is not one of mother, infant, both, none"
    )
    assert second["status"] == "answered"
    assert third["fit_problem"] == "The reply isn't JSON."
    assert data["vlm_answer_shape"] == SHAPE
    vlm = p.provenance_vlm()
    assert vlm["answer_shape_sha256"] == answer_shape.shape_sha256(
        answer_shape.validate_shape(SHAPE)
    )
    assert p.run_summary()["structured_output"] == "enforced"


def test_misfits_count_as_answered(video, tmp_path):
    FakeOllama.replies = ["nonsense"] * 10
    _, data = _run(
        video,
        tmp_path,
        answer_shape=SHAPE,
        base_url="http://127.0.0.1:11434",
        abort_after_consecutive_failures=2,
    )
    assert len(data["annotations"]) == 3


def test_openai_compatible_sends_response_format(video, tmp_path, fakes):
    _, data = _run(
        video,
        tmp_path,
        backend="openai_compatible",
        base_url=OPENAI,
        model=fakes.model,
        answer_shape=SHAPE,
    )
    body = fakes.chat_requests[0].body
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["name"] == "answer"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["messages"][0]["content"][0]["text"] == vlm_pipeline.DEFAULT_PROMPT
    assert {a["fit"] for a in data["annotations"]} == {True}
    assert fakes.recorded_capabilities == [(OPENAI, True)]


def test_a_server_that_rejects_the_schema_gets_it_in_the_prompt(video, tmp_path, fakes):
    fakes.reject_schema = True
    fakes.replies = ['{"touch": "no", "initiator": "none"}']
    p, data = _run(
        video,
        tmp_path,
        backend="openai_compatible",
        base_url=OPENAI,
        model=fakes.model,
        answer_shape=SHAPE,
    )
    with_schema = [r for r in fakes.chat_requests if "response_format" in r.body]
    assert len(with_schema) == 1
    later = fakes.chat_requests[-1].body
    assert "response_format" not in later
    assert later["messages"][0]["content"][0]["text"].endswith(
        '{"touch": "yes" or "no", "initiator": "mother" or "infant" or "both" or '
        '"none"}'
    )
    assert {a["label"] for a in data["annotations"]} == {"no"}
    assert p.run_summary()["structured_output"] == "prompt_only"
    assert fakes.recorded_capabilities == [(OPENAI, False)]


def test_unshaped_output_is_unchanged(video, tmp_path):
    _, data = _run(video, tmp_path, base_url="http://127.0.0.1:11434")
    for a in data["annotations"]:
        for key in ("fields", "fit", "fit_problem"):
            assert key not in a
    assert "vlm_answer_shape" not in data
    assert "schema" not in FakeOllama.calls[0]
