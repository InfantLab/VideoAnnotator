"""POST /api/v1/ask/suggest-shape (spec 033 R11)."""

import json

import pytest
from fastapi.testclient import TestClient

from tests.fixtures.fake_openai_server import FakeOpenAIServer
from videoannotator.api.main import app
from videoannotator.pipelines.vlm_annotation import openai_client

client = TestClient(app)

LAB = "http://127.0.0.1:8000/v1"


@pytest.fixture
def fake(monkeypatch):
    server = FakeOpenAIServer("vllm")
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
    return server


def _suggest(fake, prompt, **kwargs):
    body = {
        "prompt": prompt,
        "model": fake.model,
        "base_url": LAB,
        "backend": "openai_compatible",
    }
    body.update(kwargs)
    r = client.post("/api/v1/ask/suggest-shape", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_the_rule_makes_no_model_call(fake):
    body = _suggest(fake, "Is the adult touching? Return ONLY: TOUCH or NO_TOUCH")
    assert body == {
        "shape": [
            {"name": "answer", "kind": "choice", "values": ["TOUCH", "NO_TOUCH"]}
        ],
        "made_by": "rule",
        "message": "Suggested from your question.",
    }
    assert not fake.requests


def test_the_model_is_asked_text_only_when_the_rule_finds_nothing(fake):
    fake.replies = [
        json.dumps(
            {
                "fields": [
                    {"name": "touch", "kind": "yes_no"},
                    {
                        "name": "initiator",
                        "kind": "choice",
                        "values": ["adult", "child"],
                    },
                ]
            }
        )
    ]
    body = _suggest(fake, "Is the adult touching the child? Who started it?")
    assert body["made_by"] == "model"
    assert body["shape"] == [
        {"name": "touch", "kind": "yes_no"},
        {"name": "initiator", "kind": "choice", "values": ["adult", "child"]},
    ]
    (call,) = fake.chat_requests
    parts = call.body["messages"][0]["content"]
    assert [p["type"] for p in parts] == ["text"]
    assert "Who started it?" in parts[0]["text"]
    assert "response_format" in call.body


def test_an_unusable_proposal_is_no_shape(fake):
    fake.replies = [json.dumps({"fields": [{"name": "Label!", "kind": "text"}]})]
    body = _suggest(fake, "What is happening?")
    assert (body["shape"], body["made_by"]) == (None, None)
    assert "Add fields yourself" in body["message"]


def test_without_a_model_only_the_rule(fake):
    body = _suggest(fake, "What is happening?", model=None)
    assert body["shape"] is None
    assert not fake.requests
