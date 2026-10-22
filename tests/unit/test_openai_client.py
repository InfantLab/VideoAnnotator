"""The OpenAI-compatible client against each server's recorded replies (spec 033)."""

import base64
import json
import logging

import httpx
import pytest

from tests.fixtures.fake_openai_server import FLAVOURS, FakeOpenAIServer
from videoannotator.pipelines.vlm_annotation import openai_client
from videoannotator.pipelines.vlm_annotation.clients import (
    KeyRefusedError,
    ModelServerUnavailableError,
    WrongKindError,
)
from videoannotator.pipelines.vlm_annotation.openai_client import (
    OpenAICompatibleVLMClient,
)

BASE = "http://127.0.0.1:8000/v1"
KEY = "sk-test-0123456789abcdefSECRET"
JPEG = b"\xff\xd8\xff\xe0fake-jpeg"


def _client(server: FakeOpenAIServer, key: str | None = None, base: str = BASE):
    return OpenAICompatibleVLMClient(base, 30, api_key=key, transport=server.transport)


def _chat(client, **kwargs):
    args = {
        "model": "m",
        "prompt": "Touching?",
        "images": [JPEG, JPEG],
        "think": False,
        "keep_alive": "2h",
        "options": {"temperature": 0, "top_p": 0.9},
        "max_retries": 3,
        "retry_backoff_sec": 0,
    }
    args.update(kwargs)
    return client.chat(**args)


@pytest.fixture(params=FLAVOURS)
def server(request):
    return FakeOpenAIServer(flavour=request.param)


class TestModels:
    def test_lists_the_servers_models(self, server):
        assert _client(server).list_models() == [
            m["id"] for m in server.recorded["models"]["data"]
        ]

    def test_details_have_no_digest(self, server):
        details = _client(server).model_details(server.model)
        assert details["digest"] is None

    def test_vllm_reports_the_root(self):
        server = FakeOpenAIServer("vllm")
        assert _client(server).model_details(server.model)["root"] == server.model

    def test_down_is_unavailable(self, server):
        server.down = True
        with pytest.raises(ModelServerUnavailableError):
            _client(server).list_models()

    def test_without_v1_is_the_wrong_kind(self, server):
        with pytest.raises(WrongKindError):
            _client(server, base="http://127.0.0.1:8000").list_models()

    def test_refused_key(self, server):
        server.refuse_key = True
        with pytest.raises(KeyRefusedError):
            _client(server, key=KEY).list_models()

    def test_preflight_names_what_the_server_lists(self, server):
        with pytest.raises(ModelServerUnavailableError, match=server.model):
            _client(server).preflight("not-there")


class TestChat:
    def test_one_image_answer(self, server):
        result = _chat(_client(server), images=[JPEG])
        assert result.error is None
        assert result.raw_text == "TOUCH"
        assert result.load_time is None
        assert result.prompt_tokens > 0
        assert result.resp_tokens > 0

    def test_reasoning_from_either_field(self, server):
        server.reasoning = True
        result = _chat(_client(server))
        assert result.raw_text == "NO_TOUCH"
        assert result.thinking

    def test_body_has_one_image_part_per_frame_and_no_ollama_keys(self, server):
        _chat(_client(server), images=[JPEG, JPEG, JPEG])
        body = server.chat_requests[0].body
        parts = body["messages"][0]["content"]
        assert parts[0] == {"type": "text", "text": "Touching?"}
        images = [p for p in parts if p["type"] == "image_url"]
        assert len(images) == 3
        url = images[0]["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == JPEG
        for key in ("keep_alive", "think", "options", "images", "format"):
            assert key not in body
        assert body["temperature"] == 0
        assert body["stream"] is False

    def test_num_predict_is_max_tokens(self, server):
        _chat(_client(server), options={"num_predict": 64, "seed": 7})
        body = server.chat_requests[0].body
        assert body["max_tokens"] == 64
        assert body["seed"] == 7

    def test_llama_cpp_reports_its_rate_and_fingerprint(self):
        server = FakeOpenAIServer("llama_cpp")
        result = _chat(_client(server))
        assert result.tokens_per_sec == pytest.approx(48.8)
        assert result.system_fingerprint == "b6500-1a2b3c4d"

    def test_without_a_rate_it_stays_none(self):
        result = _chat(_client(FakeOpenAIServer("vllm")))
        assert result.tokens_per_sec is None

    def test_retries_a_503_then_answers(self, server):
        server.fail_times = 2
        result = _chat(_client(server))
        assert result.error is None
        assert len(server.chat_requests) == 3

    def test_gives_up_after_the_retries(self, server):
        server.fail_times = 10
        result = _chat(_client(server), max_retries=2)
        assert "503" in result.error
        assert len(server.chat_requests) == 2

    def test_read_timeout_is_not_retried(self):
        calls = []

        def handler(request):
            calls.append(request)
            raise httpx.ReadTimeout("slow", request=request)

        client = OpenAICompatibleVLMClient(
            BASE, 30, transport=httpx.MockTransport(handler)
        )
        result = _chat(client)
        assert result.error.startswith("read timeout")
        assert len(calls) == 1

    def test_refused_key_raises_and_is_not_retried(self, server):
        server.refuse_key = True
        with pytest.raises(KeyRefusedError):
            _chat(_client(server, key=KEY))
        assert len(server.chat_requests) == 1

    def test_images_rejected_is_a_plain_error(self, server):
        server.reject_images = True
        result = _chat(_client(server))
        assert result.error == "This model doesn't accept images."
        assert len(server.chat_requests) == 1


class TestKeys:
    def test_key_only_in_the_header(self, server):
        _chat(_client(server, key=KEY))
        request = server.chat_requests[0]
        assert request.authorized
        assert KEY not in json.dumps(request.body)

    def test_no_header_without_a_key(self, server):
        _chat(_client(server))
        assert not server.chat_requests[0].authorized

    def test_key_appears_in_no_error_or_log(self, server, caplog):
        caplog.set_level(logging.DEBUG)
        client = _client(server, key=KEY)
        texts = []
        server.refuse_key = True
        for call in (client.list_models, lambda: _chat(client)):
            with pytest.raises(KeyRefusedError) as e:
                call()
            texts.append(str(e.value))
        server.refuse_key = False
        server.down = True
        with pytest.raises(ModelServerUnavailableError) as e:
            client.list_models()
        texts.append(str(e.value))
        texts.append(_chat(client).error or "")
        for text in texts:
            assert KEY not in text
        for record in caplog.records:
            assert KEY not in record.getMessage()

    def test_a_server_echoing_the_key_is_scrubbed(self):
        def handler(request):
            return httpx.Response(400, json={"error": {"message": f"bad key {KEY}"}})

        client = OpenAICompatibleVLMClient(
            BASE, 30, api_key=KEY, transport=httpx.MockTransport(handler)
        )
        result = _chat(client, images=[])
        assert KEY not in result.error
        assert "<redacted>" in result.error


def test_default_transport_is_used(monkeypatch):
    server = FakeOpenAIServer("vllm")
    monkeypatch.setattr(openai_client, "DEFAULT_TRANSPORT", server.transport)
    assert OpenAICompatibleVLMClient(BASE, 30).list_models() == [server.model]
