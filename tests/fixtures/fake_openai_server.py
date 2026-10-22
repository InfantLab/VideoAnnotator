"""A fake OpenAI-compatible model server, as an httpx transport (spec 033 R15).

Answers `/v1/models` and `/v1/chat/completions` from replies recorded per
server (`openai_compatible/<flavour>.json`). Switches make it refuse the key,
reject `response_format`, images or video, or be down. Every request is
logged without the key: only whether an Authorization header was present.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

FIXTURES = Path(__file__).parent / "openai_compatible"
FLAVOURS = ("vllm", "llama_cpp", "lm_studio", "ollama_v1")


@dataclass
class Request:
    method: str
    path: str
    body: dict[str, Any] | None
    authorized: bool


@dataclass
class FakeOpenAIServer:
    flavour: str = "vllm"
    refuse_key: bool = False
    reject_schema: bool = False
    reject_images: bool = False
    reject_video: bool = False
    down: bool = False
    # Answers this many chat requests, then refuses the key.
    refuse_key_after: int | None = None
    reasoning: bool = False
    # The key the server accepts; None accepts any (or none).
    key: str | None = None
    # Answers instead of the recorded content, in turn; the last one repeats.
    replies: list[str] = field(default_factory=list)
    # Also answers Ollama's own API at the same host (/api/tags).
    ollama_native: bool = False
    # Fails with this status this many times before answering.
    fail_status: int = 503
    fail_times: int = 0
    requests: list[Request] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.recorded: dict[str, Any] = json.loads(
            (FIXTURES / f"{self.flavour}.json").read_text(encoding="utf-8")
        )
        self.transport = httpx.MockTransport(self._handle)

    @property
    def model(self) -> str:
        return str(self.recorded["models"]["data"][0]["id"])

    @property
    def chat_requests(self) -> list[Request]:
        return [r for r in self.requests if r.path.endswith("/chat/completions")]

    def _error(self, name: str) -> httpx.Response:
        err = self.recorded["errors"][name]
        return httpx.Response(err["status"], json=err["body"])

    def _handle(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        auth = request.headers.get("authorization")
        self.requests.append(
            Request(request.method, request.url.path, body, auth is not None)
        )
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        path = request.url.path
        if path == "/api/tags" and self.ollama_native:
            return httpx.Response(200, json={"models": [{"name": self.model}]})
        if not path.startswith("/v1/"):
            return httpx.Response(404, text="404 page not found")
        if self.refuse_key or (self.key is not None and auth != f"Bearer {self.key}"):
            return self._error("unauthorized")
        if path == "/v1/models" and request.method == "GET":
            return httpx.Response(200, json=self.recorded["models"])
        if path != "/v1/chat/completions" or request.method != "POST":
            return httpx.Response(404, json={"error": "not found"})
        if self.fail_times > 0:
            self.fail_times -= 1
            return httpx.Response(self.fail_status, json={"error": "busy"})
        if self.refuse_key_after is not None:
            if len(self.chat_requests) > self.refuse_key_after:
                return self._error("unauthorized")
        assert body is not None
        parts = body["messages"][0]["content"]
        kinds = {p["type"] for p in parts}
        if "video_url" in kinds and self.reject_video:
            return self._error("video_rejected")
        if "image_url" in kinds and self.reject_images:
            return self._error("images_rejected")
        if "response_format" in body and self.reject_schema:
            return self._error("schema_rejected")
        if "response_format" in body:
            reply = copy.deepcopy(self.recorded["chat_schema"])
        elif self.reasoning:
            reply = copy.deepcopy(self.recorded["chat_reasoning"])
        else:
            reply = copy.deepcopy(self.recorded["chat"])
        if self.replies:
            content = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
            reply["choices"][0]["message"]["content"] = content
        return httpx.Response(200, json=reply)
