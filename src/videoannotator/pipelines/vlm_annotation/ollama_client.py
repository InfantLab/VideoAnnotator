"""Thin client wrapper around the `ollama` Python package.

Ported from mother-infant-touch-detection's core/inference.py and
core/context_inference.py: every call is one fresh, stateless
`client.chat(...)` — no conversation history is carried between calls, so a
prediction for one frame (or frame burst) can never be biased by a previous
one. That "reset every time" property is the whole point of the harness, not
an implementation detail — see docs/research_roadmap.md in the touch-
detection repo for why it matters for this kind of annotation task.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx
from ollama import Client


class OllamaUnavailableError(RuntimeError):
    """The configured Ollama server isn't reachable, or the model isn't pulled."""


@dataclass
class VLMCallResult:
    raw_text: str
    thinking: str
    total_time: float
    load_time: float
    prompt_tokens: int
    resp_tokens: int
    tokens_per_sec: float
    error: str | None = None


class OllamaVLMClient:
    """Wraps `ollama.Client` with the preflight/retry behaviour the touch-
    detection pipeline relies on in production."""

    def __init__(self, base_url: str, timeout: int):
        self.base_url = base_url
        self._client = Client(host=base_url, timeout=timeout)

    def list_models(self) -> list[str]:
        """Return the names of models currently pulled on this server.

        Raises:
            OllamaUnavailableError: the server isn't reachable at all --
                distinct from a reachable server that simply has zero
                models pulled (which returns an empty list, not an error;
                spec 009 FR-005).
        """
        try:
            return [
                m.get("model", m.get("name", ""))
                for m in self._client.list().get("models", [])
            ]
        except Exception as exc:
            raise OllamaUnavailableError(
                f"Cannot reach ollama server at {self.base_url} ({exc}). "
                "Start it with 'ollama serve' and retry."
            ) from exc

    def preflight(self, model_name: str) -> None:
        """Raise OllamaUnavailableError if the server is down or the model
        isn't pulled. Call once before processing any video."""
        names = self.list_models()
        if model_name not in names:
            raise OllamaUnavailableError(
                f"Model {model_name!r} not found in ollama. Available: {names}. "
                f"Pull it first: ollama pull {model_name}"
            )

    def chat(
        self,
        *,
        model: str,
        prompt: str,
        images: list[bytes],
        think: bool,
        keep_alive: str,
        options: dict[str, Any],
        max_retries: int,
        retry_backoff_sec: int,
    ) -> VLMCallResult:
        """One stateless chat call. Retries transient failures with
        exponential backoff; a read-timeout is not retried since it almost
        always repeats (matches inference.py's behaviour)."""
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            t0 = time.time()
            try:
                response = self._client.chat(
                    model=model,
                    options=options,
                    keep_alive=keep_alive,
                    think=think,
                    messages=[{"role": "user", "content": prompt, "images": images}],
                )
            except httpx.ReadTimeout:
                dt = time.time() - t0
                return VLMCallResult(
                    raw_text="",
                    thinking="",
                    total_time=dt,
                    load_time=0.0,
                    prompt_tokens=0,
                    resp_tokens=0,
                    tokens_per_sec=0.0,
                    error=f"read timeout after {dt:.1f}s",
                )
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(retry_backoff_sec * (2 ** (attempt - 1)))
                    continue
                dt = time.time() - t0
                return VLMCallResult(
                    raw_text="",
                    thinking="",
                    total_time=dt,
                    load_time=0.0,
                    prompt_tokens=0,
                    resp_tokens=0,
                    tokens_per_sec=0.0,
                    error=str(exc),
                )
            else:
                dt = time.time() - t0
                msg = response.get("message", {})
                raw = (
                    msg.get("content", "")
                    if isinstance(msg, dict)
                    else getattr(msg, "content", "")
                )
                thinking = (
                    msg.get("thinking")
                    if isinstance(msg, dict)
                    else getattr(msg, "thinking", "")
                ) or ""
                load_dur = response.get("load_duration", 0) / 1e9
                eval_dur = response.get("eval_duration", 0) / 1e9
                prompt_tokens = response.get("prompt_eval_count", 0)
                resp_tokens = response.get("eval_count", 0)
                tokens_per_sec = resp_tokens / eval_dur if eval_dur > 0 else 0.0
                return VLMCallResult(
                    raw_text=raw,
                    thinking=thinking,
                    total_time=dt,
                    load_time=load_dur,
                    prompt_tokens=prompt_tokens,
                    resp_tokens=resp_tokens,
                    tokens_per_sec=tokens_per_sec,
                    error=None,
                )
        return VLMCallResult(
            raw_text="",
            thinking="",
            total_time=0.0,
            load_time=0.0,
            prompt_tokens=0,
            resp_tokens=0,
            tokens_per_sec=0.0,
            error=str(last_exc) if last_exc else "unknown error",
        )
