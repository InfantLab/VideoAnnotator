"""The two kinds of model server a VLM call can go to (spec 033 R1).

`ollama` keeps its own client, which reads model digests for provenance (R6);
`openai_compatible` covers llama.cpp, vLLM, LM Studio, SGLang and hosted APIs
through the one contract they share. Callers choose with `make_client`.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from .ollama_client import OllamaUnavailableError, VLMCallResult

Backend = Literal["ollama", "openai_compatible"]
BACKENDS: tuple[Backend, ...] = ("ollama", "openai_compatible")


class KeyRefusedError(RuntimeError):
    """The model server refused the key (401/403). Never retried: a run stops."""


class ModelServerUnavailableError(OllamaUnavailableError):
    """The model server isn't reachable, or the model isn't on it.

    A subclass of OllamaUnavailableError so existing handlers keep working."""


class WrongKindError(ModelServerUnavailableError):
    """Something answers at the address, but not the API of this kind."""


class VLMClient(Protocol):
    base_url: str

    def list_models(self) -> list[str]: ...

    def model_details(self, model_name: str) -> dict[str, str | None]: ...

    def preflight(self, model_name: str) -> None: ...

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
        schema: dict[str, Any] | None = None,
        video: list[bytes] | None = None,
        video_fps: float | None = None,
    ) -> VLMCallResult: ...


def make_client(
    backend: str,
    base_url: str,
    timeout: int,
    api_key: str | None = None,
    *,
    ollama_class: Any = None,
) -> VLMClient:
    """A client for `backend` at `base_url`.

    `ollama_class` lets a module pass the OllamaVLMClient name it imported, so
    tests that replace that name keep replacing the client actually used.
    """
    if backend == "openai_compatible":
        from .openai_client import OpenAICompatibleVLMClient

        return OpenAICompatibleVLMClient(base_url, timeout, api_key=api_key)
    if backend != "ollama":
        raise ValueError(
            f"Unknown model server kind {backend!r}: use ollama or openai_compatible."
        )
    if ollama_class is None:
        from .ollama_client import OllamaVLMClient

        ollama_class = OllamaVLMClient
    client: VLMClient = ollama_class(base_url=base_url, timeout=timeout)
    return client


# ---------------------------------------------------------------------------
# Checking a server (spec 033 R13, FR-005)
# ---------------------------------------------------------------------------


def _get(url: str, api_key: str | None) -> tuple[int | None, Any]:
    """(status, JSON body) of a GET, or (None, None) when nothing answers."""
    import httpx

    from . import openai_client

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        with httpx.Client(
            timeout=10, headers=headers, transport=openai_client.DEFAULT_TRANSPORT
        ) as http:
            response = http.get(url)
    except httpx.HTTPError:
        return None, None
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, None


def _host_root(base_url: str) -> str:
    from urllib.parse import urlsplit

    parts = urlsplit(base_url)
    return f"{parts.scheme}://{parts.netloc}"


def _openai_models(status: int | None, body: Any) -> list[str] | None:
    data = body.get("data") if isinstance(body, dict) else None
    if status is None or status >= 400 or not isinstance(data, list):
        return None
    return [str(m["id"]) for m in data if isinstance(m, dict) and m.get("id")]


def _ollama_models(status: int | None, body: Any) -> list[str] | None:
    models = body.get("models") if isinstance(body, dict) else None
    if status is None or status >= 400 or not isinstance(models, list):
        return None
    return [
        str(m.get("model") or m.get("name"))
        for m in models
        if isinstance(m, dict) and (m.get("model") or m.get("name"))
    ]


def check_server(
    kind: str, base_url: str, api_key: str | None = None, model: str | None = None
) -> dict[str, Any]:
    """What a server at `base_url` is: one of five outcomes, each with a plain
    message and hint, and a better address or kind when one would work."""
    base = base_url.rstrip("/")
    root = _host_root(base)
    out: dict[str, Any] = {
        "outcome": "models",
        "message": "",
        "hint": None,
        "kind_matches": True,
        "suggest_kind": None,
        "suggest_base_url": None,
        "models": [],
        "video": None,
    }
    if kind == "ollama":
        status, body = _get(f"{base}/api/tags", None)
        models = _ollama_models(status, body)
        if models is None and status is not None:
            other = _openai_models(*_get(f"{base}/models", api_key))
            if other is None:
                other = _openai_models(*_get(f"{base}/v1/models", api_key))
                if other is not None:
                    out["suggest_base_url"] = f"{base}/v1"
            if other is not None:
                out["suggest_kind"] = "openai_compatible"
    else:
        status, body = _get(f"{base}/models", api_key)
        models = _openai_models(status, body)
        if status in (401, 403):
            out.update(
                outcome="key_refused",
                message="The server refused the key.",
                hint="Check the key, or the environment variable named for it.",
            )
            return out
        if models is None and status is not None:
            if not base.endswith("/v1"):
                v1 = _openai_models(*_get(f"{base}/v1/models", api_key))
                if v1 is not None:
                    out["suggest_base_url"] = f"{base}/v1"
        if _ollama_models(*_get(f"{root}/api/tags", None)) is not None:
            out["suggest_kind"] = "ollama"

    if status is None:
        out.update(
            outcome="unreachable",
            kind_matches=None,
            message=f"Can't reach {base}.",
            hint="Check the server is running, and that this server can reach that "
            "address: inside a container, localhost is the container itself.",
        )
        return out
    if models is None:
        named = "Ollama" if kind == "ollama" else "an OpenAI-compatible server"
        hint = None
        if out["suggest_base_url"]:
            hint = f"Try {out['suggest_base_url']}."
        elif out["suggest_kind"] == "ollama":
            hint = "It looks like Ollama: choose that kind."
        elif out["suggest_kind"] == "openai_compatible":
            hint = "It looks like an OpenAI-compatible server: choose that kind."
        out.update(
            outcome="wrong_kind",
            kind_matches=False,
            message=f"Something answers at {base}, but not as {named}.",
            hint=hint,
        )
        return out
    out["models"] = models
    if not models:
        out.update(
            outcome="no_models",
            message="Reachable, but it lists no models.",
            hint="Pull one with: ollama pull qwen3.5:9b"
            if kind == "ollama"
            else "Load a vision model in the server first.",
        )
        return out
    count = f"{len(models)} model" + ("" if len(models) == 1 else "s")
    out["message"] = f"Reachable. {count}."
    if out["suggest_kind"] == "ollama":
        out["hint"] = (
            "This is Ollama: the Ollama kind records each model's digest in your "
            "results."
        )
    if model and kind == "openai_compatible":
        from .openai_client import OpenAICompatibleVLMClient

        takes_video, _ = OpenAICompatibleVLMClient(base, 30, api_key).probe_video(model)
        out["video"] = takes_video
    return out
