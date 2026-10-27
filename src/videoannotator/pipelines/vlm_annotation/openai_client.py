"""A client for OpenAI-compatible model servers, on httpx (spec 033 R1, R2).

The contract llama.cpp's `llama-server`, vLLM, LM Studio, SGLang, Ollama's
`/v1` and hosted APIs share: `GET {base}/models` and
`POST {base}/chat/completions` with a text part and one `image_url` part per
frame. Like the Ollama client, every call is a fresh, stateless request.

The key goes only in the `Authorization` header. It is never logged, and any
server message that echoes it back has it replaced before it is kept.
"""

from __future__ import annotations

import base64
import re
import time
from typing import Any

import httpx

from .clients import KeyRefusedError, ModelServerUnavailableError, WrongKindError
from .ollama_client import VLMCallResult

# Tests put a fake server's transport here; None means the network.
DEFAULT_TRANSPORT: httpx.BaseTransport | None = None

_IMAGES_REFUSED = re.compile(r"image|multimodal|mmproj|vision", re.IGNORECASE)
# A server that doesn't know video parts may only say the content type is wrong.
_VIDEO_REFUSED = re.compile(
    r"video|content.{0,40}type|type.{0,40}content", re.IGNORECASE
)
_SCHEMA_REFUSED = re.compile(
    r"response_format|json_schema|structured|guided", re.IGNORECASE
)

IMAGES_REFUSED_MESSAGE = "This model doesn't accept images."
VIDEO_REFUSED_MESSAGE = "This model or server doesn't take video."
KEY_REFUSED_MESSAGE = "The model server refused the key."
# Internal: the server rejected `response_format`; chat() retries without it.
_SCHEMA_REFUSED_ERROR = "response_format refused"


def _data_url(kind: str, jpegs: list[bytes]) -> str:
    return f"data:{kind}/jpeg;base64," + ",".join(
        base64.b64encode(j).decode("ascii") for j in jpegs
    )


def _server_message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
        if body.get("message"):
            return str(body["message"])
        if body.get("detail"):
            return str(body["detail"])
    return response.text[:300]


class OpenAICompatibleVLMClient:
    def __init__(
        self,
        base_url: str,
        timeout: int,
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._api_key = api_key or None
        self._transport = transport if transport is not None else DEFAULT_TRANSPORT
        self._models: list[dict[str, Any]] | None = None
        # False once known that the server rejects `response_format`, so later
        # calls go straight to describing the shape in the prompt (R8).
        self.structured_output: bool | None = None
        # Whether the server takes `media_io_kwargs` for a clip's fps (R12).
        self.media_io_kwargs = False

    # ------------------------------------------------------------------

    def _scrub(self, text: str) -> str:
        if self._api_key:
            text = text.replace(self._api_key, "<redacted>")
        return text

    def _http(self, timeout: float | None = None) -> httpx.Client:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        return httpx.Client(
            headers=headers,
            timeout=timeout if timeout is not None else self._timeout,
            transport=self._transport,
        )

    def _get_models(self) -> list[dict[str, Any]]:
        url = f"{self.base_url}/models"
        try:
            with self._http(min(self._timeout, 10)) as http:
                response = http.get(url)
        except httpx.HTTPError as exc:
            raise ModelServerUnavailableError(
                f"Can't reach the model server at {self.base_url} "
                f"({type(exc).__name__})."
            ) from None
        if response.status_code in (401, 403):
            raise KeyRefusedError(KEY_REFUSED_MESSAGE)
        try:
            body = response.json()
        except ValueError:
            body = None
        data = body.get("data") if isinstance(body, dict) else None
        if response.status_code >= 400 or not isinstance(data, list):
            raise WrongKindError(
                f"{self.base_url} answers, but not as an OpenAI-compatible server "
                f"(GET /models gave {response.status_code})."
            )
        self._models = [m for m in data if isinstance(m, dict)]
        return self._models

    # ------------------------------------------------------------------

    def list_models(self) -> list[str]:
        return [str(m.get("id", "")) for m in self._get_models() if m.get("id")]

    def model_details(self, model_name: str) -> dict[str, str | None]:
        """What the server says about the model; no digest (the contract has none)."""
        details: dict[str, str | None] = {
            "digest": None,
            "quantization": None,
            "root": None,
            "owned_by": None,
        }
        try:
            models = self._models if self._models is not None else self._get_models()
        except Exception:  # provenance must not fail a run that already preflighted
            return details
        for m in models:
            if m.get("id") == model_name:
                root = m.get("root")
                details["root"] = str(root) if root else None
                owner = m.get("owned_by")
                details["owned_by"] = str(owner) if owner else None
        return details

    def preflight(self, model_name: str) -> None:
        names = self.list_models()
        if model_name not in names:
            listed = ", ".join(names) if names else "no models"
            raise ModelServerUnavailableError(
                f"Model {model_name!r} isn't on the model server at "
                f"{self.base_url}. The server lists: {listed}."
            )

    def probe_video(self, model: str) -> tuple[bool, bool]:
        """(takes video, takes a clip's fps in `media_io_kwargs`), from one call
        with a tiny clip of two grey frames: never a participant's (R12)."""
        frame = _grey_jpeg()
        for with_fps in (True, False):
            self.media_io_kwargs = with_fps
            result = self._call(
                model=model,
                prompt="Reply with one word.",
                images=[],
                options={"max_tokens": 1, "temperature": 0},
                max_retries=1,
                retry_backoff_sec=0,
                schema=None,
                video=[frame, frame],
                video_fps=2.0,
            )
            if result.error is None:
                return True, with_fps
            if result.error == VIDEO_REFUSED_MESSAGE:
                break
        self.media_io_kwargs = False
        return False, False

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
    ) -> VLMCallResult:
        """One stateless call. `think` and `keep_alive` are Ollama's and
        aren't sent. Raises KeyRefusedError on 401/403; every other failure
        comes back as a result with `error` set."""
        structured = "enforced" if schema else None
        if schema and self.structured_output is False:
            prompt, schema, structured = self._prompt_only(prompt, schema)
        while True:
            result = self._call(
                model=model,
                prompt=prompt,
                images=images,
                options=options,
                max_retries=max_retries,
                retry_backoff_sec=retry_backoff_sec,
                schema=schema,
                video=video,
                video_fps=video_fps,
            )
            if result.error == _SCHEMA_REFUSED_ERROR and schema:
                self.structured_output = False
                prompt, schema, structured = self._prompt_only(prompt, schema)
                continue
            if schema and not result.error:
                self.structured_output = True
            result.structured_output = structured
            return result

    @staticmethod
    def _prompt_only(prompt: str, schema: dict[str, Any]) -> tuple[str, None, str]:
        # Nothing hidden is added to the question (D5, P1): it already says the
        # shape in words the researcher can see.
        return prompt, None, "prompt_only"

    def _body(
        self,
        model: str,
        prompt: str,
        images: list[bytes],
        options: dict[str, Any],
        schema: dict[str, Any] | None,
        video: list[bytes] | None,
        video_fps: float | None,
    ) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content += [
            {"type": "image_url", "image_url": {"url": _data_url("image", [jpeg])}}
            for jpeg in images
        ]
        if video:
            content.append(
                {"type": "video_url", "video_url": {"url": _data_url("video", video)}}
            )
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": False,
        }
        for key in ("temperature", "top_p", "seed"):
            if options.get(key) is not None:
                body[key] = options[key]
        if options.get("num_predict"):
            body["max_tokens"] = options["num_predict"]
        if options.get("max_tokens"):
            body["max_tokens"] = options["max_tokens"]
        if schema:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema, "strict": True},
            }
        if video and video_fps and self.media_io_kwargs:
            body["media_io_kwargs"] = {"video": {"fps": video_fps}}
        return body

    def _call(
        self,
        *,
        model: str,
        prompt: str,
        images: list[bytes],
        options: dict[str, Any],
        max_retries: int,
        retry_backoff_sec: int,
        schema: dict[str, Any] | None,
        video: list[bytes] | None,
        video_fps: float | None,
    ) -> VLMCallResult:
        body = self._body(model, prompt, images, options, schema, video, video_fps)
        url = f"{self.base_url}/chat/completions"
        last_error = "unknown error"
        dt = 0.0
        for attempt in range(1, max(1, max_retries) + 1):
            t0 = time.time()
            try:
                with self._http() as http:
                    response = http.post(url, json=body)
            except httpx.ReadTimeout:
                dt = time.time() - t0
                return _failed(f"read timeout after {dt:.1f}s", dt)
            except httpx.HTTPError as exc:
                dt = time.time() - t0
                last_error = self._scrub(
                    f"can't reach the model server ({type(exc).__name__})"
                )
            else:
                dt = time.time() - t0
                status = response.status_code
                if status in (401, 403):
                    raise KeyRefusedError(KEY_REFUSED_MESSAGE)
                if status < 400:
                    return self._parse(response, dt)
                message = self._scrub(_server_message(response))
                if schema and _SCHEMA_REFUSED.search(message) and status < 500:
                    return _failed(_SCHEMA_REFUSED_ERROR, dt)
                if video and _VIDEO_REFUSED.search(message):
                    return _failed(VIDEO_REFUSED_MESSAGE, dt)
                if images and _IMAGES_REFUSED.search(message):
                    return _failed(IMAGES_REFUSED_MESSAGE, dt)
                last_error = f"the model server said {status}: {message}"
                if status != 429 and status < 500:
                    return _failed(last_error, dt)
            if attempt < max_retries:
                time.sleep(retry_backoff_sec * (2 ** (attempt - 1)))
        return _failed(last_error, dt)

    def _parse(self, response: httpx.Response, dt: float) -> VLMCallResult:
        try:
            body = response.json()
            message = body["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError):
            return _failed("the model server's reply had no answer in it", dt)
        content = message.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        reasoning = message.get("reasoning_content") or message.get("reasoning")
        usage = body.get("usage") or {}
        timings = body.get("timings") or {}
        tokens_per_sec = timings.get("predicted_per_second")
        fingerprint = body.get("system_fingerprint")
        return VLMCallResult(
            raw_text=str(content or ""),
            thinking=str(reasoning or ""),
            total_time=dt,
            load_time=None,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            resp_tokens=int(usage.get("completion_tokens") or 0),
            tokens_per_sec=float(tokens_per_sec)
            if isinstance(tokens_per_sec, int | float)
            else None,
            error=None,
            system_fingerprint=str(fingerprint) if fingerprint else None,
        )


def _grey_jpeg() -> bytes:
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.full((32, 32, 3), 128, np.uint8))
    return buf.tobytes()


def _failed(error: str, dt: float) -> VLMCallResult:
    return VLMCallResult(
        raw_text="",
        thinking="",
        total_time=dt,
        load_time=None,
        prompt_tokens=0,
        resp_tokens=0,
        tokens_per_sec=None,
        error=error,
    )
