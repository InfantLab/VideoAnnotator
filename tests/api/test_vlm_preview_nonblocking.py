"""A preview waiting on the model doesn't stop the server answering others."""

import threading
import time
from unittest.mock import patch

from fastapi.testclient import TestClient

from tests.fixtures.synthetic_video import synthetic_video_bytes_avi
from videoannotator.api.main import app
from videoannotator.pipelines.vlm_annotation.ollama_client import VLMCallResult


def test_other_requests_are_served_while_a_preview_waits():
    # One client for both requests: a bare TestClient gives each request its
    # own event loop, which would hide a blocked one.
    with TestClient(app) as client:
        _check(client)


def _check(client):
    image = synthetic_video_bytes_avi()  # any bytes: the fake model never decodes
    in_call = threading.Event()
    release = threading.Event()

    def slow_chat(*args, **kwargs):
        in_call.set()
        release.wait(10)
        return VLMCallResult("TOUCH", "", 1.0, 0.1, 10, 1, 1.0)

    results: dict[str, int] = {}

    def preview():
        r = client.post(
            "/api/v1/vlm/preview",
            data={"model": "qwen3.5:9b", "prompt": "Touch?"},
            files={"image": ("f.jpg", image, "image/jpeg")},
        )
        results["preview"] = r.status_code

    with (
        patch("videoannotator.api.v1.vlm.OllamaVLMClient.__init__", return_value=None),
        patch("videoannotator.api.v1.vlm.OllamaVLMClient.chat", side_effect=slow_chat),
    ):
        worker = threading.Thread(target=preview)
        worker.start()
        try:
            assert in_call.wait(10)
            started = time.monotonic()
            health = client.get("/api/v1/system/health")
            elapsed = time.monotonic() - started
            assert health.status_code == 200
            assert not release.is_set()
            assert elapsed < 5
        finally:
            release.set()
            worker.join(15)
    assert results["preview"] == 200
