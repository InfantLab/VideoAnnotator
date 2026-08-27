"""API tests for spec 009: VLM prompt preview and model discovery.

Follows tests/integration/test_job_cancellation.py's lightweight pattern
(bare TestClient(app) + per-test reset_storage_backend()) since preview/
models don't touch job storage at all -- only the video-frame-extraction
path (video_path + timestamp_sec) needs a real job's uploaded video on disk.
"""

import io
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from videoannotator.api.database import reset_storage_backend
from videoannotator.api.main import app
from videoannotator.pipelines.vlm_annotation.ollama_client import (
    OllamaUnavailableError,
    VLMCallResult,
)

client = TestClient(app)

_OLLAMA_CLIENT = "videoannotator.api.v1.vlm.OllamaVLMClient"


@pytest.fixture(autouse=True)
def reset_db():
    reset_storage_backend()
    yield
    reset_storage_backend()


def _upload_video() -> str:
    """Submit a job purely to get a real, server-side video_path to extract
    frames from -- mirrors how a real client would reference an
    already-uploaded video."""
    from tests.fixtures.synthetic_video import synthetic_video_bytes_avi

    response = client.post(
        "/api/v1/jobs/",
        files={
            "video": (
                "test.avi",
                io.BytesIO(synthetic_video_bytes_avi()),
                "video/avi",
            )
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["video_path"]


class TestListVlmModels:
    def test_reachable_server_returns_models(self):
        """US2 acceptance scenario 1."""
        with patch(f"{_OLLAMA_CLIENT}.list_models", return_value=["qwen3.5:9b"]):
            resp = client.get("/api/v1/vlm/models")
        assert resp.status_code == 200
        body = resp.json()
        assert body["models"] == ["qwen3.5:9b"]
        assert "base_url" in body

    def test_unreachable_server_returns_distinct_error(self):
        """US2 acceptance scenario 2 / FR-005."""
        with patch(
            f"{_OLLAMA_CLIENT}.list_models",
            side_effect=OllamaUnavailableError("Cannot reach ollama server"),
        ):
            resp = client.get("/api/v1/vlm/models")
        assert resp.status_code == 503
        assert resp.json()["error"]["code"] == "OLLAMA_UNREACHABLE"

    def test_reachable_but_zero_models_is_not_an_error(self):
        """Edge case: empty list vs. connection error must be distinguishable."""
        with patch(f"{_OLLAMA_CLIENT}.list_models", return_value=[]):
            resp = client.get("/api/v1/vlm/models")
        assert resp.status_code == 200
        assert resp.json()["models"] == []


class TestVlmPreview:
    def test_preview_with_uploaded_image_returns_label_and_reasoning(self):
        """US1 acceptance scenario 1."""
        fake_result = VLMCallResult(
            raw_text="TOUCH",
            thinking="the hand is touching",
            total_time=1.2,
            load_time=0.1,
            prompt_tokens=50,
            resp_tokens=5,
            tokens_per_sec=41.6,
        )
        with patch(f"{_OLLAMA_CLIENT}.chat", return_value=fake_result):
            resp = client.post(
                "/api/v1/vlm/preview",
                files={"image": ("frame.jpg", io.BytesIO(b"fake jpeg"), "image/jpeg")},
                data={"prompt": "Is there touch?", "model": "qwen3.5:9b"},
            )
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] == "TOUCH"
        assert body["reasoning"] == "the hand is touching"
        assert body["total_time"] == 1.2

    def test_preview_creates_no_job_or_annotation_record(self):
        """US1 acceptance scenario 1 / SC-001: zero side effects."""
        fake_result = VLMCallResult(
            raw_text="NO_TOUCH",
            thinking="",
            total_time=0.5,
            load_time=0.0,
            prompt_tokens=10,
            resp_tokens=2,
            tokens_per_sec=4.0,
        )
        before = client.get("/api/v1/jobs/").json()["total"]
        with patch(f"{_OLLAMA_CLIENT}.chat", return_value=fake_result):
            client.post(
                "/api/v1/vlm/preview",
                files={"image": ("frame.jpg", io.BytesIO(b"fake jpeg"), "image/jpeg")},
                data={"prompt": "p", "model": "m"},
            )
        after = client.get("/api/v1/jobs/").json()["total"]
        assert after == before

    def test_preview_by_video_path_and_timestamp_extracts_frame_itself(self):
        """US1 acceptance scenario 2."""
        video_path = _upload_video()
        fake_result = VLMCallResult(
            raw_text="TOUCH",
            thinking="",
            total_time=0.8,
            load_time=0.0,
            prompt_tokens=10,
            resp_tokens=2,
            tokens_per_sec=2.5,
        )
        with patch(f"{_OLLAMA_CLIENT}.chat", return_value=fake_result) as mock_chat:
            resp = client.post(
                "/api/v1/vlm/preview",
                data={
                    "video_path": video_path,
                    "timestamp_sec": "0.1",
                    "prompt": "p",
                    "model": "m",
                },
            )
        assert resp.status_code == 200, resp.text
        assert resp.json()["label"] == "TOUCH"
        # The extracted frame(s) were actually sent to the model.
        assert len(mock_chat.call_args.kwargs["images"]) >= 1

    def test_frame_burst_samples_the_real_pipelines_burst_window(self):
        """US1 acceptance scenario 3: reuses the pipeline's own burst
        sampling, sending all frames in one call."""
        video_path = _upload_video()
        fake_result = VLMCallResult(
            raw_text="TOUCH",
            thinking="",
            total_time=0.8,
            load_time=0.0,
            prompt_tokens=10,
            resp_tokens=2,
            tokens_per_sec=2.5,
        )
        with patch(f"{_OLLAMA_CLIENT}.chat", return_value=fake_result) as mock_chat:
            resp = client.post(
                "/api/v1/vlm/preview",
                data={
                    "video_path": video_path,
                    "timestamp_sec": "0.5",
                    "sampling_mode": "frame_burst",
                    "frame_interval_sec": "0.1",
                    "burst_offsets": "[-1, 0, 1]",
                    "prompt": "p",
                    "model": "m",
                },
            )
        assert resp.status_code == 200, resp.text
        # One single call carrying every burst frame, not one call per frame.
        assert mock_chat.call_count == 1
        assert len(mock_chat.call_args.kwargs["images"]) >= 1

    def test_frame_burst_with_uploaded_image_is_rejected(self):
        """Burst sampling needs a video reference, not a single still image."""
        resp = client.post(
            "/api/v1/vlm/preview",
            files={"image": ("frame.jpg", io.BytesIO(b"fake jpeg"), "image/jpeg")},
            data={"sampling_mode": "frame_burst", "prompt": "p", "model": "m"},
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "BURST_REQUIRES_VIDEO_REFERENCE"

    def test_timestamp_beyond_video_duration_is_a_clear_error_not_a_crash(self):
        """Edge case."""
        video_path = _upload_video()
        resp = client.post(
            "/api/v1/vlm/preview",
            data={
                "video_path": video_path,
                "timestamp_sec": "999999",
                "prompt": "p",
                "model": "m",
            },
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "TIMESTAMP_OUT_OF_RANGE"

    def test_neither_image_nor_video_reference_is_rejected(self):
        resp = client.post("/api/v1/vlm/preview", data={"prompt": "p", "model": "m"})
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "INVALID_PREVIEW_SOURCE"

    def test_both_image_and_video_reference_is_rejected(self):
        resp = client.post(
            "/api/v1/vlm/preview",
            files={"image": ("frame.jpg", io.BytesIO(b"fake jpeg"), "image/jpeg")},
            data={
                "video_path": "/some/path.mp4",
                "timestamp_sec": "1.0",
                "prompt": "p",
                "model": "m",
            },
        )
        assert resp.status_code == 422
        assert resp.json()["error"]["code"] == "INVALID_PREVIEW_SOURCE"

    def test_ollama_call_failure_returns_clear_error(self):
        fake_result = VLMCallResult(
            raw_text="",
            thinking="",
            total_time=5.0,
            load_time=0.0,
            prompt_tokens=0,
            resp_tokens=0,
            tokens_per_sec=0.0,
            error="model not found",
        )
        with patch(f"{_OLLAMA_CLIENT}.chat", return_value=fake_result):
            resp = client.post(
                "/api/v1/vlm/preview",
                files={"image": ("frame.jpg", io.BytesIO(b"fake jpeg"), "image/jpeg")},
                data={"prompt": "p", "model": "nonexistent-model"},
            )
        assert resp.status_code == 502
        assert resp.json()["error"]["code"] == "VLM_CALL_FAILED"

    def test_uses_the_same_label_parser_as_the_real_pipeline(self):
        """SC-002: shared-code-path, not merely similar output."""
        from videoannotator.pipelines.vlm_annotation.vlm_pipeline import _parse_label

        fake_result = VLMCallResult(
            raw_text="  the answer is: no_touch, clearly  ",
            thinking="",
            total_time=0.1,
            load_time=0.0,
            prompt_tokens=1,
            resp_tokens=1,
            tokens_per_sec=1.0,
        )
        with patch(f"{_OLLAMA_CLIENT}.chat", return_value=fake_result):
            resp = client.post(
                "/api/v1/vlm/preview",
                files={"image": ("frame.jpg", io.BytesIO(b"fake jpeg"), "image/jpeg")},
                data={"prompt": "p", "model": "m"},
            )
        assert resp.json()["label"] == _parse_label(fake_result.raw_text)
