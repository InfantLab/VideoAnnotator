"""Controlled vocabularies for registry metadata.

These are intentionally small for v1.2.1 and can expand in later
versions. Validation is currently soft (tests may reference these); the
loader does not yet reject unknown values to preserve flexibility while
iterating.
"""

from __future__ import annotations

TASKS = {
    "object-detection",
    "object-tracking",
    "pose-estimation",
    "face-detection",
    "face-embedding",
    "face-recognition",
    "emotion-recognition",
    "scene-detection",
    "scene-segmentation",
    "action-recognition",
    "speech-transcription",
    "speaker-diarization",
    "audio-event-detection",
    "text-detection",
    "ocr",
    "content-moderation",
    "person-reidentification",
    "interaction-analysis",
}

CAPABILITIES = {
    "zero-shot",
    "few-shot",
    "multi-modal-fusion",
    "real-time",
    "batch",
    "streaming",
    "auto-labeling",
    "embedding",
    "liveness",
    "anonymization",
    "identity-persistence",
}

MODALITIES = {"video", "image", "audio", "multimodal", "sensor-lidar", "sensor-radar"}

BACKENDS = {"pytorch", "onnx", "tensorrt", "openvino", "cpu", "cuda"}

STABILITY = {"experimental", "beta", "stable", "deprecated"}

__all__ = [
    "BACKENDS",
    "CAPABILITIES",
    "MODALITIES",
    "STABILITY",
    "TASKS",
]
