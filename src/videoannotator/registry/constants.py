"""Controlled vocabularies for registry metadata.

These are intentionally small for v1.2.1 and can expand in later
versions. Validation is currently soft (tests may reference these); the
loader does not yet reject unknown values to preserve flexibility while
iterating.
"""

from __future__ import annotations

TASKS = {
    "age-estimation",
    "audio-analysis",
    "automatic-speech-recognition",
    "gender-prediction",
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
    "speaker-segmentation",
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
    "empathic-analysis",
    "frame-level-analysis",
    "liveness",
    "multilingual",
    "anonymization",
    "identity-persistence",
    "person-linking",
    "speaker-turns",
    "timeline",
    "word-timestamps",
}

MODALITIES = {"video", "image", "audio", "multimodal", "sensor-lidar", "sensor-radar"}

BACKENDS = {
    "pytorch",
    "tensorflow",
    "huggingface",
    "onnx",
    "opencv",
    "tensorrt",
    "openvino",
    "cpu",
    "cuda",
}

STABILITY = {"experimental", "beta", "stable", "deprecated"}

__all__ = [
    "BACKENDS",
    "CAPABILITIES",
    "MODALITIES",
    "STABILITY",
    "TASKS",
]
