# Auto-PCI Video Annotation Pipeline Specification

This document outlines the architecture, data formats, and implementation strategy for the automated video annotation component of Auto-PCI, based on the [babyjokes](https://github.com/InfantLab/babyjokes) project structure.

## 🧠 Overview

We will build a modular, JSON-centric annotation system to extract interpretable labels from short videos of parent–child interactions. The pipeline should support both:
- **ML training** use cases (structured outputs per frame or per second)
- **Human interpretation** (overlay/caption display in UI or export to annotation tools)

## 🎯 Design Goals

- Modular processors, each handling one type of annotation (pose, speech, objects…)
- Standardized JSON output for all pipelines
- Configurable segmentation via `start_time`, `end_time`, and `pps` (predictions per second)
- GPU-enabled processing where available
- Outputs interoperable with manual annotation tools (CVAT, ELAN, LabelStudio)

---

## 📦 Project Structure (Based on `babyjokes`)

```plaintext
project_root/
├── data/ # Raw videos and metadata
├── pipelines/ # One module per annotation type
│ ├── person_tracker.py
│ ├── pose_estimator.py
│ ├── face_emotion.py
│ └── ...
├── output/ # JSON annotation outputs
│ └── vid123/
│ ├── person_bbox.json
│ ├── pose_keypoints.json
│ └── ...
├── code/ # Analysis, training, demo notebooks
├── requirements.txt # Dependencies
└── README.md
```

## 🧩 Annotation Modules

Each module in `pipelines/` will follow this interface:

```python
def process(video_path: str, start_time: float, end_time: float, pps: float) -> List[dict]:
    ...
```

All annotations are returned and optionally saved as structured JSON records.

1. Person Detection & Tracking
Model: YOLOv8 + DeepSORT

Output JSON:

```json
{
  "type": "person_bbox",
  "video_id": "vid123",
  "t": 12.34,
  "bbox": [x, y, w, h],
  "person_id": 1,
  "score": 0.87
}
```


2. Pose Estimation & Tracking
Model: YOLO-pose or AlphaPose

Format: COCO-style keypoints per frame

Output JSON:

```json
{
  "type": "pose_keypoints",
  "video_id": "vid123",
  "t": 12.34,
  "person_id": 1,
  "keypoints": [
    { "joint": "nose", "x": 123, "y": 456, "conf": 0.98 },
    ...
  ]
}
```

3. Object Recognition
Model: YOLOv8 (COCO classes)

Output JSON::

```json
{
  "type": "object",
  "video_id": "vid123",
  "t": 12.34,
  "label": "toy",
  "bbox": [x, y, w, h],
  "score": 0.91
}
```

4. Scene & Shot Classification
Function: Detect shot changes, classify environment (indoor/outdoor)

PPS: 0 (per segment only)

Output JSON:

```json
{
  "type": "scene_label",
  "video_id": "vid123",
  "start": 0,
  "end": 20,
  "label": "indoor-living-room",
  "confidence": 0.88
}
```

5. Face Detection & Emotion Recognition
Model: MTCNN + DeepFace

Output JSON:

```json
{
  "type": "facial_emotion",
  "video_id": "vid123",
  "t": 12.34,
  "person_id": 1,
  "bbox": [x, y, w, h],
  "emotion": "happy",
  "confidence": 0.91
}
```

6. Speech Recognition (ASR)
Model: Whisper
Output JSON:

Model: Whisper

Output JSON:

```json
{
  "type": "transcript",
  "video_id": "vid123",
  "start": 12.0,
  "end": 14.2,
  "text": "Hello baby",
  "confidence": 0.92
}
```


7. Speaker Diarization
Model: pyannote-audio

Output JSON:

```json
{
  "type": "speaker_turn",
  "video_id": "vid123",
  "start": 12.0,
  "end": 14.2,
  "speaker": "spk_01"
}
```

8. Audio Event Detection (e.g., laughter, crying)
Optional: Specialized classifiers

Output JSON:

```json
{
  "type": "audio_event",
  "video_id": "vid123",
  "start": 7.0,
  "end": 8.4,
  "event": "baby_laugh",
  "confidence": 0.88
}
```

## 🗂️ Output & File Conventions
All pipeline outputs are stored in output/{video_id}/{pipeline}.json

Each file is a list of time-stamped JSON records

JSON-Lines format (.jsonl) allowed if needed for streaming

## 📡 Config Interface
Each processor supports:

--video_path

--start_time

--end_time

--pps (0 = per segment, 1 = per second, ≥ FPS = per frame)

--output_dir

## 🔗 Interoperability
JSON annotations compatible with:

CVAT / Datumaro JSON format

ELAN via exported EAF XML

LabelStudio and VIA with minor transformation

## 🚀 Hardware & Dependencies
GPU-enabled PyTorch or TensorFlow pipelines

Key dependencies:

ultralytics (YOLOv8)

openai-whisper

pyannote-audio

deepface

opencv-python

ffmpeg, librosa, torchaudio

## 📌 Summary
This video annotation framework is designed to be:

Modular, maintainable and extensible

Fully JSON-based and compatible with downstream ML and human-in-the-loop tasks

Based on modern open-source libraries and accelerates research on parent–child interaction at scale

## 📝 Conclusion
This specification provides a high-level overview of the architecture and data formats for the Auto-PCI video annotation pipeline. Each module can be developed independently, allowing for easy updates and integration of new models or features as they become available.