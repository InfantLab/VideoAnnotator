# 🎬 Scene & Person Tracking Module Specification

## Overview

This module performs automated **scene segmentation**, **person detection**, **multi-object tracking**, and **scene-level classification** using **Ultralytics YOLO11**. Compared to previous designs, YOLO11 supports unified detection, pose estimation, and tracking within one model family—reducing dependency overhead and improving performance.

---

## 🔁 Pipeline Summary (with YOLO11)

| Stage                       | Tool/Model                                                      | Output                                          |
| --------------------------- | --------------------------------------------------------------- | ----------------------------------------------- |
| 1. Scene Segmentation       | `PySceneDetect`                                                 | Shot boundaries (start/end timestamps)          |
| 2. Scene Classification     | `CLIP` / `ImageBind`                                            | Scene labels (e.g. “living room”)               |
| 3. Person Detection & Pose  | `YOLO11-pose`                                                   | Bounding boxes, skeleton keypoints, confidences |
| 4. Object Tracking          | `YOLO11-pose` with `track` mode (ByteTrack/BoT-SORT integrated) | Consistent `person_id`s across frames           |
| 5. Audio Context (optional) | `CLAP` / `VGGish`                                               | Audio tags per scene (e.g. “speech”)            |

---

## 🧱 Module Components

### `scene_splitter/`

- Input: Raw video
- Tool: `PySceneDetect`
- Output: List of `{start_time, end_time}` for each scene

### `scene_classifier/`

- Input: Keyframes from each scene
- Tool: `CLIP` or `ImageBind`
- Output: Semantic label per scene with confidence

### `person_pose_and_track/`

- Input: Full video or scenes
- Model: `YOLO11-pose`
- Mode: Run in `track` mode to get detection + tracking in one pass
- Output JSON:

```json
{
  "type": "person_skeleton",
  "video_id": "vid123",
  "t": 12.34,
  "person_id": 1,
  "bbox": [x, y, w, h],
  "keypoints": [
    { "joint": "nose", "x": 123, "y": 456, "conf": 0.98 },
    ...
  ]
}
```

---

## 🔧 Configuration Parameters

```yaml
scene_detect:
  threshold: 30
  min_scene_length: 2.0

clip:
  prompts: ["living room", "clinic", "outdoor", "nursery"]

tracking:
  model: yolo26n-pose
  track_mode: true
  conf_thresh: 0.4
```

---

## 📦 Output JSON Format

Each scene produces:

```json
{
  "scene_id": "scene_02",
  "start": 15.2,
  "end": 32.4,
  "label": "living room",
  "people": [
    {
      "person_id": 1,
      "entry": 15.3,
      "exit": 31.2,
      "trajectory": [
        { "t": 15.3, "bbox": [x1, y1, x2, y2], "keypoints": {...} },
        ...
      ]
    }
  ],
  "audio_tags": ["speech"]
}
```

---

## 🚀 Advantages of YOLO11

- **Unified model support**: Pose estimation, detection, tracking, and optional segmentation via `yolo11-pose` or `yolo11-seg` models in a single package([docs.ultralytics.com][1], [docs.ultralytics.com][2], [medium.com][3], [ultralytics.com][4])
- **Track mode built-in**: Supports object tracking out-of-the-box via `track` mode using ByteTrack or BoT-SORT, removing extra dependencies like DeepSORT
- **Efficient and accurate**: Offers high mAP with fewer parameters than YOLOv8 (e.g. same or better performance with lower compute overhead)([docs.ultralytics.com][1])

---

## 🗂 Project Structure Example

```
/scene_person_module/
├── scene_splitter/
├── scene_classifier/
├── person_pose_and_track/
├── outputs/
│   ├── scenes/
│   ├── tracking/
│   └── merged_annotations.json
├── config.yaml
└── run_pipeline.py
```

---

## 🎯 Summary

Leveraging **Ultralytics YOLO11**, this module simplifies the stack by combining detection, pose, and tracking features in one open-source model. It provides robust, GPU-accelerated annotations and scene context in JSON format, aligning well with your annotation and viewing workflows.

[1]: https://docs.ultralytics.com/models/yolo11/?utm_source=chatgpt.com "Ultralytics YOLO11"
[2]: https://docs.ultralytics.com/tasks/pose/?utm_source=chatgpt.com "Pose Estimation - Ultralytics YOLO Docs"
[3]: https://medium.com/%40beam_villa/object-tracking-made-easy-with-yolov11-bytetrack-73aac16a9f4a?utm_source=chatgpt.com "Object Tracking Made Easy with YOLOv11 + ByteTrack - Medium"
[4]: https://www.ultralytics.com/blog/a-guide-on-tracking-moving-objects-in-videos-with-ultralytics-yolo-models?utm_source=chatgpt.com "A Guide on Tracking Moving Objects in Videos with Ultralytics YOLO ..."
