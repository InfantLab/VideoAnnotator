# Video Annotation Pipelines and Data Spec

We will build a modular video annotation system (evolving from our our old [babyjokes github repo](https://github.com/infantlab/babyjokes)) where each processor analyzes a video segment and emits JSON labels. Key annotation pipelines include person detection and tracking, pose estimation, object recognition, scene analysis, face detection & emotion recognition, speech transcription, and speaker/voice diarization. For example, Google’s Video Intelligence API illustrates these modalities: it offers label (object) detection, shot-change and scene detection, face detection, speech transcription, text/OCR detection, object/person tracking, etc
cloud.google.com
. We will replicate and extend such capabilities using open models. Each pipeline should accept a video segment (start_time, end_time) and a “predictions per second” (PPS) parameter. If PPS ≥ FPS, output is per frame; PPS=1 yields one annotation per second; PPS=0 yields one result for the entire segment. All outputs will be in JSON for compatibility.

## Scene Segmentation
We will segment videos into fixed-length time slices (e.g. 10 seconds) or use shot boundaries if available. Each segment will be processed independently by the annotation pipelines. The segment length can be configurable, but 10 seconds is a good default for many applications. This allows parallel processing and avoids excessive memory usage.

## Person Detection & Tracking. 
Use an object-detection model (e.g. YOLOv8) to detect humans in each frame segment, then apply a tracker (e.g. DeepSORT or KCF) to assign consistent IDs across frames. Output each person’s bounding box, ID, and confidence at the requested PPS. Optionally infer roles (parent vs. infant) by comparing box sizes (e.g. “baby ≪ adult” as suggested in BabyJokes Issue #4). These labels become annotations per frame or second.
## Pose Estimation & Tracking. 
Apply a pose estimation model (OpenPose, AlphaPose, or YOLO-pose) to get 2D skeleton keypoints (joints) of each person. As Ultralytics notes, pose models output [x,y] coordinates (and confidence) for each keypoint
docs.ultralytics.com
. We store the skeleton (a list of joint coordinates) for each person per frame or time-slice. Pose tracking can be maintained by tracking the corresponding person ID. Pose JSON data include keypoints with (x,y) and optionally confidence, following COCO keypoint format or similar.
## Object Recognition. 
Run a general object detector (e.g. YOLOv8 trained on COCO) on each segment or frame. This outputs classes and bounding boxes for recognized objects (toys, phones, chairs, etc). We record each object label, its bounding box, timestamp/frame, and confidence. These annotations help describe the scene context (e.g. “toy in infant’s hand”).
## Scene/Shot Analysis. 
Optionally apply shot-change or scene classification. For example, detect shot boundaries or classify the overall environment (indoor/outdoor, room type) using a scene classifier. We record segments (with start/end times) where the scene changes or a notable event (e.g. “laughter event”) occurs. These can be coarse labels (one per segment) since PPS=0 for the segment.
## Face Detection & Emotion Recognition. 
Detect faces in each frame (e.g. with MTCNN or a face detector), then run an emotion/attribute classifier (DeepFace or similar) on each face. Store each face’s bounding box, identity (if clustering or known IDs are used), and predicted emotion (e.g. happy/neutral, etc). These face-emotion annotations are timestamped per frame or per-second as needed.
## Speech-to-Text (ASR). 
Extract the audio track and run a speech recognition model (e.g. OpenAI’s Whisper) to transcribe spoken words. Output time-aligned ## transcripts (start/end times) with text and confidence scores. This yields annotations like dialogue captions. We include metadata (language, speaker label if available).
Speaker Diarization (Voice Tracking). Perform speaker diarization (e.g. using pyannote-audio) to segment the audio by speaker turns. Align each speaker segment with the detected person (if possible via lip-sync or timing). Emit annotations indicating “Speaker 1 talks from 00:01.2s–00:03.5s, Speaker 2 from 00:03.5–00:05.0”. These allow linking voice to video person IDs.
## Audio Events (e.g. Laughter Detection). 
Optionally run audio-event detectors for specific events like baby laughter or crying. Each detected event is annotated with its time range and event type.
Each pipeline module can run on GPU (PyTorch, TensorFlow, etc.) for speed. They should follow a common interface (e.g. a Python class) taking (video_path, start_time, end_time, pps) and returning JSON.
## Data Format and Storage
All outputs will be JSON to maximize interoperability. We will define a schema per pipeline, but generally each annotation record will include the video ID, segment/frame timestamp, pipeline name, and data fields. For example, a person-detection JSON might have:
```json
{ "type": "person_bbox", "video_id": "vid123",
  "t": 12.34, "bbox": [x,y,w,h], "person_id": 1, "score": 0.87 }
```
See [Pipeline Specs](Pipeline%20Specs.md) for more details on each pipeline’s output format.

We will collect all JSONs in a standardized folder (e.g. output/) keyed by video ID and pipeline. For large-scale use, these could also be stored in a database or cloud bucket, but JSON files (or JSON-lines) suffice for now. To ensure compatibility with manual annotation tools, we will follow existing formats. For instance, CVAT’s Datumaro JSON format supports typical video annotation shapes (bounding boxes, skeletons, etc) and can include all attributes in JSON
docs.cvat.ai
. We can mirror this style (or directly export to Datumaro JSON) so that our automated annotations can be imported into CVAT or LabelStudio for review. In any case, using plain JSON (with fields like timestamps, coords, labels) makes it easy to overlay annotations on video (for captioning or drawing overlays) and to feed into ML training pipelines. All annotation records should include the segment boundaries (start/end times) or frame index. When PPS=0 (one annotation per segment), we emit one JSON record that spans the entire segment. When PPS>0, we emit records at the specified rate (e.g. once per frame or per second), each with its own timestamp.

## Project Structure
We will organize code similar to babyjokes:
`pipelines/`: Contains one module/class per annotation pipeline. E.g. person_tracker.py, pose_estimator.py, face_emotion.py, speech_recognizer.py, etc. Each module implements a common interface (e.g. a process(video_path, start, end, pps) function) and outputs JSON.
`data/`: Video files and any auxiliary data (like known labels).
`output/`: Directory for JSON annotation outputs, organized by video ID and pipeline.
`code/`: Jupyter notebooks or scripts for analysis and integration (similar to babyjokes’s notebooks).
`serve/` (optional): API or UI code if we want an interactive annotation viewer.
Environment and Dependencies: Use requirements.txt or environment.yml as in babyjokes. Key libraries will be PyTorch, YOLOv8/Ultralytics, OpenAI-Whisper, DeepFace (for emotions), OpenCV, PyAnnote (for diarization), etc (as listed in babyjokes README
github.com
). 

GPU support should be enabled for all heavy models.
Each pipeline will log its output to JSON in the output/ folder. We will write utilities to merge or aggregate these JSONs if needed (for example, combining per-frame person bboxes into track segments). Because all annotations are JSON, they can be easily loaded by ML training code or by a front-end for overlay/caption display. In summary, this spec calls for a modular JSON-centric pipeline for video annotation. Each processor handles one modality (pose, objects, faces, speech, etc.), segments the video, and emits JSON labels at a configurable rate. The structure mirrors the BabyJokes repo (with a pipelines/ directory and clear output folder) and uses standard JSON formats (e.g. Datumaro/CVAT-style) to ensure compatibility with manual annotation tools
docs.cvat.ai

All components should be documented and configurable, leveraging GPUs for efficiency.