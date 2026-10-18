# Pipeline Specifications

Generated: 2026-10-18T13:55:20.787330Z

This file is auto-generated. Do not edit by hand.

| Name | Display Name | Family | Variant | Tasks | Modalities | Capabilities | Backends | Stability | Outputs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| face_analysis | Face Analysis (DeepFace) | face | deepface | face-detection,emotion-recognition,age-estimation,gender-prediction | video | person-linking,frame-level-analysis | tensorflow,opencv | stable | COCO:face_detections/emotions/demographics |
| face_openface3_embedding | OpenFace3 Face Embedding | face | openface3-embedding | face-embedding | image,video | embedding | onnx,pytorch | experimental | JSON:embeddings;JSON:face_features |
| person_tracking | Person Tracking & Pose | person | yolo26n-pose-bytetrack | object-tracking,pose-estimation | video | real-time,identity-persistence | pytorch | beta | COCO:person_detection/keypoints/tracking;JSON:person_tracks |
| scene_detection | Scene Detection | scene | pyscenedetect-clip | scene-detection,scene-segmentation | video | batch,embedding | pytorch | beta | JSON:scene_boundary/scene_category |
| speaker_diarization | Speaker Diarization | audio | pyannote | speaker-diarization,speaker-segmentation | audio | timeline,speaker-turns | pytorch | stable | RTTM:speaker_turns |
| speech_recognition | Speech Recognition | audio | faster-whisper | speech-transcription,automatic-speech-recognition | audio | streaming,word-timestamps,multilingual | ctranslate2 | stable | WebVTT:transcript |
| vlm_annotation | VLM Frame Annotation | vlm | ollama | frame-classification,vlm-captioning | video | batch | ollama | experimental | JSON:vlm_label/vlm_reasoning |

Total pipelines: 7

## Pipeline guides

### Faces: where they are and their expression (`face_analysis`)

Finds faces in each sampled frame and estimates each face's expression. Can also estimate age and gender, which are off by default.

What to measure: Smallest face (`min_face_size`, default `30` px), Most faces per frame (`max_faces`, default `10` faces), Expression (`detect_emotions`, default `True`), Age (`detect_age`, default `False`), Gender (`detect_gender`, default `False`).

Homepage: <https://github.com/serengil/deepface>

- WIDER FACE (hard): AP 91.4% (RetinaFace, the face detector used here). [Source](https://arxiv.org/abs/1905.00641)
- DeepFace age model: MAE ± 4.65 (test data not named in the README). [Source](https://github.com/serengil/deepface)
- DeepFace gender model: accuracy 97.44% (test data not named in the README). [Source](https://github.com/serengil/deepface)

None of these figures is for infants or young children. DeepFace's expression model was trained on adult faces, and DeepFace publishes no accuracy figure for it.

Cite: `serengil2026deepface`, `deng2020retinaface`.

### Faces: action units, gaze, head pose (`face_openface3_embedding`)

Finds faces in each sampled frame and measures facial action units, gaze direction, head pose and expression. Also describes each face in a way that tells people apart.

What to measure: Smallest face (`min_face_size`, default `32` px).

Homepage: <https://github.com/CMU-MultiComp-Lab/OpenFace-3.0>

- DISFA: action unit F1 60 (OpenFace 3.0 (MTL) row, Table III). [Source](https://arxiv.org/abs/2506.02891)
- MPII: gaze, difference in angle 4.25 (OpenFace 3.0 (MTL) row, Table III). [Source](https://arxiv.org/abs/2506.02891)
- AffectNet: 8-emotion accuracy 0.56 (OpenFace 3.0 (MTL) row, Table III). [Source](https://arxiv.org/abs/2506.02891)
- 300W: landmark NME (inter-ocular) 2.87 (OpenFace 3.0 (MTL) row, Table III). [Source](https://arxiv.org/abs/2506.02891)

The paper reports a multitask model (these figures) and single-task ones; it does not say which the released package uses. Installed from `openface-test`, the package CMU's repository names; neither has changed since June 2025. Its licence allows non-commercial research use only.

Cite: `hu2025openface3`.

### People: where they are and how they move (`person_tracking`)

Finds each person in each sampled frame, follows them through the video, and marks 17 body points (nose, shoulders, wrists, knees and so on).

What to measure: Follow each person through the video (`track_mode`, default `True`).

Homepage: <https://docs.ultralytics.com/tasks/pose/>

- COCO Keypoints val2017: mAP pose 50-95 (e2e) 57.2 (yolo26n-pose at 640 px, end-to-end (no NMS); yolo11n-pose, the previous default, 50.0). [Source](https://docs.ultralytics.com/models/yolo26/)

Cite: `jocher2026yolo26`, `zhang2022bytetrack`.

### Scenes: cuts and settings (`scene_detection`)

Finds where the video cuts from one shot to another and labels the kind of setting each scene shows.

What to measure: Shortest scene (`min_scene_length`, default `1.0` s).

Homepage: <https://www.scenedetect.com/>

No published benchmark.

Neither PySceneDetect nor CLIP publishes an accuracy figure for finding cuts or labelling scene settings.

Cite: `castellano_pyscenedetect`, `radford2021clip`, `ilharco2021openclip`, `schuhmann2022laion`.

### Speaker turns: who spoke when (`speaker_diarization`)

Splits the audio into turns and says which speaker each turn belongs to, without naming them.

What to measure: Fewest speakers (`min_speakers`, default `1` speakers), Most speakers (`max_speakers`, default `10` speakers).

Homepage: <https://huggingface.co/pyannote/speaker-diarization-community-1>

- AMI (IHM): DER % 17.0 (community-1; benchmark last updated 2025-09). [Source](https://github.com/pyannote/pyannote-audio#benchmark)
- DIHARD 3 (full): DER % 20.2 (community-1; benchmark last updated 2025-09). [Source](https://github.com/pyannote/pyannote-audio#benchmark)
- VoxConverse (v0.3): DER % 11.2 (community-1; benchmark last updated 2025-09). [Source](https://github.com/pyannote/pyannote-audio#benchmark)

These sets are meetings, broadcasts and phone calls between adults; none measures parent–infant interaction or infant vocalisations.

Cite: `plaquet2023powerset`, `pyannote2025community1`, `bredin2023pyannote`.

### Speech: what was said (`speech_recognition`)

Transcribes speech into timed text, with the time of each word.

What to measure: Language (`language`, default `None`), Transcribe or translate (`task`, default `transcribe`), Time of each word (`word_timestamps`, default `True`), Shortest segment (`min_segment_duration`, default `1.0` s).

Homepage: <https://github.com/SYSTRAN/faster-whisper>

- Transcribing 13 minutes of audio (GPU, large-v2, beam 5): time 1m03s (openai-whisper 2m23s) (faster-whisper fp16 against openai-whisper fp16, on an RTX 3070 Ti). [Source](https://github.com/SYSTRAN/faster-whisper#benchmark)

OpenAI publishes large-v3-turbo's accuracy only as charts: "Across languages, the turbo model performs similarly to large-v2", with larger drops for some languages such as Thai and Cantonese (github.com/openai/whisper/discussions/2363). No figure measures child-directed speech, infant vocalisations or overlapping talk. On a CPU, a smaller model (whisper_model small) is much faster.

Cite: `radford2023whisper`.

### Ask AI: your question about each moment (`vlm_annotation`)

Asks a vision-language model your question about frames sampled every few seconds, and records its answer and reasoning for each one.

What to measure: Question (`prompt`, default `Classify as TOUCH only if:
The mother's hand or arm is actually in physical contact with the infant's body.

Classify as NO_TOUCH if:
The mother's hand or arm visually overlaps the infant in the image, but one is clearly in front of or behind the other (the apparent contact is camera perspective, not real contact).
The mother's hand or arm is near but not actually touching the infant.

Return ONLY: TOUCH or NO_TOUCH.
`), Frames per sample (`sampling_mode`, default `single_frame`), Sample every (`frame_interval_sec`, default `5.0` s).

Homepage: <https://ollama.com/>

No published benchmark.

Depends on the model you choose and on your question, so no figure applies to this pipeline as such. See your model's page, for example https://ollama.com/library/qwen3.5.

Cite: `ollama`.
