# Pipeline Specifications

Generated: 2026-10-17T14:42:30.473268Z

This file is auto-generated. Do not edit by hand.

| Name | Display Name | Family | Variant | Tasks | Modalities | Capabilities | Backends | Stability | Outputs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| face_analysis | Face Analysis (DeepFace) | face | deepface | face-detection,emotion-recognition,age-estimation,gender-prediction | video | person-linking,frame-level-analysis | tensorflow,opencv | stable | COCO:face_detections/emotions/demographics |
| face_openface3_embedding | OpenFace3 Face Embedding | face | openface3-embedding | face-embedding | image,video | embedding | onnx,pytorch | experimental | JSON:embeddings;JSON:face_features |
| person_tracking | Person Tracking & Pose | person | yolov11n-pose-bytetrack | object-tracking,pose-estimation | video | real-time,identity-persistence | pytorch | beta | COCO:person_detection/keypoints/tracking;JSON:person_tracks |
| scene_detection | Scene Detection | scene | pyscenedetect-clip | scene-detection,scene-segmentation | video | batch,embedding | pytorch | beta | JSON:scene_boundary/scene_category |
| speaker_diarization | Speaker Diarization | audio | pyannote | speaker-diarization,speaker-segmentation | audio | timeline,speaker-turns | pytorch | stable | RTTM:speaker_turns |
| speech_recognition | Speech Recognition | audio | whisper | speech-transcription,automatic-speech-recognition | audio | streaming,word-timestamps,multilingual | pytorch | stable | WebVTT:transcript |
| vlm_annotation | VLM Frame Annotation | vlm | ollama | frame-classification,vlm-captioning | video | batch | ollama | experimental | JSON:vlm_label/vlm_reasoning |

Total pipelines: 7

## Pipeline guides

### Faces: expression, age, gender (`face_analysis`)

Finds faces in each sampled frame and estimates each face's expression, age and gender.

What to measure: Smallest face (`min_face_size`, default `30` px), Most faces per frame (`max_faces`, default `10` faces), Expression (`detect_emotions`, default `True`), Age (`detect_age`, default `True`), Gender (`detect_gender`, default `True`).

Homepage: <https://github.com/serengil/deepface>

- WIDER FACE (hard): AP 91.4% (RetinaFace, the face detector used here). [Source](https://arxiv.org/abs/1905.00641)
- DeepFace age model: MAE ± 4.65 (test data not named in the README). [Source](https://github.com/serengil/deepface)
- DeepFace gender model: accuracy 97.44% (test data not named in the README). [Source](https://github.com/serengil/deepface)

None of these figures is for infants or young children. DeepFace publishes no accuracy figure for its expression model.

Cite: `serengil2026deepface`, `deng2020retinaface`.

### Faces: action units, gaze, head pose (`face_openface3_embedding`)

Finds faces in each sampled frame and measures facial action units, gaze direction, head pose and expression. Also describes each face in a way that tells people apart.

What to measure: Smallest face (`min_face_size`, default `32` px).

Homepage: <https://github.com/CMU-MultiComp-Lab/OpenFace-3.0>

- DISFA: action unit F1 60 (OpenFace 3.0 (MTL) row, Table III). [Source](https://arxiv.org/abs/2506.02891)
- MPII: gaze, difference in angle 4.25 (OpenFace 3.0 (MTL) row, Table III). [Source](https://arxiv.org/abs/2506.02891)
- AffectNet: 8-emotion accuracy 0.56 (OpenFace 3.0 (MTL) row, Table III). [Source](https://arxiv.org/abs/2506.02891)
- 300W: landmark NME (inter-ocular) 2.87 (OpenFace 3.0 (MTL) row, Table III). [Source](https://arxiv.org/abs/2506.02891)

The paper reports a multitask model (these figures) and single-task ones; it does not say which the released package uses.

Cite: `hu2025openface3`.

### People: where they are and how they move (`person_tracking`)

Finds each person in each sampled frame, follows them through the video, and marks 17 body points (nose, shoulders, wrists, knees and so on).

What to measure: Follow each person through the video (`track_mode`, default `True`).

Homepage: <https://docs.ultralytics.com/tasks/pose/>

- COCO Keypoints val2017: mAP pose 50-95 50.0 (yolo11n-pose at 640 px; dataset as named on the Ultralytics pose docs). [Source](https://github.com/ultralytics/ultralytics/blob/v8.3.183/docs/en/macros/yolo-pose-perf.md)

Cite: `jocher2024yolo11`, `zhang2022bytetrack`.

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

Homepage: <https://huggingface.co/pyannote/speaker-diarization-3.1>

- AMI (headset mix, only_words): DER % 18.8 (no forgiveness collar, overlapped speech evaluated). [Source](https://huggingface.co/pyannote/speaker-diarization-3.1)
- DIHARD 3 (Full): DER % 21.7 (no forgiveness collar, overlapped speech evaluated). [Source](https://huggingface.co/pyannote/speaker-diarization-3.1)
- VoxConverse (v0.3): DER % 11.3 (no forgiveness collar, overlapped speech evaluated). [Source](https://huggingface.co/pyannote/speaker-diarization-3.1)

Cite: `plaquet2023powerset`, `bredin2023pyannote`.

### Speech: what was said (`speech_recognition`)

Transcribes speech into timed text, with the time of each word.

What to measure: Language (`language`, default `None`), Transcribe or translate (`task`, default `transcribe`), Time of each word (`word_timestamps`, default `True`), Shortest segment (`min_segment_duration`, default `1.0` s).

Homepage: <https://github.com/openai/whisper>

- LibriSpeech test-clean: WER % 5.0 (Whisper base, greedy decoding, Table 8). [Source](https://arxiv.org/abs/2212.04356)
- LibriSpeech test-other: WER % 12.4 (Whisper base, greedy decoding, Table 8). [Source](https://arxiv.org/abs/2212.04356)

LibriSpeech is English read aloud by adults; none of these figures measures child-directed speech, infant vocalisations or overlapping talk.

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
