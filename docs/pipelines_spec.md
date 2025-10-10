# Pipeline Specifications

Generated: 2025-09-17T19:59:09.212864Z

This file is auto-generated. Do not edit by hand.

| Name                     | Display Name                             | Family | Variant                  | Tasks                                               | Modalities  | Capabilities                   | Backends     | Stability    | Outputs                                  |
| ------------------------ | ---------------------------------------- | ------ | ------------------------ | --------------------------------------------------- | ----------- | ------------------------------ | ------------ | ------------ | ---------------------------------------- |
| audio_processing         | Audio Processing (Speech + Diarization)  | audio  | whisper-pyannote         | speech-transcription,speaker-diarization            | audio       | streaming,embedding            | pytorch      | beta         | WebVTT:transcript;RTTM:speaker_turns     |
| face_laion_clip          | LAION CLIP Face Semantic Embedding       | face   | laion-clip-face          | face-embedding,face-recognition,emotion-recognition | image,video | zero-shot,embedding,real-time  | pytorch      | experimental | JSON:embeddings/attributes               |
| face_openface3_embedding | OpenFace3 Face Embedding                 | face   | openface3-embedding      | face-embedding                                      | image,video | embedding                      | onnx,pytorch | experimental | JSON:embeddings                          |
| person_tracking          | Person Tracking & Pose                   | person | yolov11n-pose-bytetrack  | object-tracking,pose-estimation                     | video       | real-time,identity-persistence | pytorch      | beta         | COCO:person_detection/keypoints/tracking |
| scene_detection          | Scene Detection                          | scene  | pyscenedetect-clip       | scene-detection,scene-segmentation                  | video       | batch,embedding                | pytorch      | beta         | JSON:scene_boundary/scene_category       |
| voice_emotion_baseline   | Voice Emotion + Transcription (Baseline) | audio  | whisper-spectral-emotion | speech-transcription,emotion-recognition            | audio       | streaming,embedding            | pytorch      | experimental | WebVTT:transcript;JSON:emotion_segments  |

Total pipelines: 6
