# Pipeline Specifications

Generated: 2025-10-31T16:38:14.268275Z

This file is auto-generated. Do not edit by hand.

| Name | Display Name | Family | Variant | Tasks | Modalities | Capabilities | Backends | Stability | Outputs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| audio_processing | Audio Processing (Speech + Diarization) | audio | whisper-pyannote | speech-transcription,speaker-diarization | audio | streaming,embedding | pytorch | beta | WebVTT:transcript;RTTM:speaker_turns |
| face_analysis | Face Analysis (DeepFace) | face | deepface | face-detection,emotion-recognition,age-estimation,gender-prediction | video | person-linking,frame-level-analysis | tensorflow,opencv | stable | COCO:face_detections/emotions/demographics |
| face_laion_clip | LAION CLIP Face Semantic Embedding | face | laion-clip-face | face-embedding,face-recognition,emotion-recognition | image,video | zero-shot,embedding,real-time | pytorch | experimental | JSON:embeddings/attributes |
| face_openface3_embedding | OpenFace3 Face Embedding | face | openface3-embedding | face-embedding | image,video | embedding | onnx,pytorch | experimental | JSON:embeddings |
| laion_voice | LAION Empathic Voice Analysis | audio | laion-empathic-whisper | emotion-recognition,audio-analysis | audio | embedding,empathic-analysis | pytorch,huggingface | stable | JSON:emotion_segments/empathic_scores;WebVTT:emotion_timeline |
| person_tracking | Person Tracking & Pose | person | yolov11n-pose-bytetrack | object-tracking,pose-estimation | video | real-time,identity-persistence | pytorch | beta | COCO:person_detection/keypoints/tracking |
| scene_detection | Scene Detection | scene | pyscenedetect-clip | scene-detection,scene-segmentation | video | batch,embedding | pytorch | beta | JSON:scene_boundary/scene_category |
| speaker_diarization | Speaker Diarization | audio | pyannote | speaker-diarization,speaker-segmentation | audio | timeline,speaker-turns | pytorch | stable | RTTM:speaker_turns |
| speech_recognition | Speech Recognition | audio | whisper | speech-transcription,automatic-speech-recognition | audio | streaming,word-timestamps,multilingual | pytorch | stable | WebVTT:transcript |
| voice_emotion_baseline | Voice Emotion + Transcription (Baseline) | audio | whisper-spectral-emotion | speech-transcription,emotion-recognition | audio | streaming,embedding | pytorch | experimental | WebVTT:transcript;JSON:emotion_segments |

Total pipelines: 10
