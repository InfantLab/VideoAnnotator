# Pipeline review for v1.6.0

**Status**: decided, 2026-10-01: the maintainer accepted every recommendation (the **Decision**
column). Removing TensorFlow (with `face_analysis`'s replacement) is the priority, as the path to
Python 3.14. Part of v1.6.0 Phase 1
([`roadmap_v1.6.0.md`](roadmap_v1.6.0.md)); it sets the scope of the dependency audit
([`dependency_audit_v1.6.0.md`](dependency_audit_v1.6.0.md)) and of Phase 5's model work.

Library *versions* are the audit's subject. Switching a pipeline's *default model* waits for the
Phase 5 benchmark, so "replace" below means "replace the tool, once the benchmark agrees".

## 1. What the field asks

Developmental researchers coding parent–child video ask a short list of questions. For each, the
pipeline we have today, and the tool that answers it best on infant and child data.

| Question | What we have | Best current tool for infant/child data | Gap |
|---|---|---|---|
| Who is speaking: adult female, adult male, key child, other child? | `speaker_diarization` (pyannote 3.1): anonymous speaker labels, trained on adult speech | **VTC 2** (voice type classifier, LAAC-LSCP): KCHI / OCH / FEM / MAL, trained on child-centred recordings in 10 languages; VTC 1 already beat LENA by a wide margin. Needs Python ≥ 3.13. Licence not stated in the repo or model card: confirm before adopting | Large: the field's standard question has no pipeline |
| What does the caregiver say? | `speech_recognition` (openai-whisper `base`) | Whisper family. faster-whisper (MIT, maintained, same models, much faster) or WhisperX (BSD-2, word timestamps by forced alignment) | Small: same models, better runtimes |
| Infant vocalisations: when, how many, what kind? | none (Whisper hallucinates text on non-speech) | VTC 2's KCHI segments for *when*; vocal maturity (canonical babbling) classifiers are research-grade | Large |
| Faces and expressions | `face_analysis` (DeepFace: emotion, age, gender), `face_openface3_embedding` (landmarks, action units, gaze, emotion) | OpenFace 3.0 for landmarks and action units; adult-trained expression models transfer poorly to infants, so AUs are more defensible than emotion labels | Medium |
| Movement, posture | `person_tracking` (Ultralytics YOLO11-pose + ByteTrack) | On infant video, ViTPose and RTMPose lead; RTMPose most accurate and robust in home video, MediaPipe weakest (Gama et al. 2024; 2026 home-video comparison). `rtmlib` runs RTMPose with only ONNX Runtime, Apache-2.0 | Medium (licence, see §2) |
| Gaze, looking | OpenFace 3 gaze angles (adult-trained) | **iCatcher+**: infant looking (on/off, left/right) at near-human accuracy, 4 months–3.5 years, lab, field and webcam video. Licence to confirm | Large for looking-time paradigms |
| Joint attention | none | No mature automatic tool; a research question built from gaze + pose + objects | Out of scope for v1.6.0 |
| Touch | none | No mature automatic tool | Out of scope for v1.6.0 |
| Scene and context changes | `scene_detection` (PySceneDetect + CLIP labels) | PySceneDetect is the standard for cuts | None |
| Anything else, described in words | `vlm_annotation` (local VLM via Ollama) | Same | None |

## 2. Existing pipelines

| Pipeline | Question | Upstream | Install / download | Licence | Infants | Overlap | Recommendation | Decision |
|---|---|---|---|---|---|---|---|---|
| `speech_recognition` | Caregiver speech | openai-whisper: last release 2025-06, slow-moving | `audio` extra; `base` model 140 MB; sdist build | MIT | Fine for adult speech; not for infant sounds | Duplicated inside `audio_processing` | **Keep.** Consider faster-whisper as the backend (same weights) in the audit || Agreed |
| `speaker_diarization` | Who speaks | pyannote.audio active (4.0.7, 2026-06); `community-1` model, free and open, better speaker counting than 3.1 | `audio` extra; 35 MB, gated (HF token + licence acceptance) | MIT code; model card terms | Adult-trained; anonymous labels, not child/adult | Duplicated inside `audio_processing` | **Keep**, migrate to pyannote.audio 4 (required by the torch upgrade, see audit). Model switch to `community-1` waits for Phase 5 || Agreed |
| `audio_processing` | Speech + who speaks, in one pass | as above | as above | as above | as above | Entirely duplicates the two above | **Drop**: two pipelines doing the same work twice. Keep its name as an alias for one release if presets or docs use it || Agreed |
| `laion_voice` | Emotion in voice | LAION Empathic-Insight Voice: last updated 2025-05; 1–4 downloads/month | `audio-laion`; **16–32 GB** of weights | per model card | Trained on adult (largely acted) speech | Uses Whisper embeddings | **Drop** (can return as a v1.7.0 plugin). Cost far exceeds any infant-research value || Agreed |
| `face_analysis` | Faces, expressions | DeepFace maintained (0.0.101, 2026-09) | `face` extra; pulls **TensorFlow** (only reason it's in the tree; blocks Python 3.14); age + gender models 1.1 GB | MIT | Age and gender are meaningless for infants; FER-style emotion is trained on adult posed faces. Its default detector (`opencv`) is the Haar cascade that **opencv-python 5 removed** | Detection overlaps `face_openface3_embedding` | **Replace** in Phase 5: face detection and expressions from OpenFace 3 or a torch-based detector; drop age/gender now. Until then, mark experimental and default its detector to `retinaface`. On the viewer's demo video (a parent and infant playing peekaboo) it found **no faces** with its default settings, on Python 3.12 and 3.13 alike || Agreed |
| `face_laion_clip` | Face attributes by CLIP | LAION face models: no downloads in 30 days | `face-laion` (needs `face` too); 30–300 MB | per model card | Zero-shot labels, unvalidated on infants | Uses `face_analysis` for detection | **Drop** (v1.7.0 plugin possible) || Agreed |
| `face_openface3_embedding` | Landmarks, action units, gaze, emotion | OpenFace 3.0 (CMU MultiComp, 2025). The official package is `openface-test` (CMU's README says so) but it ships with no author, homepage or licence metadata, and releases after 0.1.13 pin Pillow 9.4 / numpy 1.26 / scipy 1.13 | `face-openface3`; 282 MB | **Academic / non-profit, non-commercial research use only** | Best face tool we have for infants (AUs, gaze), still adult-trained | Detection overlaps `face_analysis` | **Keep**, as an opt-in extra with the licence shown before install (viewer and CLI). Rename to say what it outputs (e.g. `face_openface3`). Keep `==0.1.13` || Agreed |
| `person_tracking` | Movement, posture | Ultralytics very active | `person` extra; `yolo11n-pose` | **AGPL-3.0**. Fine for open research use, but anyone who modifies and serves it over a network, or ships it in a product, takes on AGPL obligations; models trained with it fall under AGPL too. Our MIT licence doesn't change that | YOLO-pose isn't in the infant pose benchmarks; ViTPose/RTMPose lead | — | **Keep for v1.6.0, replace in Phase 5** with RTMPose via `rtmlib` (Apache-2.0, ONNX, best in home-video infant benchmarks), subject to the benchmark. State the AGPL in the install notice until then || Agreed |
| `scene_detection` | Cuts, context | PySceneDetect active (0.7.1, 2026-07); open_clip active | `scene` extra; CLIP `ViT-B-32` | BSD-3 / MIT | Neutral (cuts); CLIP labels unvalidated | — | **Keep.** Mark the CLIP labels experimental || Agreed |
| `vlm_annotation` | Anything, in words | Ollama | `llm` extra; models via Ollama | MIT client; model licences vary | Depends on the model and prompt | — | **Keep** (experimental) || Agreed |

## 3. Candidates to add

The full survey, beyond these three, is in [`pipeline_landscape_v1.6.0.md`](pipeline_landscape_v1.6.0.md).

Not v1.6.0 scope by default (Phase 5 decides), but they answer the field's top questions:

1. **VTC 2** for who is speaking in child-centred terms. Requires Python ≥ 3.13, which the audit
   recommends anyway. Confirm the weights' licence first.
2. **iCatcher+** for infant looking. Confirm licence and packaging.
3. **RTMPose via `rtmlib`** as the pose backend (see `person_tracking`).

## 4. Consequences for the dependency audit

If the recommendations stand:
- Dropping `laion_voice`, `face_laion_clip` and `audio_processing` removes the `audio-laion` and
  `face-laion` extras and 16–32 GB of optional downloads.
- Replacing `face_analysis` removes TensorFlow, tf-keras and DeepFace, which unblocks Python 3.14
  and the opencv-python 5 upgrade. Until then, opencv-python stays `<5`.
- Keeping `speaker_diarization` with the torch upgrade forces the pyannote.audio 4 migration:
  `torch==2.6.0` exists only because torchaudio ≥ 2.9 removed `AudioMetaData`, which
  pyannote.audio 3 uses.

## Sources

- Lavechin et al., *An open-source voice type classifier for child-centered daylong recordings*
  (2020): https://arxiv.org/abs/2005.12656
- Charlot et al., *BabyHuBERT* (VTC 2, 2025): https://arxiv.org/abs/2509.15001 ; code
  https://github.com/LAAC-LSCP/VTC ; weights https://huggingface.co/coml/VTC-2.0
- Gama et al., *Automatic infant 2D pose estimation from videos: comparing seven deep neural
  network methods* (2024): https://arxiv.org/abs/2406.17382
- Erel et al., *iCatcher+* (2023): https://pmc.ncbi.nlm.nih.gov/articles/PMC10471135
- pyannote.audio 4 and community-1: https://pyannote.ai/blog/community-1
- OpenFace 3.0: https://github.com/CMU-MultiComp-Lab/OpenFace-3.0 (licence in its `LICENSE`)
- Ultralytics licence: https://www.ultralytics.com/license
- rtmlib: https://github.com/Tau-J/rtmlib
- WhisperX: https://github.com/m-bain/whisperx ; faster-whisper:
  https://github.com/SYSTRAN/faster-whisper
- Package versions, dates and licences: PyPI JSON API, 2026-09-30
