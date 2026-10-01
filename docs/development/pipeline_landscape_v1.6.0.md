# Processing landscape for v1.6.0: what we have, what's obsolete, what's current, what's missing

**Status**: draft, 2026-10-01. Companion to the [pipeline review](pipeline_review_v1.6.0.md)
(decisions on the pipelines we already have) and input to Phase 5 (Models) and the v1.7.0 plugin
work. Tools named here are candidates to benchmark, not decisions. Every tool marked
**licence: confirm** had no licence I could verify; check before adopting.

Ground rules for picking a tool, from the constitution and the review:
- runs locally (Principle I); no research-only or non-commercial weights as a *default*
  (Engineering Standards: licensing); AGPL isolated to a plugin;
- validated on, or at least sensible for, infants and young children;
- maintained upstream; install and download cost proportionate to what it answers.

## 1. Audio

| Processing | What we have | Obsolete? | Current best (2026) | Licence | Infant/child fit | Recommendation |
|---|---|---|---|---|---|---|
| **Who is speaking, by voice type** (key child, other child, adult female, adult male) | nothing (diarization gives anonymous speakers) | — | **VTC 2** (BabyHuBERT, LAAC-LSCP); VTC 1 beat LENA by a wide margin | licence: confirm | Built for child-centred recordings, 10 languages | **Phase 5, top priority.** Needs Python ≥ 3.13 (now supported) |
| **Adult word / syllable / phoneme counts** (the LENA AWC measure) | nothing | — | **ALICE** (Räsänen et al. 2021): counts from adult speech, using VTC for who is speaking | licence: confirm | Built for child-centred recordings | Phase 5, with VTC |
| **Conversational turns** (the LENA CTC measure) | nothing | — | Derived from voice-type segments (adult ↔ child within a window); no model needed | ours | Standard measure in the field | Analysis layer (Phase 3 tidy export / Phase 6), once VTC is in |
| Speech transcription | `speech_recognition`: openai-whisper `base` | Runtime ageing: openai-whisper's last release was 2025-06 | Whisper large-v3-turbo (MIT, 809M) via faster-whisper (MIT); NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0; English/European, more accurate than Whisper large-v3 at a quarter the size); Canary-Qwen 2.5B (most accurate) | MIT / CC-BY-4.0 | All adult-trained; transcribe caregiver speech, not infant sounds | Phase 5 benchmark: faster-whisper + turbo vs Parakeet on parent speech. Keep Whisper models (multilingual) as default unless Parakeet wins clearly in English |
| Speaker diarization | `speaker_diarization`: pyannote 3.1 | Superseded by community-1 | pyannote **community-1** (pyannote.audio 4): better speaker counting, same segmentation | MIT code, model card terms | Adult-trained | Library upgrade in spec 3; model switch in Phase 5. VTC answers the child-specific version of this question |
| **Infant vocalisations** (cry, laugh, babble, canonical babbling) | nothing; Whisper hallucinates text on them | — | VTC 2's key-child segments for *when*; cry detectors exist as research code (e.g. a self-training cry detector, open source); vocal-maturity classifiers are research-grade | varies | Specific | Phase 5: VTC first; cry/laugh as a v1.7.0 plugin when a maintained model exists |
| **Prosody / infant-directed speech** (pitch, pitch range, speech rate) | nothing (librosa is installed for audio I/O) | — | librosa `pyin` for pitch (ISC); Praat via parselmouth is the field standard but **GPL-3** | ISC / GPL-3 | Language-agnostic | Small pipeline on librosa: pitch contour and summary per voice-type segment. Cheap, widely used in IDS research |
| Voice emotion | `laion_voice` (dropped by the review) | Yes | No infant-validated model | — | — | None for now |

## 2. Faces

| Processing | What we have | Obsolete? | Current best | Licence | Infant/child fit | Recommendation |
|---|---|---|---|---|---|---|
| Face detection | `face_analysis` (DeepFace; default Haar cascade) | **Yes**: Haar cascades are gone from OpenCV 5; found no faces in the demo video | RetinaFace (MIT, used inside OpenFace 3); YuNet / OpenCV FaceDetectorYN (Apache) | MIT / Apache | Works on infant faces far better than Haar | Spec 5 (face stack) |
| Expression / action units | `face_analysis` emotion (FER-style, adult posed); OpenFace 3 AUs + 8 emotions | DeepFace emotion: yes for infants | **Infant AFAR** (Pittsburgh): AU detection fine-tuned on infants, publicly available; Baby FaceReader (Noldus, commercial, Baby FACS) | licence: confirm (AFAR) / commercial | Infant AFAR is built for exactly this; adult AU models generalise poorly to infant faces | Phase 5: evaluate Infant AFAR next to OpenFace 3 AUs |
| Age / gender | `face_analysis` (DeepFace) | Yes: meaningless for infants | Not needed as such; see "who is who" | — | — | Drop (review) |
| Face identity / re-identification across a video | `face_openface3_embedding`, person-identity utils | — | Face embeddings + track association | — | OK | Keep; part of "who is who" |

## 3. Gaze and attention

| Processing | What we have | Obsolete? | Current best | Licence | Infant/child fit | Recommendation |
|---|---|---|---|---|---|---|
| **Infant looking** (on/off, left/right: looking-time paradigms) | nothing | — | **iCatcher+**: near-human accuracy, 4 months–3.5 years, lab, field and webcam video | licence: confirm | Built for infants | Phase 5 / plugin. Directly replaces hand-coding in a whole class of experiments |
| Gaze direction (angles) | OpenFace 3 gaze | — | OpenFace 3; L2CS-Net | academic-only (OpenFace) | Adult-trained | Keep (opt-in) |
| **Gaze target** (where in the scene someone looks) | nothing | — | **Gaze-LLE** (CVPR 2025, frozen DINOv2 encoder) | licence: confirm | Adult-trained; untested on infants | v1.7.0 plugin candidate; building block for joint attention |
| Mutual gaze / joint attention | nothing | — | No mature tool; derived from gaze target + people + objects | — | Research question | Out of scope until gaze target and objects are in |

## 4. Body and movement

| Processing | What we have | Obsolete? | Current best | Licence | Infant/child fit | Recommendation |
|---|---|---|---|---|---|---|
| 2D pose | `person_tracking`: Ultralytics YOLO11-pose | Licence problem (AGPL-3.0) | **RTMPose** (via `rtmlib`, ONNX only), ViTPose, Sapiens: best on infant video (Gama et al. 2024; 2026 home-video comparison); whole-body variants include hands and face | Apache-2.0 (rtmlib) | Benchmarked on infants | Phase 5 (review decision) |
| Tracking | ByteTrack (via Ultralytics) | — | ByteTrack / BoT-SORT exist outside Ultralytics too | MIT | OK | Move with pose |
| **Movement quantity and synchrony** (motion energy analysis) | nothing | — | Motion energy analysis (frame differencing in regions of interest) and optical flow (OpenCV Farnebäck / Lucas-Kanade; RAFT); MEA synchrony predicts maternal sensitivity and reciprocity in free play | BSD / Apache | Standard in parent–infant synchrony research | **Small, cheap pipeline**: per-person motion energy from pose boxes + dyadic cross-correlation. High value for the effort |
| 3D pose / infant body models (SMIL) | nothing | — | Research-grade | — | — | Not now |

## 5. Who is who

| Processing | What we have | Obsolete? | Current best | Licence | Recommendation |
|---|---|---|---|---|---|
| **Adult vs child vs infant** for each tracked person | size-based heuristic inside `person_tracking` (`utils/size_based_person_analysis.py`): smaller box = child | Fragile: wrong whenever an adult is further from the camera than the child | Body + face age estimation (e.g. MiVOLO, face *and* body crops, state of the art on age benchmarks); pose proportions (head-to-body ratio) | licence: confirm (MiVOLO) | Phase 5. Nearly every downstream measure (who touches whom, who looks at whom, whose voice) needs it |
| Linking voices to people (**active speaker detection**) | nothing | — | LoCoNet (CVPR 2024): state of the art on AVA-ActiveSpeaker, Talkies, Ego4D | licence: confirm | v1.7.0 plugin. Joins VTC/diarization (audio) to tracked people (video) |

## 6. Interaction measures (derived, mostly no new model)

These answer the questions the field actually reports, by combining existing outputs. They belong
in the analysis layer (Phase 3's tidy export, Phase 6's corpus view), not as model pipelines:

| Measure | Built from | Notes |
|---|---|---|
| Conversational turns, response latency, vocal contingency | voice-type segments | LENA CTC equivalent |
| Movement synchrony | motion energy per person | Cross-correlation with lags |
| Proximity | pose (+ monocular depth for distance from camera: Depth Anything V2 **Small** is Apache-2.0; Base/Large/Giant are CC-BY-NC) | Relative, not metric, without calibration |
| Touch / contact | pose + person segmentation (Doyran et al. 2023–24: contact signatures in parent–infant play) | Research code; v1.7.0 plugin candidate |
| Joint attention | gaze target + objects + people | Not until gaze target exists |

## 7. Objects, context, activity

| Processing | What we have | Obsolete? | Current best | Licence | Recommendation |
|---|---|---|---|---|---|
| Scene cuts | `scene_detection` (PySceneDetect) | — | Same | BSD-3 | Keep |
| Scene labels | CLIP `ViT-B-32` zero-shot | Dated but cheap | Same family; a VLM does better | MIT | Keep as experimental |
| **Objects** (toys, books, food) | nothing | — | Open-vocabulary detection: Grounding DINO (Apache-2.0), OWLv2; segmentation + tracking: SAM 2 (Apache-2.0), SAM 3 (concept prompts, "SAM License": permissive, pass-through terms, use restrictions) | Apache / SAM License | v1.7.0 plugin: "find and track every `toy` / `book`", feeding joint-attention and object-play measures |
| Activity / free description | `vlm_annotation` (Ollama) | — | Qwen3-VL (Apache-2.0, native long video, runs in Ollama at small sizes) | Apache-2.0 | Keep; Phase 5 picks the default VLM. Covers most "what is happening" coding without a dedicated action-recognition model |

## 8. Utilities we've overlooked

| Utility | Why it matters | Tool | Licence | Recommendation |
|---|---|---|---|---|
| **De-identification** (blur faces in exported video) | Sharing data (Databrary, OSF, supplementary material) needs faces removed; labs do this by hand | `deface` (CLI, face detection + blur/box) | MIT (confirm) | **Cheap, high value**: an export option, reusing our own face detections |
| Recording quality checks | Garbage in: dark, out-of-focus, occluded or silent stretches waste pipeline time and confuse results | brightness, blur (Laplacian variance), audio level, face/person visibility from existing outputs | ours | Small pipeline or part of ingest |
| Synchronised multi-camera / multi-microphone | Many lab setups record two or more angles | audio cross-correlation for alignment | ours | v1.7.0 |

## 9. Summary

**Obsolete in today's stack** (most already decided in the review): DeepFace age/gender and
FER-style emotion for infants; Haar-cascade face detection (gone in OpenCV 5); the LAION voice and
face models; `audio_processing` (duplicate); pyannote 3.1 (superseded by community-1);
openai-whisper as a runtime (slow; same weights run faster elsewhere); Ultralytics for pose
(licence, not quality); size-based adult/child labelling.

**Overlooked processing, in order of value for developmental research**:
1. Voice type classification (VTC 2) and the counts built on it: adult words (ALICE), conversational
   turns. The field's standard measures; we have none of them.
2. Infant looking (iCatcher+).
3. Movement quantity and dyadic synchrony (motion energy): cheap, no new model.
4. Adult / child / infant role for each person, replacing the size heuristic.
5. De-identification of exported video.
6. Prosody of caregiver speech (pitch, rate): cheap, no new model.
7. Infant-specific facial action units (Infant AFAR).
8. Objects and joint attention, active speaker detection, touch (v1.7.0 plugins).

**Suggested placement**:
- *v1.6.0 Phase 5 (benchmark, then default)*: VTC 2 (+ ALICE), RTMPose, faster-whisper /
  large-v3-turbo vs Parakeet, pyannote community-1, face detection replacement, adult/child role,
  default VLM.
- *v1.6.0, small new pipelines or export options*: motion energy + synchrony, prosody, quality
  checks, de-identified export.
- *v1.7.0 plugins*: iCatcher+ (unless licence and packaging make it easy sooner), Infant AFAR,
  Gaze-LLE, objects (Grounding DINO / SAM), active speaker detection, touch, cry/laugh.

Licences to confirm before any of these is adopted: VTC 2, ALICE, iCatcher+, Infant AFAR,
Gaze-LLE, LoCoNet, MiVOLO, deface.

## Sources

- VTC: https://arxiv.org/abs/2005.12656 ; VTC 2 / BabyHuBERT: https://arxiv.org/abs/2509.15001 ,
  https://github.com/LAAC-LSCP/VTC
- ALICE: Räsänen et al. 2021, *Behavior Research Methods*:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8062390/
- iCatcher+: https://pmc.ncbi.nlm.nih.gov/articles/PMC10471135
- Infant pose comparison: https://arxiv.org/abs/2406.17382 ; rtmlib: https://github.com/Tau-J/rtmlib
- Infant AFAR: https://local.psy.miami.edu/faculty/dmessinger/c_c/rsrcs/rdgs/emot/OnalErtugrul2022_Article_InfantAFARAutomatedFacialActio.pdf ;
  Baby FaceReader: https://noldus.com/index.php/blog/baby-facereader-automated-facial-expression-analysis
- Motion energy analysis in mother–infant free play: https://link.springer.com/article/10.3758/s13428-024-02563-5
- Touch / contact signatures: https://research-portal.uu.nl/en/publications/decoding-contact-automatic-estimation-of-contact-signatures-in-pa/
- Gaze-LLE: https://openaccess.thecvf.com/content/CVPR2025/html/Ryan_Gaze-LLE_Gaze_Target_Estimation_via_Large-Scale_Learned_Encoders_CVPR_2025_paper.html
- LoCoNet: https://arxiv.org/abs/2301.08237
- MiVOLO: https://github.com/WildChlamydia/MiVOLO
- Speech recognition comparison (2026): https://northflank.com/blog/best-open-source-speech-to-text-stt-model-in-2026-benchmarks ,
  https://openwhispr.com/blog/parakeet-vs-whisper-vs-nemotron
- pyannote community-1: https://pyannote.ai/blog/community-1 ; benchmark https://www.pyannote.ai/benchmark
- SAM 3 licence: https://github.com/facebookresearch/sam3/blob/main/LICENSE ; SAM 2 (Apache-2.0)
- Grounding DINO: https://playground.roboflow.com/models/idea-research/grounding-dino
- Depth Anything V2 (licences by size): https://github.com/DepthAnything/Depth-Anything-V2
- Qwen3-VL: https://docs.kanaries.net/articles/qwen3-vl
- deface: https://pypi.org/project/deface/
- Infant cry detection: https://link.springer.com/article/10.1007/s00521-022-08129-w
