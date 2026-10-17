# Contract: metadata changes per pipeline

Additive to the metadata schema (Principle II): the optional `reason`/`reason_source` keys on a
setting ([data-model.md §1](../data-model.md)). Everything else is values. Benchmark figures are
quoted exactly from the source (spec 026 rule; SC-005); those below are from
[research.md](../research.md) and are re-checked against the source when written.

## speech_recognition.yaml

- `description`: names faster-whisper and large-v3-turbo; `homepage`:
  https://github.com/SYSTRAN/faster-whisper; `variant`: `faster-whisper`.
- `whisper_model.default: large-v3-turbo`, with `reason` (turbo: pruned large-v3, "minor quality
  degradation" vs large-v3, Open ASR mean WER 7.83; runtime 2.3× faster than openai-whisper on its
  benchmark) and `reason_source` (the model card).
- `compute_type`: new, technical (contracts/speech-settings.md).
- `task` description: translate needs a non-turbo model.
- `benchmarks`: replace Whisper base's LibriSpeech rows with turbo's Open ASR Leaderboard mean WER
  (7.83, model card); keep `quality_note` (adult speech only).
- `requirements.packages`: `faster-whisper`; `backends: [ctranslate2]`.
- `weights`: the CTranslate2 repository, ~1,618 MB, `cache: huggingface`.
- `citations`: keep `radford2023whisper` (the model); faster-whisper has no paper.

## speaker_diarization.yaml

- `homepage` and `model.default`: `pyannote/speaker-diarization-community-1`, `reason` (lower DER
  on 10 of 12 published sets; one licence instead of three), `reason_source` (model card).
- `model.description`: other pyannote pipelines (e.g. 3.1) can be named; they need their own
  licences.
- `benchmarks`: community-1's rows for AMI (IHM) 17.0, DIHARD 3 (full) 20.2, VoxConverse (v0.3)
  11.2, with the card's evaluation note.
- `requires_setup`: the secret, and the community-1 licence only.
- `weights`: community-1, ~33 MB.
- `citations`: add the citation the community-1 card gives (CC-BY-4.0 attribution, FR-012).

## person_tracking.yaml

- `description`, `variant` (`yolo26n-pose-bytetrack`), `model.default`
  (`models/yolo/yolo26n-pose.pt`) with `reason` (COCO-pose mAP50-95 50.0 → 57.2) and
  `reason_source` (Ultralytics pose docs).
- `benchmarks`: YOLO26n-pose 57.2 mAP pose 50-95 (e2e), note "end-to-end (NMS-free) path, 640 px".
- `iou_threshold.description`: "no effect with YOLO26 (NMS-free)" if step 4b shows so.
- `requirements.packages`: `ultralytics>=8.4`.
- `citations`: YOLO26's as Ultralytics gives it (replacing YOLO11's).

## face_analysis.yaml

- `detect_age`, `detect_gender`: `default: false`, `reason` ("trained on adult faces; says nothing
  useful about infants", pipeline review), `reason_source` (DeepFace README for the figures).
- Expression: the guide's `quality_note` says DeepFace's expression model was trained on adult
  faces (FR-023).
- Detector: RetinaFace's reason (WIDER FACE hard AP 91.4 %, arXiv:1905.00641) as a `reason` on
  the detector setting or in `quality_note` (FR-025); a benchmark entry with that figure.
- `weights`: RetinaFace and expression only.

## face_openface3_embedding.yaml

- `quality_note` or `description`: "`openface-test` is the distribution CMU names; no upstream
  change since 2025-06; non-commercial research licence" (FR-026). Pin unchanged.
