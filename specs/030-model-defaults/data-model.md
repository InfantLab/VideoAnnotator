# Data Model: Refresh Model Defaults (spec 030)

No new tables, endpoints or output keys. Three things change shape: a setting in pipeline
metadata can say why its default is what it is, the speech result is built from faster-whisper's
objects instead of openai-whisper's dict, and each changed pipeline's provenance names a
different kind of model reference.

## 1. Setting reason (metadata, registry, API, CLI)

`PipelineConfigField` (`registry/pipeline_registry.py`) gains two optional fields:

| Field | Type | Rule |
|---|---|---|
| `reason` | `str \| None` | One sentence: why this default. Plain text, shown as is. |
| `reason_source` | `str \| None` | An `https://` link to the evaluation or record the reason rests on. Ignored with a load warning if `reason` is absent or the link isn't `https://`. |

YAML:

```yaml
config_schema:
  model:
    group: technical
    label: "Diarization model"
    type: string
    default: pyannote/speaker-diarization-community-1
    reason: "Lower diarization error than speaker-diarization-3.1 on 10 of 12 published sets (e.g. AMI headset 18.8 → 17.0 % DER)."
    reason_source: https://huggingface.co/pyannote/speaker-diarization-community-1
```

API (`GET /api/v1/pipelines`, `/pipelines/{name}`): the config field object gains `reason` and
`reason_source`, both nullable: additive, as spec 026's `group`/`label`/`unit` were.
`videoannotator pipelines describe <name>` prints `Why this default: <reason> (<reason_source>)`
under the setting. The viewer ignores both for now.

Used on: `speech_recognition.whisper_model`, `speaker_diarization.model`,
`person_tracking.model`, `face_analysis.detect_age`, `face_analysis.detect_gender`,
`face_analysis.deepface.detector_backend` (if nested settings are in `config_schema`; otherwise
the RetinaFace reason goes in the pipeline's `quality_note`).

## 2. Speech result (unchanged keys, new source)

`SpeechPipeline.transcribe_audio()` returns the same dict as today; only where the values come
from changes. faster-whisper returns `(segments, info)`: a generator of `Segment` and a
`TranscriptionInfo`.

| Result key | Today (openai-whisper dict) | Now (faster-whisper) |
|---|---|---|
| `transcript` | `result["text"].strip()` | `"".join(s.text for s in segments).strip()` |
| `language` | `result["language"]` | `info.language` |
| `words[]` `word/start/end/confidence` | `segment["words"][i]` `word/start/end/probability` | `s.words[i]` `.word/.start/.end/.probability` (`None` words when `word_timestamps` is off) |
| `metadata.segments[]` `id/start/end/text/tokens/temperature/avg_logprob/compression_ratio/no_speech_prob` | same names in each segment dict | same names as `Segment` attributes |
| `metadata.model` | configured `whisper_model` | configured `whisper_model` (unchanged) |
| `metadata.num_segments`, `num_words`, `start_time`, `end_time`, `total_duration`, `task` | computed | computed the same way |

The generator is consumed once, inside `transcribe_audio()`, so a decoding error raises there and
the pipeline fails (spec 2026-10-03 "a pipeline that fails must say so"). `write_webvtt()` is
unchanged: it reads `metadata.segments[].start/end/text`.

## 3. Provenance per changed model (spec 017 `ModelRef`)

| Pipeline | Before | After |
|---|---|---|
| speech | `whisper_ref`: name `whisper base`, source `openai-whisper`, revision = sha256 from the download URL, kind `sha256` | `hub_ref`: name = repository (`mobiuslabsgmbh/faster-whisper-large-v3-turbo` or as cached, research S4), source `huggingface`, revision = snapshot commit, kind `git`; plus `quantisation` = CTranslate2's effective compute type in the pipeline's determinism/settings block |
| speaker turns | `hub_ref("pyannote/speaker-diarization-3.1")` | `hub_ref("pyannote/speaker-diarization-community-1")` (same code, new default) |
| people | `weights_ref("models/yolo/yolo11n-pose.pt", "ultralytics", path)` | same, `yolo26n-pose.pt` |
| faces | `weights_ref` for RetinaFace + Emotion + Age + Gender | RetinaFace + Emotion only, unless age/gender are switched on |

`whisper_ref()` is removed with openai-whisper. Outputs written before v1.6.0 keep their records;
the methods paragraph (spec 025) reads both kinds already.

## 4. Readiness weights (spec 011 `WeightSpec`)

| Pipeline | Before | After |
|---|---|---|
| speech | `base`, 140 MB, `cache: whisper` | faster-whisper turbo repository, ~1,618 MB, `cache: huggingface` |
| speaker turns | `pyannote/speaker-diarization-3.1`, 35 MB | `pyannote/speaker-diarization-community-1`, ~33 MB |
| people | (none listed; Ultralytics downloads) | unchanged |
| faces | retinaface 119, expression 6, age 540, gender 540 MB | retinaface 119, expression 6 MB |

The `whisper` cache kind stays in `readiness.py` (a user's metadata may still name it) but no
default uses it.

## 5. Settings added or changed

| Pipeline | Setting | Change |
|---|---|---|
| speech | `whisper_model` | default `base` → `large-v3-turbo`; values: faster-whisper size names or a CTranslate2 repository id |
| speech | `compute_type` | new, technical, default `default` (research S2) |
| speech | `use_fp16` | kept: `false` means `compute_type: float32` |
| speaker turns | `model` | default → `pyannote/speaker-diarization-community-1` |
| people | `model` | default → `models/yolo/yolo26n-pose.pt`; `iou_threshold` described as having no effect if step 4b shows so |
| faces | `detect_age`, `detect_gender` | default `true` → `false` |
| faces | `deepface.emotion_model` | already unread by the code (research §1c); removed from metadata with spec 026 FR-012's rule, or left if 026 already handled it |
