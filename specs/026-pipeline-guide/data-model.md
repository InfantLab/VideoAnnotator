# Data Model: Pipeline Guide

No database or output-file changes. The model is the pipeline metadata
(`src/videoannotator/registry/metadata/<pipeline>.yaml`) and its parsed form
(`registry/pipeline_registry.py`).

## PipelineMetadata (additions)

| Field | Type | Required for built-ins | Fallback when absent | Notes |
|---|---|---|---|---|
| `title` | string | yes | `display_name` | Behavioural, what it finds: "Faces: expression, age, gender" (R2) |
| `summary` | string, ≤ 2 sentences | yes | `description` | For researchers; names a model only when the model is the point (FR-001) |
| `homepage` | URL | yes | none (About omits it) | The method's or model's own page or repository (FR-006) |
| `benchmarks` | list of Benchmark | yes (may be `[]`) | treated as "not stated" | `[]` = "No published benchmark" (FR-007) |
| `quality_note` | string | no | none | For model-dependent pipelines (the VLM), shown instead of figures |
| `citations` | (exists, spec 025) | yes | — | Now also returned by the API and shown in About |

Unchanged: `name`, `display_name`, `description` and every other existing field.

## Benchmark

| Field | Type | Required | Rule |
|---|---|---|---|
| `name` | string | yes | Benchmark or dataset and split, e.g. "WIDER FACE (hard)" |
| `measure` | string | yes | As the source names it: "AP", "WER", "DER" |
| `figure` | string | yes | Quoted from the source, units and decimals as published; never computed |
| `source` | URL | yes | Paper, official model card, or official leaderboard |
| `note` | string | no | Model size, variant, conditions ("ResNet-50 backbone") |

An entry missing `name`, `measure`, `figure` or `source` is dropped with a warning (lenient parsing,
as for `citations`).

## PipelineConfigField (additions)

| Field | Type | Default | Rule |
|---|---|---|---|
| `group` | `behavioural` \| `technical` | `technical` | Any other value → technical, with a warning (R3) |
| `label` | string | key, title-cased | Plain words: "Most faces per frame" |
| `unit` | string | none | "s", "faces", "px", "%" |
| `min`, `max` | number | none | For numeric types |
| `applies_when` | map setting → list of values | none | All entries must match (AND); values compared as strings (R4) |

Unchanged: `type`, `default`, `description`, `widget`, `enum`.

### Validation (metadata check, built-ins only: FR-011, SC-001)

- `title`, `summary`, `homepage` present; `benchmarks` key present (empty allowed).
- `summary` is at most two sentences.
- Every setting has an explicit `group` and a `label`.
- At most five behavioural settings per pipeline (SC-002).
- Every key named in an `applies_when` is a setting of the same pipeline, and its listed values are
  valid for it (in its `enum`, or of its type).
- Every `config_schema` key is read by the pipeline's package (R6 word search).

### Visibility rule (viewer and CLI)

A setting with `applies_when` is **shown** when every condition matches the current values; when a
condition fails it is **hidden**, unless its current value differs from its default, in which case
it is shown, marked "not used with these settings" (FR-005, Edge Cases).

## Classification of the built-in settings (first pass, to confirm in implementation)

| Pipeline | Behavioural (shown) | Technical (collapsed) |
|---|---|---|
| face_analysis | detect_emotions, detect_age, detect_gender, max_faces, min_face_size | detection_backend, emotion_backend, confidence_threshold, scale_factor\*, min_neighbors\*, deepface, person_identity |
| face_openface3_embedding | min_face_size | model_path, batch_size |
| person_tracking | track_mode | model, conf_threshold, iou_threshold |
| scene_detection | min_scene_length | threshold (`enable_clip_labels` removed, R6) |
| speaker_diarization | min_speakers, max_speakers | model, use_auth_token |
| speech_recognition | language, task, word_timestamps, min_segment_duration | whisper_model, beam_size, best_of, temperature, patience, length_penalty, prepend_punctuations, append_punctuations, suppress_tokens |
| vlm_annotation | prompt, frame_interval_sec, sampling_mode | model, base_url, burst_offsets\*, think, request_timeout_sec, keep_alive |
| audio_processing (deprecated) | min_speakers, max_speakers | whisper_model (`enable_diarization` removed, R6) |

Judgement calls, to confirm with the maintainer during implementation: `confidence_threshold` and
`conf_threshold` are technical (how sure the detector must be is a method choice, and the default
is the upstream one), though a researcher may want them; `whisper_model` is technical though it
trades speed for accuracy; scene `threshold` is technical (a PySceneDetect content score with no
behavioural unit). The VLM's model is technical here; spec 027 presents it prominently anyway.

\* `applies_when`: `scale_factor`, `min_neighbors` when `detection_backend: [opencv]`;
`burst_offsets` when `sampling_mode: [frame_burst]`.
