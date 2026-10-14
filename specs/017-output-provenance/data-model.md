# Data Model: Output Provenance

## ProvenanceRecord (schema_version 1)

The JSON object embedded in or beside each output file, and stored per pipeline on the job.

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | `1`. Readers ignore unknown fields; new fields bump only on breaking change. |
| `pipeline` | object | `{ "name": str, "sub_pipeline": str \| null }`. `sub_pipeline` set only for `audio_processing`'s files. |
| `videoannotator_version` | str | `videoannotator.__version__` |
| `models` | ModelRef[] | may be empty (e.g. scene detection without CLIP) |
| `settings` | object | effective pipeline config, secrets redacted (`"<redacted>"`) |
| `determinism` | object | `{ "deterministic": bool, "cudnn_benchmark": bool, "cudnn_deterministic": bool, "deterministic_algorithms": bool, "cublas_workspace_config": str \| null }`; `{}` when torch isn't installed |
| `created_at` | str | UTC ISO-8601, `+00:00` |
| `job_id` | str \| null | null outside a job |
| `input` | object | `{ "name": str, "sha256": str \| null }` |
| `vlm` | object \| absent | VLM only: `{ "prompt_sha256": str, "model_digest": str \| null, "quantization": str \| null, "base_url": str }` |

Validation: `pipeline.name`, `videoannotator_version`, `created_at`, `schema_version` required;
everything else may be null/empty but present.

## ModelRef

| Field | Type | Notes |
|---|---|---|
| `name` | str | e.g. `yolo11n-pose.pt`, `pyannote/speaker-diarization-3.1`, `whisper base` |
| `source` | str | `file`, `huggingface`, `openai-whisper`, `open_clip`, `ollama`, `deepface` |
| `revision` | str \| null | sha256 of the weights file, Hub commit hash, or Ollama digest |
| `revision_kind` | str | `sha256`, `git`, `ollama-digest`, or `unknown` |
| `revision_note` | str \| absent | why `revision` is null |

## PipelineResult (existing; extended)

Adds `provenance: ProvenanceRecord | null`. Stored in `pipeline_results.provenance` (JSON,
nullable, added by additive migration). Null for jobs before this feature, and for a pipeline that
failed before its record could be built; a pipeline that failed after initialising keeps its
record (spec US4 scenario 2).

## Viewer: ProvenanceInfo and attribution

```ts
type ProvenanceInfo =
  | { kind: 'recorded'; record: ProvenanceRecord }           // full record found
  | { kind: 'partial'; videoannotatorVersion?: string }       // e.g. old COCO info.version only
  | { kind: 'none' };                                         // nothing recorded
```

`StandardAnnotationData.provenance?: Partial<Record<TrackKey, ProvenanceInfo>>`, where `TrackKey`
is the existing per-overlay key (person_tracking, face_analysis, openface3, speech_recognition,
speaker_diarization, scene_detection, vlm_annotation, elan_ground_truth).
