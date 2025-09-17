# Output Naming Conventions (v1.2.x)

Status: Stable (subject to extension, not breaking renames in v1.2.x)
Scope: File names emitted by pipelines or batch jobs placed under job result directories.

## Goals
- Predictable discovery for downstream tooling (parsers, exporters, UI).
- Avoid ad hoc glob patterns per pipeline.
- Provide forward-compatible placeholder segments for future multimodal fusion.

## General Pattern
```
{job_id}_{pipeline}[_descriptor].{format_extension}
```
Where:
- `job_id`: Canonical job identifier (UUID-like or incremental).
- `pipeline`: Registry `name` field (stable slug).
- `_descriptor`: Optional, only when multiple artifacts of same logical format exist.
- `format_extension`: Chosen by output format standard or JSON when unambiguous.

## Standard Formats
| Output Format | Primary Extension | Example | Notes |
|---------------|-------------------|---------|-------|
| COCO          | `.coco.json`      | `123e4567_scene_detection.coco.json` | Allows distinguishing from arbitrary JSON |
| RTTM          | `.rttm`           | `123e4567_voice_diarization.rttm` | Follows diarization ecosystem norm |
| WebVTT        | `.vtt`            | `123e4567_scene_summary.vtt` | Subtitle-compatible |
| JSON (generic)| `.json`           | `123e4567_person_tracking.json` | Only when schema is pipeline-specific |
| Emotion (spec)| `.emotion.json`   | `123e4567_voice_emotion_baseline.emotion.json` | Uses defined emotion schema |
| Embeddings    | `.embeddings.json`| `123e4567_face_openface3_embedding.embeddings.json` | Vector arrays or per-face entries |
| Tracks (COCO) | `.tracks.coco.json`| `123e4567_person_tracking.tracks.coco.json` | Distinguish from detection-only outputs |

## Descriptor Segment Guidelines
Descriptor (the bracketed optional segment) SHOULD be used when:
- Multiple related artifacts differ semantically (e.g., `tracks` vs `detections`).
- Emotion timeline separate from other JSON outputs (`.emotion.json`).
- Embeddings exported distinct from structural annotations (`.embeddings.json`).

Avoid multiple descriptors; prefer a single concise token.

## Stability Policy
- Extensions above are frozen for v1.2.x line.
- New descriptors MAY be added (e.g., `.fusion.emotion.json`) but existing ones will not be renamed.
- Any breaking rename requires a major (v1.x -> v2.0.0) or explicit migration note.

## Future Reserved Descriptors
| Descriptor | Intended Use |
|-----------|--------------|
| `fusion`  | Multimodal merged outputs (e.g., combined face+voice emotion) |
| `features`| Generic feature arrays not covered by embeddings |
| `qa`      | Quality assessment metrics |

## Rationale
Explicit extensions reduce downstream branching logic (e.g., can glob `*.emotion.json`). Using pipeline slug ensures uniqueness and traceability to registry metadata.

## Consumer Recommendations
- Prefer explicit extension matching over substring heuristics.
- Validate emotion files against `emotion_validator` (see validator section once implemented).
- Derive pipeline family or tasks via registry API rather than file name parsing beyond the pipeline slug.

## Examples
```
5ff12ab2_person_tracking.tracks.coco.json
5ff12ab2_face_laion_clip.emotion.json
5ff12ab2_voice_emotion_baseline.emotion.json
5ff12ab2_face_openface3_embedding.embeddings.json
```

## Open Questions (v1.3.0+)
- Namespace merged multimodal outputs (`fusion` descriptor) vs new pipeline name.
- Add optional checksum or schema_version embedding in filename? (Likely no; keep internal.)
