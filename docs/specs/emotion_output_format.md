# Emotion Output Format Specification (v0.1)

Status: Draft (v1.2.1 scope)
Audience: Pipeline developers, downstream analytics, storage adapters

## Goals

Provide a lightweight, consistent JSON structure for emotion-related outputs across video and audio pipelines (facial affect, vocal affect, multimodal fusion) to enable:

- Unified downstream aggregation (timeline fusion, parent-child interaction metrics)
- Stable storage & export (future: conversion to CSV / parquet)
- Extensibility for confidence calibration and multi-label cases

## Non-Goals (v0.1)

- Complex hierarchical affect models (e.g., arousal/valence dominance maps)
- Real-time streaming envelope (can wrap this structure later)
- Multimodal fusion arbitration logic (handled by separate fusion pipeline)

## Top-Level File Structure

```
{
  "schema_version": 1,
  "source_pipeline": "face_laion_clip",
  "media": {
    "type": "video",
    "path": "relative/or/logical/reference.mp4",
    "frame_rate": 29.97
  },
  "emotions": [ ... entries ... ],
  "labels": {
    "taxonomy": "basic-7",
    "definitions": {
      "happy": "Positive affect / joy",
      "sad": "Downward affect / sorrow",
      "neutral": "No strong affect signal",
      "angry": "Irritation / anger",
      "fear": "Fear / anxiety",
      "surprise": "Startle / surprise",
      "disgust": "Disgust / aversion"
    }
  },
  "metadata": {
    "pipeline_variant": "laion-clip-face",
    "generation_time": "2025-09-17T12:34:56Z"
  }
}
```

## Emotion Entry Object

Each element in `emotions` array:

```
{
  "start": 12.533,          // seconds (float)
  "end": 12.867,            // seconds (float, end-exclusive)
  "labels": [               // ranked or multi-label
    { "label": "happy", "confidence": 0.82 },
    { "label": "surprise", "confidence": 0.11 }
  ],
  "source": {
    "modality": "video",    // video | audio | multimodal
    "channels": ["face"],    // e.g., face, voice, posture
    "aggregation": "frame-window" // frame-window | segment | fusion
  },
  "subject": {
    "id": "child_01",       // optional stable ID (link to tracking / diarization)
    "type": "person"         // person | speaker | group
  },
  "origin_refs": {
    "face_track_id": 4,      // optional reference to tracking output
    "speaker_turn_id": null
  },
  "confidence": 0.82,        // duplicate of top label for convenience
  "quality_flags": []        // e.g., ["low-light", "occluded"]
}
```

### Required Fields

- `start`, `end`
- `labels` (non-empty, sorted by confidence desc)
- `source.modality`

### Optional Fields

- `subject`, `origin_refs`, `quality_flags`, `confidence`

### Label Taxonomies

Initial recommended taxonomies:

- `basic-7` : happy, sad, neutral, angry, fear, surprise, disgust
- `basic-6` : anger, disgust, fear, happiness, sadness, surprise (neutral omitted)
- `valence-arousal` (Future): Provide continuous values instead of discrete labels: `{ "valence": -0.2, "arousal": 0.7 }` in place of `labels`.

Pipelines SHOULD declare which taxonomy they emit in `labels.taxonomy`.

## Validation Rules (v1 Minimal)

- `end > start`
- All `confidence` values within [0,1]
- Labels unique within a single entry
- First label confidence equals top-level `confidence` if present

## COEXISTENCE WITH OTHER OUTPUTS

- Face pipelines may also emit embeddings in separate JSON; emotion file is independent but may cross-reference track IDs.
- Audio pipelines may produce speech diarization (RTTM) linking speaker turns to `subject.id`.
- A future fusion pipeline can merge per-modality emotion entries into unified multimodal segments (set `source.aggregation = "fusion"`).

## Export / Conversion (Future Guidance)

Potential derivative tabular format (not implemented yet):

```
start,end,subject_id,label,confidence,modality,channels
12.533,12.867,child_01,happy,0.82,video,face
```

## Extension Points

| Field                  | Future Use                                                |
| ---------------------- | --------------------------------------------------------- |
| quality_flags          | occlusion, motion-blur, clipping, low-volume              |
| origin_refs            | keypoint-based gaze IDs, joint attention pair identifiers |
| labels.alt_confidences | calibration scores from ensemble models                   |

## Rationale

A flat list of time-bounded entries keeps ingestion simple, aligns with common diarization and transcription segment structures, and avoids premature hierarchical emotion modeling.

## Next Steps

1. Add validator helper in a future iteration.
2. Add fusion pipeline spec referencing both face and audio sources.
3. Consider `valence-arousal` continuous representation once needed.
