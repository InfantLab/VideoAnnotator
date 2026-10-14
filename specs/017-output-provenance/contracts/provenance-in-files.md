# Contract: Provenance in Output Files

Readers (the viewer, the tidy export, third-party scripts) can rely on the following. The record's
shape is in [`../data-model.md`](../data-model.md) (`ProvenanceRecord`, schema_version 1).

## JSON outputs (COCO and others)

Top-level key `provenance`:

```json
{
  "info": { "description": "VideoAnnotator COCO Export", "version": "1.6.0", "year": 2026,
            "contributor": "VideoAnnotator", "date_created": "2026-10-04T12:00:00+00:00" },
  "images": [], "annotations": [], "categories": [],
  "provenance": { "schema_version": 1, "pipeline": { "name": "person_tracking", "sub_pipeline": null },
                  "videoannotator_version": "1.6.0", "models": [ ... ], "...": "..." }
}
```

- `info.date_created` is the real UTC creation time (no longer `2025-01-01T00:00:00Z`).
- Files: `*_person_tracking.json`, `*_person_tracks.json`, `*_scene_detection.json`,
  `*_face_detections.json`, `*_vlm_annotation.json`, `*_openface3_analysis.json`,
  `*_openface3_detailed.json`.

## WebVTT (`*_speech_recognition.vtt`)

The first block after the header is:

```
WEBVTT

NOTE videoannotator-provenance {"schema_version":1,"pipeline":{"name":"speech_recognition",...}}

1
00:00:00.000 --> 00:00:01.320
Ready, baby girl?
```

- One line of JSON after the marker `NOTE videoannotator-provenance `.
- Any `-->` inside the JSON is written as `-->`.
- Cue numbering and cues are unchanged.

## RTTM (`*_speaker_diarization.rttm`)

Unchanged. The record is in a companion file `<rttm file name>.provenance.json` in the same
folder, containing the record alone (not wrapped). Every place that ships the RTTM file (job folder,
artifacts ZIP, `results/files/{pipeline}` listing, `videoannotator process --output`) ships the
companion too.

## Absence

A file without a record is valid (older outputs, or a pipeline used outside a job). Readers MUST
treat a missing record as "not recorded" and MUST NOT infer one.

## API

`GET /api/v1/jobs/{id}/results` → `pipeline_results.{name}.provenance`: the same record as the
pipeline's files, or `null`.
