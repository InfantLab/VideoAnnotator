# Contract: The Results Table (format version 1)

What analysis code can rely on. Later versions only add columns (at the end) or measures (FR-014).

## Columns, in this order

| Column | Type | Meaning |
|---|---|---|
| `run` | text | the run's folder name (`BabyJokes wave 2 (2026-10-06)`); empty outside a run |
| `video` | text | the video's folder name within the run (or the video's file stem); unique within `run` |
| `pipeline` | text | the pipeline that made the row (`person_tracking`, `audio_processing`, …) |
| `event` | integer | numbers events within (`run`, `video`, `pipeline`); rows of one event share it |
| `track` | text | who or what the pipeline followed over time (person id, speaker); empty when none |
| `start` | number | seconds from the start of the video |
| `end` | number | seconds; empty for an observation at one moment |
| `frame` | integer | video frame number, when the pipeline recorded it |
| `measure` | text | what the row is (see below) |
| `label` | text | categorical or text result, or the component a `value` belongs to |
| `value` | number | numeric result |
| `confidence` | number | the pipeline's own score for this row, when it gives one |
| `x`, `y`, `width`, `height` | number | bounding box in pixels (top-left origin), on `person`/`face` rows |
| `model` | text | model names from the file's provenance, ` + `-joined; `not recorded` without one |
| `videoannotator_version` | text | version that made the file; `not recorded` without provenance |
| `file` | text | the native file the row came from (file name only) |

CSV: UTF-8 without BOM, comma-separated, header row, RFC 4180 quoting where needed, `\n` line ends,
missing values as empty fields, numbers unrounded. Parquet: same columns; numeric columns typed
`double`/`int64`, text `string`; metadata keys `videoannotator.table_format_version` = `1` and
`videoannotator.export_summary` (JSON).

## Row order

`run`, `video`, then `start`, `pipeline`, `event`, then measure order within the event (as listed
below). Stable for unchanged input (FR-015).

## Measures, by pipeline

| Pipeline (file) | `measure` | `label` | `value` | `confidence` | other |
|---|---|---|---|---|---|
| person_tracking (`_person_tracking.json`) | `person` | `person_label` (`parent`, `infant`, …) | | detection `score` | `track` = `person_id`; box; `frame` |
| face_analysis (`_face_detections.json`) | `face` | | | detection `score` | box; `frame` |
| | `emotion` | dominant emotion | | `emotion_confidence` | |
| | `emotion_score` | emotion name | score | | one row per emotion |
| | `age` | | estimated age | | |
| | `gender` | `gender` | | `gender_confidence` | |
| face_openface3_embedding (`_openface3_analysis.json`) | `face` | | | `openface3.confidence` | box; `track` = `openface3.track_id` when present |
| | `action_unit` | AU name (`AU01_Inner_Brow_Raiser`) | intensity | | one row per AU |
| | `action_unit_presence` | AU name | 1 or 0 | | one row per AU |
| | `head_pose` | `pitch`, `yaw`, `roll` | degrees | pose `confidence` | |
| | `gaze` | `direction_x`, `direction_y`, `direction_z` | component | gaze `confidence` | |
| | `emotion` | `dominant` | | its probability | when emotions were enabled |
| | `emotion_score` | emotion name | probability | | |
| | `valence`, `arousal` | | value | | |
| scene_detection (`_scene_detection.json`) | `scene` | `scene_type` | | `score` | `start`/`end` = scene bounds |
| | `scene_score` | scene class | score | | one row per class |
| speaker_diarization (`_speaker_diarization.rttm`) | `speaker_turn` | speaker label | | | `track` = speaker; `start`/`end` |
| speech_recognition (`_speech_recognition.vtt`) | `speech` | cue text | | | `start`/`end` |
| vlm_annotation (`_vlm_annotation.json`) | `vlm_label` | `label` | | | `frame`; at the sample time |
| | `vlm_reasoning` | `reasoning` | | | only when present and different from the label |
| | `vlm.<field>` | the field's value (choice, yes/no, text) | the number (number fields) | | spec 033: one row per field of a shaped answer that parsed |
| | `vlm_fit` | `yes` / `no` | | | spec 033: shaped answers only |
| | `vlm_fit_problem` | why it didn't fit | | | spec 033: only when `vlm_fit` is `no` |

Times: COCO `timestamp` (OpenFace: `openface3.timestamp`, VLM: `timestamp_sec`); scenes'
`start_time`/`end_time`; RTTM start and start + duration; WebVTT cue times.

When a pipeline ran inside `audio_processing`, its rows say `pipeline = audio_processing`; the
`measure` tells speech from speaker turns.

Not in the table: keypoints, landmarks, embeddings, segmentation, per-annotation run statistics
(VLM timings and tokens), `person_tracks.json`, `openface3_detailed.json`.

## Example (demo clip)

```
run,video,pipeline,event,track,start,end,frame,measure,label,value,confidence,x,y,width,height,model,videoannotator_version,file
,demo_clip,speaker_diarization,1,SPEAKER_00,0.638,1.431,,speaker_turn,SPEAKER_00,,,,,,,pyannote/speaker-diarization-3.1,1.5.0,demo_clip_speaker_diarization.rttm
,demo_clip,speech_recognition,1,,0.0,1.32,,speech,"Ready, baby girl?",,,,,,,whisper base,1.5.0,demo_clip_speech_recognition.vtt
```
