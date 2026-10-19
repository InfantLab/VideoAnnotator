# Data Model: Video Page

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Date**: 2026-10-19

## Server

### `video_notes` (new table)

A researcher's note at a moment or over a span of one video (FR-017–FR-019). Keyed to the
video, not to a run (research R9).

| Column | Type | Rules |
|---|---|---|
| `id` | GUID, primary key | |
| `video_path` | str(1000), indexed | the video's resolved server path (R1) |
| `video_name` | str(255) | file name, for exports and listings |
| `video_sha256` | str(64), nullable | when known (provenance or upload); lets spec 029 join copies |
| `start_sec` | float | ≥ 0; at most the video's duration when known |
| `end_sec` | float, nullable | null for a moment; else > `start_sec` |
| `text` | text | 1–2,000 characters after trimming; an empty note is not saved (US6.1) |
| `author_id` | str(64) | the signed-in user who created it |
| `author_name` | str(200) | their name or email at the time, for exports |
| `created_at` | datetime (UTC) | |
| `updated_at` | datetime (UTC), nullable | |
| `updated_by` | str(64), nullable | |

- Several notes may share a time (edge case); order is `start_sec`, then `created_at`.
- A span made with Shift+N in reverse order is stored with start and end swapped.
- Deleting is permanent (no review workflow, out of scope).
- Created by the existing `Base.metadata.create_all` at startup (additive, like
  `ask_entries`).

### `GET /videos/runs` response (computed, not stored)

| Field | From |
|---|---|
| `path`, `name`, `available`, `unavailable_reason`, `duration_sec`, `sha256` | resolved path; file check; cv2 duration; provenance input hash when a run recorded one |
| `runs[]` | every job with this resolved `video_path`, newest first |
| `runs[].job_id`, `status`, `created_at`, `completed_at`, `batch_id`, `dataset_id`, `run_name` | the job; `run_name` is the spec 022 run folder name when there is one |
| `runs[].pipelines[]` | `name`, `status`, `error`, `files[]` (`name`, `download_url`, `kind`: result / provenance / other) |
| `projects[]` | `id`, `name` of saved datasets linked by a job's `dataset_id` or whose folder or manifest holds the video |

### Table format version 2 (spec 025 contract, additive)

- New last column `author` (text): the note's author on `note` rows; empty elsewhere.
- New measure `note`: `pipeline` = `notes`, `track` empty, `start`, `end` (empty for a moment),
  `label` = the note's text, `model` empty, `videoannotator_version` = the exporting version,
  `file` empty.
- `TABLE_FORMAT_VERSION` = 2; Parquet metadata likewise. Nothing renamed or removed.

## Viewer

### Lane (in memory)

| Field | Meaning |
|---|---|
| `kind` | lane kind (`speech`, `speakers`, `people`, `faces`, `emotion`, `scenes`, `ask_ai`, `action_units`, `head_pose`, `gaze`, `notes`, or `other:<pipeline>/<measure>`) |
| `pipeline` | the pipeline that produced the rows (`face_analysis`, `face_openface3_embedding`, …); null for notes |
| `label` | what it shows, in plain words (Decision 1), with the pipeline's short name appended when another lane of the same kind comes from a different pipeline ("Faces · OpenFace 3") |
| `run` | job id, run name and date; null for notes and standalone files |
| `form` | `span` · `moment` · `numeric` · `presence` |
| `rows[]` | `key` (track or label), `label`, `colour` (stable per key, shared with the overlay), `events` sorted by start |
| `status` | `ok` · `failed` (with reason) · `empty` |
| `overlay` | whether this kind draws on the video (no eye when it doesn't) |
| `source` | file names and provenance, for About |

Built from table rows (R5/R6): `rows → group by (pipeline, measure group, run) → rows by track or
label`. A compare lane is a second Lane of the same kind with another `run`.

### Event

`start`, `end?`, `value?`, `label?`, `confidence?`, `frame?`, and the index of its table row (so a
click in either view finds the other: SC-005).

### Lane preferences (browser, `idbKv`)

Key `lanePrefs:<server url>:<user id>` (standalone: `lanePrefs:local`):

```json
{ "version": 1,
  "collapsed": { "scenes": true },
  "overlay":   { "faces": false, "faces.landmarks": false },
  "timelineHeight": 200,
  "dataFilter": "all",
  "panelOpen": true }
```

Unknown kinds default to expanded with the overlay on. Old `OverlaySettings` keys (`pose`,
`faces`, `emotions`, `subtitles`, `speakers`, `scenes`) map onto kinds the first time.

### Page context (URL)

`/video?path=<server path>[&from=project:<id>|run:<batch id>|folder:<server path>][&run=<job id>][&t=<seconds>]`

`t` starts the video at that time (used by links from the table and, later, 029).
