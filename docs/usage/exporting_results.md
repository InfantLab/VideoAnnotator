# Getting results out: one table, ELAN, a methods paragraph

Each pipeline writes its own standard format (COCO JSON, WebVTT, RTTM). Those files stay as they
are. For analysis, VideoAnnotator also turns them into:

- **One table** of every pipeline's results, the same columns for every pipeline, as CSV (or
  Parquet). Read it with one `read.csv` in R or `pandas.read_csv` in Python.
- **ELAN files** (`.eaf`), one per video, with one tier per pipeline track, to check or correct the
  results next to your own coding.
- **A methods paragraph** with BibTeX, naming every model and its exact weights, the settings that
  differ from the defaults, and what to cite.

All three are built from the result files and the provenance recorded in them, so they work on any
results folder: one on your computer, one a colleague sent you, or one from an older version
(where something wasn't recorded, they say `not recorded`; they never guess).

## In the viewer

On the **Videos** page, **Export ▾** exports what's shown (a folder, a project, a run) or, in the
selection bar, the videos you selected; a video's [Video page](video_page.md) exports the runs its
tracks show. A finished run's header leads with its table: **N videos → results table**.

- **Results table (CSV)** / **(Parquet)**: downloads the table.
- **ELAN (.eaf)**: one file for one video; a zip with one file per video otherwise.
- **Methods**: shows the paragraph and BibTeX, each with a Copy button, and a Markdown download.

For a folder, a project or a selection, choose **Latest results** (for each video and pipeline, its
newest finished result: what the Video page shows) or **Every run** (all of them, with a `run`
column telling them apart). See [Videos and projects](videos_and_projects.md#exporting-results).

If anything was left out (a file VideoAnnotator can't read, a video whose job failed), the viewer
lists it under the buttons.

## From a terminal

These need no server and no network: point them at any results folder (one video's folder, a run's
folder, a folder of runs, or an extracted results download).

```bash
videoannotator export table   "~/VideoAnnotator/BabyJokes wave 2 (2026-10-06)"
videoannotator export elan    "~/VideoAnnotator/BabyJokes wave 2 (2026-10-06)"
videoannotator export methods "~/VideoAnnotator/BabyJokes wave 2 (2026-10-06)" -o methods.md
```

- `export table` writes `<folder>/<folder name>.csv` unless you give `-o PATH` (`-o -` prints it).
  `--format parquet` writes Parquet.
- `export elan` writes `<video>.eaf` beside each video's results, or into `-o FOLDER`.
- `export methods` prints Markdown (prose, model revisions, BibTeX, any VLM prompts).
- `--json` prints a machine-readable summary instead (for scripts and agents).
- Exit codes: 0 done (even if some files were skipped, which are listed), 1 nothing to export,
  2 bad arguments, 3 Parquet asked for but not installed.

## Over the API

```
GET /api/v1/jobs/{job_id}/export/table?format=csv|parquet
GET /api/v1/batches/{batch_id}/export/table?format=csv|parquet
GET /api/v1/datasets/{dataset_id}/export/table?format=csv|parquet
GET /api/v1/jobs/{job_id}/export/elan        (one .eaf)
GET /api/v1/batches/{batch_id}/export/elan   (zip, one .eaf per video)
GET /api/v1/jobs/{job_id}/methods            (JSON; ?format=markdown for a .md)
GET /api/v1/batches/{batch_id}/methods
POST /api/v1/videos/export/table             (any videos; see below)
POST /api/v1/videos/export/elan
POST /api/v1/videos/export/methods
```

The `/videos/export/*` routes take a set of videos as JSON: exactly one of `"paths": [...]`,
`"project_id"` (a saved dataset) or `"run_id"` (a batch id, or a job id), plus `"results":
"latest"` (each video's newest completed result per pipeline, the default) or `"all"`, and
`"format"` (`csv`/`parquet` for the table, `json`/`markdown` for methods).

The `X-VideoAnnotator-Export-Summary` response header holds what the export covered and left out,
as JSON.

## The table

One row per **measure** of one **event**. An event is one thing a pipeline found: a person in a
frame, a face, a speech segment, a speaker turn, a scene, a VLM label. Its first row says what it
is; further rows with the same `event` number are what was measured about it. This long layout keeps
the same columns whatever pipelines you ran; turn it wide in one call (`tidyr::pivot_wider`,
`pandas.pivot_table`) when you want one row per event.

| Column | Meaning |
|---|---|
| `run` | the run's folder name; empty outside a run |
| `video` | the video's folder name in the run (or the video's file name, without extension) |
| `pipeline` | the pipeline that made the row |
| `event` | numbers events within a run, video and pipeline; rows of one event share it |
| `track` | who or what the pipeline followed over time (a person's id, a speaker); empty when none |
| `start` | seconds from the start of the video |
| `end` | seconds; empty for something seen at one moment (a frame) |
| `frame` | the video frame, when the pipeline recorded it |
| `measure` | what the row is (below) |
| `label` | a category or text (an emotion, a transcript), or which component a `value` is |
| `value` | a number |
| `confidence` | the pipeline's own score for this row, when it gives one |
| `x`, `y`, `width`, `height` | bounding box in pixels from the top-left, on `person` and `face` rows |
| `model` | the models that made the row, from the file's provenance; `not recorded` without it |
| `videoannotator_version` | the version that made the file; `not recorded` without provenance |
| `file` | the result file the row came from |
| `author` | who wrote a researcher's note (`note` rows); empty on every pipeline row |

CSV files are UTF-8 with a header row; missing values are empty fields.

### Measures, by pipeline

| Pipeline | `measure` | `label` | `value` | `confidence` |
|---|---|---|---|---|
| person_tracking | `person` | `parent`, `infant`, … (a size heuristic: check it) | | detection score |
| face_analysis | `face` | | | detection score |
| | `emotion` | most likely expression | | its probability |
| | `emotion_score` | each expression | its score | |
| | `age` | | estimated age | |
| | `gender` | estimated gender | | its probability |
| face_openface3_embedding | `face` | | | detection score |
| | `action_unit` | AU name (`AU01_Inner_Brow_Raiser`) | intensity | |
| | `action_unit_presence` | AU name | 1 present, 0 absent | |
| | `head_pose` | `pitch`, `yaw`, `roll` | degrees | pose confidence |
| | `gaze` | `direction_x`, `direction_y`, `direction_z` | component | gaze confidence |
| | `emotion`, `emotion_score`, `valence`, `arousal` | as above | | |
| scene_detection | `scene` | scene type | | score |
| | `scene_score` | each scene type | its score | |
| speaker_diarization | `speaker_turn` | speaker (`SPEAKER_00`, …) | | |
| speech_recognition | `speech` | the words | | |
| vlm_annotation | `vlm_label` | the model's label (with an answer shape: the first field, or `DIDN'T FIT`) | | |
| | `vlm_reasoning` | its reasoning, when different from the label | | |
| | `vlm.<field>` | a field's value (one of a list, yes/no, text) | the number (number fields) | |
| | `vlm_fit` | `yes` or `no`: whether the answer fit its shape | | |
| | `vlm_fit_problem` | why it didn't fit | | |
| notes (researchers' notes, not a pipeline) | `note` | the note's text | | |

When speech and speaker turns came from `audio_processing`, their rows say `audio_processing`; the
`measure` tells them apart.

Notes are the ones written on the video's page (Notes, `N`). They belong to the video, not a run, so
their `run` is empty and a video's notes appear once per export, however many runs of it the export
covers; `end` is empty for a note at a moment. An export from a copied folder, without the server's
database, has no notes and says so in its summary.

Not in the table: pose keypoints, facial landmarks, face embeddings (one number is meaningless on
its own; they are in the result files the `file` column names).

### Reading it

```r
library(readr)
results <- read_csv("BabyJokes wave 2 (2026-10-06).csv")
turns <- subset(results, measure == "speaker_turn")
aggregate(end - start ~ track, data = turns, FUN = sum)   # seconds each speaker talks
```

```python
import pandas as pd

results = pd.read_csv("BabyJokes wave 2 (2026-10-06).csv")
turns = results[results.measure == "speaker_turn"]
(turns.end - turns.start).groupby(turns.track).sum()  # seconds each speaker talks
```

A question with an answer shape gives one `vlm.<field>` row per field. One column per field is a
pivot away (or download Ask AI's "Answers, one column per field", or
`GET /api/v1/jobs/{id}/export/wide`):

```r
library(tidyr)
vlm <- subset(results, startsWith(measure, "vlm.") | measure == "vlm_fit")
wide <- pivot_wider(vlm, id_cols = c(video, start), names_from = measure,
                    values_from = c(label, value))
```

```python
vlm = results[results.measure.str.startswith("vlm.") | (results.measure == "vlm_fit")]
wide = vlm.pivot_table(
    index=["video", "start"], columns="measure", values="label", aggfunc="first"
)
```

Two worked notebooks, one in R and one in Python, are in
[`examples/notebooks/`](https://github.com/InfantLab/VideoAnnotator/tree/master/examples/notebooks).

### Parquet

Parquet keeps the column types and is smaller and faster for large runs. It needs an optional
component on the server or computer doing the export: `pip install videoannotator[parquet]` (an
administrator can also install it from the viewer when Parquet is first asked for). The file's
metadata holds the export summary, including every source file's provenance record.

### Stability

The table has a format version (2). Later versions only add columns (at the end) or measures, so
code written against it keeps working: version 2 added the `author` column and the `note` measure;
columns 1 to 19 are as in version 1.

## ELAN files

One `.eaf` per video, linked to the video where its location is known. Tiers:

| Tier | Annotations |
|---|---|
| `scene_detection scenes` | each scene, labelled with its type |
| `speech_recognition speech` | each transcribed segment |
| `speaker_diarization SPEAKER_00` (one per speaker) | each turn |
| `person_tracking <person> (<label>)` (one per tracked person) | `in view` |
| `face_analysis faces`, `face_openface3_embedding faces` | `in view` (a face is detected) |
| `vlm_annotation labels` | each label, until the next sample |
| `Notes (researcher)` (`ANNOTATOR` = its author; one tier per author when several wrote notes) | each note: a span as written, a moment as an instant |

Pipelines sample frames, so "in view" joins consecutive detections of the same person while the
gap is at most twice the sampling interval, and lasts one interval past the last detection. Numbers
measured per detection (emotions, action units, gaze) stay in the table: ELAN codes what happens
when.

Add your own tiers in ELAN and save; the viewer still opens the file, showing both.

## Methods paragraph

For example (abridged):

> All 40 videos were annotated automatically with VideoAnnotator 1.6.0
> [@addyman2026videoannotator]. Speech was transcribed with
> mobiuslabsgmbh/faster-whisper-large-v3-turbo (revision 0a363e9161cb) at float32
> [@radford2023whisper]. Speaker turns were identified with
> pyannote/speaker-diarization-community-1 (revision 3533c8cf8e36) [@plaquet2023powerset;
> @pyannote2025community1; @bredin2023pyannote]. GPU deterministic mode was off. Settings not
> listed were the pipelines' defaults.

- Citations are `[@key]` keys for the BibTeX that comes with it (R Markdown, Quarto and Pandoc
  resolve them; reference managers import the BibTeX).
- Model revisions are shortened in the prose; the full ones follow in a list.
- Videos processed the same way are described once; differences (a rerun with a newer version, a
  changed setting) are stated with how many videos they apply to.
- It names no video, folder or server, since video names often carry participant codes.
- A VLM prompt is identified by a fingerprint in the prose, with its full text alongside.
- Settings are compared with the defaults in each pipeline's metadata; check that the listed
  settings are what you meant.
