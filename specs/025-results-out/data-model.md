# Data Model: Results Out

No database changes. Everything here is in memory or in an export file. Storage backends gain one
query, `list_jobs_by_dataset(dataset_id) -> list[str]` (the `dataset_id` column exists since 008).

## ResultsSet

What an export reads: an ordered list of `VideoResults`.

| Field | Type | Notes |
|---|---|---|
| `videos` | list[VideoResults] | sorted by (`run`, `video`) |
| `missing` | list[MissingVideo] | videos with no result files: `run`, `video`, `reason` |

Built two ways (both end in `VideoResults` with the same fields):

- **From a folder** (`results_set.from_folder(path)`): walks `path`. A folder holding `run.json` is
  a run; its subfolders are its videos (`run` = the run folder's name, `video` = the subfolder's
  name; `run.json`'s `videos[].status`/`error` give `missing` reasons). A folder whose parent holds
  `run.json` is one video of that run. Any other folder with result files directly in it is one
  "video group": its files are grouped by video stem (`video` = stem; `run` = empty for the folder
  exported, else that folder's path relative to it, so two loose folders holding the same video's
  results stay apart). Subfolders are walked recursively, so a folder of runs works.
- **From jobs** (`results_set.from_jobs(jobs)`): per job, `job_folder(job)`; `run` = the run folder's
  name if `run_folder_of(job)`, else `job.batch_name` or empty; `video` = the video folder's name in a
  run layout, else the video's stem. Jobs not completed and without files → `missing` with
  `job.error_message` or the status ("still running", "cancelled").

## VideoResults

| Field | Type | Notes |
|---|---|---|
| `run` | str | may be empty |
| `video` | str | unique within `run` |
| `files` | list[ResultFile] | recognised result files, by suffix order of the reader table |
| `skipped` | list[SkippedFile] | `path`, `reason` (`no reader for this file`, `unreadable: <error>`) |
| `media_path` | str \| None | the video's recorded path (run.json `source.path`), for ELAN |
| `duration` | float \| None | from provenance-free sources only when recorded; else None |

## ResultFile

| Field | Type | Notes |
|---|---|---|
| `path` | Path | |
| `kind` | str | reader key: `person`, `face_deepface`, `face_openface3`, `scene`, `vlm`, `rttm`, `webvtt` |
| `pipeline` | str | provenance `pipeline.name`, else the reader table's default |
| `provenance` | dict \| None | `provenance.read_record(path)` (spec 017) |

## Row (one table row)

The 19 columns of `contracts/table.md`, in order. Types: `start`, `end`, `value`, `confidence`, `x`,
`y`, `width`, `height` float or None; `event`, `frame` int or None; the rest str (empty when
missing, never None in CSV).

Invariants:

- `start` is not None for every row.
- `end` is None or `end >= start`.
- `label` and `value` are not both None, except on a `face` / `person` row with a box.
- Rows of one event share `run`, `video`, `pipeline`, `event`, `start`, `end`, `frame`, `track`.

## ExportSummary

Returned by every export (CLI prints it; API sends it in the `X-VideoAnnotator-Export-Summary`
header and, for Parquet, in the file's metadata).

```json
{
  "table_format_version": 1,
  "rows": 1284,
  "videos": 12,
  "pipelines": ["person_tracking", "speech_recognition"],
  "skipped": [{"run": "BabyJokes (2026-10-06)", "video": "clip3", "file": "notes.txt",
               "reason": "no reader for this file"}],
  "missing": [{"run": "BabyJokes (2026-10-06)", "video": "clip7",
               "reason": "failed: No audio could be extracted"}],
  "left_out": ["continuous measures (action units, gaze, head pose, scores, age) are in the table, not in ELAN"],
  "truncated": false
}
```

`left_out` is used by ELAN only. Parquet's metadata adds `sources`: one entry per file,
`{run, video, file, provenance}`.

## Citation (pipeline metadata, additive)

`PipelineMetadata.citations: list[Citation]`, default empty.

| Field | Type | Notes |
|---|---|---|
| `key` | str | BibTeX key, unique across shipped pipelines (`radford2023whisper`) |
| `bibtex` | str | one complete entry whose key equals `key` |

Validation (unit test over every shipped YAML): each `bibtex` parses as one `@type{key, ...}` entry
with balanced braces and the same key; every shipped pipeline has at least one citation.

## MethodsDescription

| Field | Type | Notes |
|---|---|---|
| `prose` | str | Markdown paragraphs; citation keys as `[@key]` (Pandoc style, which R Markdown, Quarto and most reference managers' Markdown plugins resolve) |
| `bibtex` | str | every entry cited in `prose`, VideoAnnotator's first, de-duplicated by key |
| `prompts` | list[{sha256, text}] | VLM prompts used, in order of first use |
| `summary` | {videos, groups, not_recorded} | counts; `not_recorded` lists pipelines with no provenance |

### Signature (grouping, FR-025)

Per video, per pipeline: `(videoannotator_version, ((model name, revision), ...), differing
settings as sorted items, deterministic)`. Videos with identical signatures for every pipeline form
one group. Groups sorted by size, largest first.
