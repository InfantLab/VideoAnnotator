# Data Model: Videos and Projects

**Spec**: [spec.md](spec.md) | **Research**: [research.md](research.md) | **Date**: 2026-10-20

Every change to stored data is additive (Principle V): new nullable columns, one new table, one
optional field in a JSON list. No existing column changes meaning.

## Stored on the server

### `saved_datasets` (a project): one new column

| Column | Type | New | Meaning |
|---|---|---|---|
| `current_settings` | JSON, nullable | ✓ | The settings a new run on the project starts from (R5); null until a run starts or someone edits them |

`current_settings` shape:

```json
{
  "selected_pipelines": ["face_analysis", "speech_recognition"],
  "config": {"face_analysis": {"min_face_size": 40}},
  "source": {"kind": "run", "id": "<batch or job id>", "name": "Peekaboo study (2026-10-12)"},
  "updated_at": "2026-10-12T10:03:00Z",
  "updated_by": "<user id or 'anonymous'>"
}
```

`source.kind`: `run` (set when a run started on the project), `edit` (changed in the project's
header), `workflow` (copied from a workflow; `id` and `name` are the preset's).

Written by: every route that creates jobs with a `dataset_id` (`POST /videos/run`, `POST /ingest`,
`POST /datasets/{id}/run`, `POST /batches/{id}/rerun` when its jobs carry one), and
`PUT /datasets/{id}` with `current_settings`.

### `saved_datasets.video_manifest` entries: one optional field

| Field | Type | New | Meaning |
|---|---|---|---|
| `filename` | str | | As before |
| `size_bytes` | int | | As before |
| `last_seen_at` | datetime? | | As before |
| `relative_path` | str? | | As before: within `server_folder` |
| `path` | str? | ✓ | The video's resolved server path, for a video added from anywhere (R4) |

### Project membership (computed, R4)

| Dataset | Holds |
|---|---|
| `server_folder`, `server_selection` false ("follows the folder") | Every video in the folder (recursively if `server_folder_recursive`), including new ones |
| `server_folder`, `server_selection` true | Entries with `relative_path` (joined to the folder) or `path`; folder videos not in it are "new in the folder" (offered) |
| no `server_folder` (uploads) | Entries with `path`; the videos of jobs with this `dataset_id`; stored copies matched by filename and size (as `/stored-videos`) |

A video (R1, a group of paths) is in a project when any of its paths matches.

### `saved_pipeline_presets` (a workflow): two new columns

| Column | Type | New | Meaning |
|---|---|---|---|
| `export_choices` | JSON, nullable | ✓ | `{"table": "csv" \| "parquet", "elan": bool, "methods": bool}`; null means the defaults (CSV table, ELAN and methods offered) |
| `source` | JSON, nullable | ✓ | `{"kind": "project" \| "run", "id": "…", "name": "…"}`: what it was saved from |

`selected_pipelines` and `config` are unchanged; Ask AI questions and sampling are
`config.vlm_annotation`.

### `user_preferences` (new table)

| Column | Type | Meaning |
|---|---|---|
| `user_key` | str(64), PK part | The user's id, or `anonymous` when the server runs without sign-in |
| `name` | str(100), PK part | `last_run_settings` (the only one in 029) |
| `value` | JSON | For `last_run_settings`: `{selected_pipelines, config, source, updated_at}` as above, `source.kind` = `run` |
| `updated_at` | datetime | |

### Jobs: unchanged

A **run** is still the jobs sharing a `batch_id`, or one job without one. Nothing is added to
`jobs` or `pipeline_results`. `job_summaries()` (new, `StorageBackend`) reads per job: id,
`video_path`, status, `created_at`, `completed_at`, `batch_id`, `batch_name`, `dataset_id`,
`rerun_of`, `storage_path`, and per pipeline its status and error, in one query; the provenance
`sha256` is read from `pipeline_results.provenance` (R1).

## Computed on the server

### Video (R1)

| Field | Meaning |
|---|---|
| `path` | Its newest existing path (the address to open it by) |
| `paths` | Every path of the group, newest first |
| `name` | File name of `path` |
| `folder` | The folder of `path`, or `uploads` for a stored upload |
| `available` | `path` exists |
| `size_bytes` | From the file, else the newest job's record |
| `projects` | `[{id, name}]` (R4) |
| `results` | Per pipeline that ever ran on it: `{state, run_id, finished_at}`, `state` one of `done`, `running`, `waiting`, `failed`; from its newest run of that pipeline that finished, else the newest |
| `last_run_at` | When its newest run started |

Grouping keys (any one joins two paths): same resolved path; same `(st_dev, st_ino)`; a job's
`rerun_of` chain where the newer job's video is a stored copy; the same provenance `sha256`.

### Run (as listed in a project's header, a filter, and the activity indicator)

| Field | Meaning |
|---|---|
| `id` | `batch_id`, or the job id for a run of one video without one |
| `name` | `batch_name`, else the run folder's name (spec 022), else "Run of <video>" |
| `project` | `{id, name}` from its jobs' `dataset_id`, if any |
| `total`, `done`, `failed`, `running`, `waiting`, `cancelled` | Counts of its videos (jobs) |
| `started_at`, `finished_at` | First job created, last job finished |
| `pipelines` | The union of its jobs' pipelines |

States: `waiting` → `running` → `done` (all videos finished, some may have failed pipelines) |
`stopped` (cancelled). A run is **active** while any video is waiting or running.

### Run settings prefill (R6)

`{selected_pipelines, config, source: {kind: workflow | project | last_used | defaults, name?,
changed_at?}}`, chosen in that order.

## In the viewer

### Videos page state (the address)

`/videos[?project=<id> | folder=<path> | not_in_project=1 | run=<id>][&q=<search>][&sel=…]`

- one scope at a time; none means All videos;
- `q` filters by name; the selection is page state, not in the address;
- opening a video passes the scope on as 028's `from=` (`project:`, `folder:`, `run:`; new:
  `all`, `unassigned`), so ‹ › move through the same list.

### RunTarget (what New run is opened on, R9)

```ts
type RunTarget =
  | { kind: 'videos'; paths: string[]; label: string }   // a selection, or one video
  | { kind: 'project'; id: string; name: string }
  | { kind: 'folder'; path: string; recursive: boolean }
  | { kind: 'rerun'; runId: string; label: string }       // "Run again…", today's rerun modes
  | null;                                                   // choose videos (upload, folder, project)
```

Passed as router state (as `WizardStart` is now, which it replaces; the old shapes are still
read, R13).

### Words

| Viewer says | API keeps |
|---|---|
| video | job's `video_path`, upload |
| run | batch, job (a run of one video) |
| project | dataset |
| workflow | preset |
| track | (028's lanes) |
| results table | export table |
