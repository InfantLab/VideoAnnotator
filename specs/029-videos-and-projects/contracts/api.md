# API contract: Videos and Projects

**Spec**: [../spec.md](../spec.md) | **Data model**: [../data-model.md](../data-model.md)

All routes are under `/api/v1`, need the caller's API key when the server requires sign-in, and
answer errors in the existing envelope (`{error: {code, message, hint}}`). Every change to an
existing route is additive: new optional fields and parameters only (Principle V).

## New routes

### `GET /videos`

The videos the server knows, for one scope (R2).

Query: one of `project=<dataset id>`, `folder=<path>` (+ `recursive=true`), `not_in_project=true`,
`run=<batch id or job id>`; none means all. `q=` filters names (case-insensitive substring).
`offset=0`, `limit=500` (max 2,000).

```json
{
  "scope": {"kind": "project", "id": "…", "name": "Peekaboo study"},
  "videos": [
    {
      "path": "/data/pk/p01_t1.mp4",
      "paths": ["/data/pk/p01_t1.mp4"],
      "name": "p01_t1.mp4",
      "folder": "/data/pk",
      "available": true,
      "size_bytes": 148000000,
      "projects": [{"id": "…", "name": "Peekaboo study"}],
      "results": {
        "face_analysis": {"state": "done", "run_id": "b1", "finished_at": "…"},
        "vlm_annotation": {"state": "running", "run_id": "b2", "finished_at": null}
      },
      "last_run_at": "…"
    }
  ],
  "total": 48,
  "counts": {
    "all": 212,
    "not_in_project": 37,
    "projects": [{"id": "…", "name": "Peekaboo study", "owner_name": "caspar", "count": 48}],
    "folders": [{"path": "/data/pk", "label": "/data/pk", "count": 48, "uploads": false}]
  },
  "new_in_folder": [{"path": "/data/pk/p49_t1.mp4", "name": "p49_t1.mp4"}],
  "runs": [ { "id": "b2", "name": "…", "total": 48, "done": 31, "failed": 0, "running": 2,
              "waiting": 15, "cancelled": 0, "started_at": "…", "finished_at": null,
              "pipelines": ["vlm_annotation"], "project": {"id": "…", "name": "…"} } ]
}
```

- `results` lists only pipelines that ever ran on the video; `state` is `done`, `running`,
  `waiting` or `failed`, from the newest run of that pipeline that finished, else the newest.
- `counts.projects[].owner_name` is present for every project; the viewer shows it when two
  names clash.
- `new_in_folder` only for a project that is a selection of a folder (R4); else `[]`.
- `runs`: for `project=` its runs newest first; for `run=` that one run; else `[]`.
- `subfolders` (added in the walk-through): for a folder, the folders directly inside it with
  videos, `[{path, name}]`, so a shared folder's subfolders can be opened from Videos.
- `videos[].run_job_id`: in a run's list, the video's job in that run (the Video page opens on it).
- `results[*].error`: why a pipeline failed or stopped.
- A folder outside the shared folders and the uploads folder: 403 `PATH_NOT_ALLOWED` (as
  `readable_video`). Unknown project or run: 404 `DATASET_NOT_FOUND` / `RUN_NOT_FOUND`.
- Performance: under 1 s with 5,000 jobs and 2,000 videos (R11).

### `POST /videos/run`

One run on any videos the server can read (R8).

```json
{
  "paths": ["/data/pk/p01_t1.mp4", "/srv/va/storage/jobs/…/clip.mp4"],
  "selected_pipelines": ["face_analysis"],
  "config": {"face_analysis": {"min_face_size": 40}},
  "name": "Peekaboo faces",
  "project_id": "…"
}
```

→ 201

```json
{"run_id": "<batch id>", "name": "Peekaboo faces", "created": ["<job id>", "…"],
 "skipped": [{"path": "…", "reason": "No video at '…': it may have been moved or deleted."}],
 "results_folder": {"path": "…", "display_path": "…"}}
```

- Pipelines and settings are validated as `POST /jobs` does (422 on an unknown pipeline or bad
  setting). Every path unusable: 422 `NOTHING_TO_RUN` with the reasons.
- A shared-folder video is read in place; a stored upload is hard-linked (copied if linking
  fails) into the new job's folder.
- Records `project_id` as the jobs' `dataset_id`, sets the project's `current_settings`
  (`source.kind = run`) and the caller's `last_run_settings`.

### `GET /videos/run-settings`

Query: `project=<id>`, `workflow=<preset id>` (both optional).

```json
{"selected_pipelines": ["…"], "config": {},
 "source": {"kind": "project", "name": "Peekaboo study", "changed_at": "2026-10-12T10:03:00Z"}}
```

`source.kind`: `workflow`, `project`, `last_used`, `defaults`, the first that has settings in
that order (FR-012).

### `GET /videos/activity`

```json
{"active": [{"id": "b2", "name": "…", "total": 48, "done": 31, "failed": 0, "running": 2,
             "waiting": 15, "project": {"id": "…", "name": "…"}, "started_at": "…"}],
 "recent": [{"id": "b1", "name": "…", "total": 48, "done": 48, "failed": 1,
             "finished_at": "…", "stopped": false}]}
```

`active`: runs with a video waiting or running. `recent`: up to 5 runs finished in the last
24 hours (so the viewer can raise a finished toast after a reload).

### `POST /videos/export/table`, `/videos/export/elan`, `/videos/export/methods`

```json
{"paths": ["…"], "project_id": null, "run_id": null, "results": "latest", "format": "csv"}
```

Exactly one of `paths`, `project_id`, `run_id`. `results`: `latest` (per video and pipeline, the
newest completed result) or `all` (every run; the `run` column tells them apart). `format`:
table `csv` | `parquet`; methods `json` | `markdown`; ELAN ignores it (one `.eaf`, or a zip of
one per video). Responses, summary header and errors as spec 025's routes
(`X-VideoAnnotator-Export-Summary`, 409 `NOTHING_TO_EXPORT`, 422 `PARQUET_NOT_INSTALLED`).

## Changed routes (additive)

| Route | Change |
|---|---|
| `GET /videos/runs` | The runs of every path of the video's group (R1); new field `paths` |
| `GET/POST/PUT /datasets…` | `current_settings` in responses; `PUT` accepts it; manifest entries accept and return `path` |
| `PUT /datasets/{id}` | Any signed-in user may change `name`, `description`, `video_manifest`, `server_selection`, `current_settings`; delete stays owner or admin |
| `POST /datasets/{id}/videos` (new, convenience) | `{"paths": [...]}` adds videos (manifest entries with `path`); a project that followed its folder becomes a selection of it plus these (R4). Returns the dataset |
| `DELETE /datasets/{id}/videos` (new, convenience) | `{"paths": [...]}` removes those entries; never touches files or results |
| `GET/POST/PUT /presets…` | `export_choices` and `source` in requests and responses; export/import JSON includes them |
| `POST /ingest`, `POST /datasets/{id}/run`, `POST /batches/{id}/rerun` | Also set `current_settings` and `last_run_settings` (no response change) |

## Saving without sign-in

On a server without sign-in (`AUTH_REQUIRED=false`) and no key, saved datasets and presets are
owned by a local `anonymous` user (created on first use, not counted as a user by
`generate-token`). With sign-in required, saving without an identity is still 400
`OWNER_IDENTITY_REQUIRED`.

## Unchanged

`/jobs…`, `/batches…` (beyond the above), `/ingest/access|browse|scan` (gates unchanged),
`/exports` routes of spec 025, `/videos` upload (`POST /videos`), notes and streaming (028),
`/ask…` (027). The CLI is unchanged; `videoannotator` commands keep their words.
