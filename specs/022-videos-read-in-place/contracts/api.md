# API Contract: Videos and Results Where You Expect Them

All changes are additive, except the job artifacts zip's default (marked **changed**). Existing
requests keep working (FR-018).

## New: `GET /api/v1/ingest/access`

What this caller may do. Computed per request; any authenticated caller may ask.

```json
{
  "same_machine": true,
  "can_read_in_place": true,
  "reason": null,
  "allowed_folders": [{"path": "/home/ada", "display_path": "/home/ada"}],
  "results_root": {"path": "/home/ada/VideoAnnotator", "display_path": "/home/ada/VideoAnnotator"},
  "can_open_folders": true
}
```

- **`same_machine`**: the caller's address is loopback, or the server has
  `VIDEOANNOTATOR_PUBLISHED_LOCALLY` set.
- **`can_read_in_place`**: `same_machine`, the caller is an administrator, and at least one allowed
  folder exists.
- **`reason`**: why `can_read_in_place` is false, written for researchers. For example: "This
  server is on another computer", "Only an administrator can read videos in place", or "No video
  folder is set up. Set VIDEOS_DIR when starting Docker (see docs)".
- **`display_path`**: the host path under Docker (`VIDEOANNOTATOR_HOST_PATHS`); otherwise the same
  as `path`.
- **`can_open_folders`**: `same_machine` and the server has a desktop to open folders on (R11).

## Changed (additive): `POST /api/v1/ingest`

New optional field:

```json
{"path": "/home/ada/Studies/BabyJokes", "files": ["child01.mp4", "site_a/child02.mp4"], "...": "..."}
```

- **`files`**: paths relative to `path`. When present, only these become jobs; `recursive` is
  ignored. Each is resolved and must lie inside an allowed folder; otherwise it is reported in
  `skipped` with the reason. Duplicates (by resolved path) are dropped.
- **Before creating any job**, the server checks it can write to the results root. If it can't:
  422 `RESULTS_DIR_UNWRITABLE`, with the folder in the message (FR-026).
- **Each created job** gets `output_dir` in the new run folder (data-model.md).
- **The response** gains `results_folder: {"path", "display_path"}`, the run folder.

The same results-folder behaviour applies to job upload (`POST /api/v1/jobs/`), dataset run
(`POST /api/v1/datasets/{id}/run`), job and batch rerun, and the CLI's `process`. Their responses
gain `results_folder` where they describe a run.

## Changed (additive): job and batch responses

- `JobResponse` gains `results_folder: {"path", "display_path"} | null`: the job's `output_dir`, or
  null for jobs from before this feature. That folder may still exist, but it is not laid out by
  run.
- `BatchSummary` gains `results_folder: {"path", "display_path"} | null`: the run folder.
- `JobResponse` gains `video_available: bool`: whether `video_path` exists now (FR-014).

## New: `POST /api/v1/results/open`

```json
{"path": "/home/ada/VideoAnnotator/BabyJokes wave 2 (2026-10-06)"}
```

- **Who may call it**: same-machine callers only. Otherwise 403 `NOT_SAME_MACHINE`.
- **What it may open**: `path` must resolve inside the results root. Otherwise 422
  `PATH_OUTSIDE_RESULTS`.
- **Success**: 204, after asking the OS file manager to open the folder.
- **No desktop to open it on**: 409 `OPEN_FOLDER_UNSUPPORTED`. The viewer then shows the location
  with a copy button.

## New: `GET /api/v1/batches/{batch_id}/results.zip`

Streams the run's results in their on-disk layout (`run.json`, one folder per video), never videos
(FR-028). For runs from before this feature, it builds the same layout from each job's folder. 404
`BATCH_NOT_FOUND` for an unknown batch.

## New: `DELETE /api/v1/batches/{batch_id}`

Deletes every job in the run (as `DELETE /jobs/{id}`), their video folders, and the run folder.
Original videos are never touched (FR-030). Running jobs are cancelled first. 204 on success, 404
`BATCH_NOT_FOUND` for an unknown batch.

## Changed: `GET /api/v1/jobs/{job_id}/artifacts`

New query parameter `include_video` (default **false**). Before, the video was always included.
`include_video=true` restores that. This user-facing change goes in the CHANGELOG (FR-029). When
the video is left out, the response carries
`X-VideoAnnotator-Notice: video excluded; use include_video=true`, so scripts that relied on it
can notice.

## New: `POST /api/v1/batches/{batch_id}/rerun?check=true`

A dry run: returns the same `BatchRerunResponse` shape with `created: []`, and `skipped` listing
every job that couldn't be rerun, such as a video no longer at its location. It creates nothing.
The viewer calls it before a rerun to show missing videos first (FR-015).

## Changed (additive): `POST /api/v1/batches/{batch_id}/rerun` relocation

New optional query parameters `relocate_folder` (a folder inside the allowed folders) and
`recursive` (default false). For each job whose video is missing, the server looks in that folder
(and its subfolders when `recursive`) for a file with the same name **and** size. Matched jobs
rerun from the new path; unmatched ones stay in `skipped`. The response gains
`relocated: [{"job_id", "from", "to"}]`. With `check=true` it reports the matches and creates
nothing. A `relocate_folder` outside the allowed folders gives 422 (FR-015).

## Changed (additive): saved datasets

`DatasetCreateRequest`, `DatasetUpdateRequest` and `DatasetResponse` gain
`server_selection: bool = false` (data-model.md).

## Job failure message (runner)

A job whose `video_path` is missing when it starts fails with:

`Video not found: <display path> (moved or deleted since the job was created)`

The rest of the run continues (FR-016).
