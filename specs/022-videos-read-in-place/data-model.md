# Data Model: Videos and Results Where You Expect Them

Changes are additive. Existing rows and files keep their meaning (FR-018).

## Job (`jobs` table, `BatchJob`): no new columns

| Field | Before | After |
|---|---|---|
| `video_path` | Uploads: the copy in the job folder. Ingest: the original. CLI `process`: a hard link or copy in the job folder. | Uploads: unchanged (copy in internal storage). Ingest, dataset runs of in-place videos, CLI `process`: the original, read in place. |
| `output_dir` | Unset at creation; the runner fell back to `storage_path`. | Set at creation to `<results root>/<run folder>/<video folder>`, on every creation path. Unset only on jobs from before this feature. |
| `storage_path` | Job folder holding the video copy and the outputs. | Job folder for internal bookkeeping and, on the upload route only, the uploaded copy. No outputs for new jobs. |
| `batch_id`, `batch_name` | As before. | `batch_name` also feeds the run folder name. |

**Rules**:
- `output_dir` is never inside `storage_path`, and never inside an allowed (video) folder (FR-021).
  This is checked at job creation.
- The results root must not overlap any allowed root. The server refuses to start in-place reading
  with a clear message if the two are configured to overlap.

## Run folder (filesystem; no table)

One per batch, or per unbatched job: `<results root>/<run folder name>/`.

- **Name**: `<run name> (<YYYY-MM-DD>[ N])`. Rules in [research.md](research.md) R2.
- **Created**: exclusively, when the run is created (ingest, dataset run, rerun, upload batch, CLI).
- **Contents**: `run.json`, plus one video folder per job.
- **Lifecycle**: deleting a job removes its video folder. When the run folder holds nothing but
  `run.json`, it is removed with it. Deleting the last job of a run removes the run folder.

### `run.json`

```json
{
  "format": "videoannotator-run",
  "format_version": 1,
  "run": {"batch_id": "…", "name": "BabyJokes wave 2", "created_at": "2026-10-06T10:00:00Z"},
  "videoannotator_version": "1.6.0",
  "pipelines": ["face_analysis", "speech_recognition"],
  "config": {"face_analysis": {"confidence_threshold": 0.7}},
  "videos": [
    {
      "job_id": "…",
      "folder": "4JDccE.joke5.rep3.take1",
      "source": {"kind": "in_place", "path": "/home/ada/Studies/BabyJokes/4JDccE.joke5.rep3.take1.mp4",
                 "size_bytes": 499675},
      "status": "completed",
      "finished_at": "2026-10-06T10:01:12Z",
      "files": ["4JDccE.joke5.rep3.take1_face_detections.json", "…"],
      "models": {"face_analysis": [{"name": "DeepFace detector retinaface", "revision": "…"}]}
    }
  ]
}
```

- `source.kind` is `in_place` or `uploaded`. An uploaded source records `original_filename`,
  never a server path.
- `config` is the run-level configuration. A job edited separately ("Edit and run again") is a new
  run.
- Rewritten atomically (temporary file, then rename) whenever a job's entry changes.
- `models` copies what spec 017 records per pipeline result. Full provenance stays in the output
  files themselves.

## Video folder (filesystem)

`<run folder>/<video folder name>/`: that job's `output_dir`. Holds the pipelines' output files
under their existing names (`<video stem>_<suffix>`) and their `.provenance.json` companions.
Never a video (FR-020).

## Saved dataset (`saved_datasets` table): one new column

| Column | Type | Default | Meaning |
|---|---|---|---|
| `server_selection` | BOOLEAN | `false` | With `server_folder` set: the manifest is the selection. Using the dataset runs exactly those files; files added to the folder are not differences. |

Migration: `ALTER TABLE saved_datasets ADD COLUMN server_selection BOOLEAN DEFAULT 0`, following the
existing saved-dataset column additions in `database/migrations.py`. The API's dataset
request and response models gain `server_selection: bool = False`.

## Server settings (environment or config)

| Setting | Default | Purpose |
|---|---|---|
| `VIDEOANNOTATOR_RESULTS_DIR` | `~/VideoAnnotator` | Results root (FR-019, FR-025). |
| `VIDEOANNOTATOR_INGEST_ROOTS` | Server user's home (unchanged) | Allowed video folders (FR-010). |
| `VIDEOANNOTATOR_PUBLISHED_LOCALLY` | unset | Docker: every caller is on the same machine because the port is published on the host's loopback only (R6). |
| `VIDEOANNOTATOR_HOST_PATHS` | unset | Docker: `container=host` prefix pairs for showing locations (R7). |

## Caller access (computed per request; not stored)

`same_machine`, `can_read_in_place`, `allowed_folders`, `results_root`, `can_open_folders`. See
[contracts/api.md](contracts/api.md).
