# Implementation Plan: Run It Again

**Branch**: `1.6-dev` (spec `019-run-it-again`) | **Date**: 2026-10-04 | **Spec**: [spec.md](spec.md)

## Summary

A finished job or batch can be run again as a new job/batch linked to the original (`rerun_of`),
as is or with changed pipelines/settings, without touching the original. Its settings can be
reused on new videos or saved as a preset. The viewer offers these where the decision is made:
job and batch pages, failed-job messages, a completed batch, and the wizard's first step.

## Technical Context

**Language/Version**: Python 3.12/3.13; TypeScript/React (Bun)
**Primary Dependencies**: existing only
**Storage**: `jobs.rerun_of` (nullable, indexed; additive migration in the storage backend's
`_ensure_batch_columns`); `BatchJob.rerun_of`; file backend serialises it
**Testing**: pytest (API, storage, CLI), Vitest (pages/components), Playwright check
**Project Type**: web application
**Constraints**: originals byte-for-byte unchanged (SC-003); Retry (reset in place) unchanged

## Design decisions (research)

- **The video of a rerun.** A video stored in the original job's folder (an upload) is hard-linked
  (copied where linking fails) into the new job's folder, so deleting either job can't break the
  other. A video outside it (a server-folder ingest) is referenced in place, as ingest does. A
  missing video → 409 `RERUN_VIDEO_MISSING`, and the viewer offers Edit and run again with new
  videos.
- **Endpoints.** `POST /api/v1/jobs/{id}/rerun` with optional `selected_pipelines`, `config`
  (omitted = the original's). Allowed for completed, failed or cancelled jobs (409 otherwise).
  Unknown/unavailable pipelines are validated as on submission. The new job joins a new batch with
  one job (same as any submission). `POST /api/v1/batches/{id}/rerun` (same body) creates a new
  batch named "<name> (rerun)" with one rerun per job, skipping (and reporting) jobs whose video is
  gone.
- **Links.** `JobResponse` gains `rerun_of` and `reruns` (ids, oldest first);
  `StorageBackend.list_reruns(job_id)`.
- **Edit and run again** opens the wizard with the job's settings and a read-only "Videos from
  job <name>" source; Submit calls the rerun endpoint with the edited settings. For a batch, the
  batch rerun endpoint.
- **Use these settings** opens the wizard at Choose videos with pipelines/config filled; **Save as
  preset** posts to the existing presets API.
- **Wizard first step** shows the five most recent finished jobs ("Use settings") and the five
  most recently used presets ("Apply") above the sources.
- **CLI**: `videoannotator job rerun <id> [--pipelines a,b] [--config file] [--api-key]`.

## Constitution Check

| Principle | Assessment |
|---|---|
| I. Local-First | Pass. |
| II. Pipeline Contract | Not touched. |
| III. Provenance | Pass. `rerun_of` links runs; each run keeps its own provenance (spec 017). |
| IV. Modular | Pass. |
| V. Backward Compatibility | Pass. Additive column/fields; Retry unchanged. |
| VI. Faithful Display | N/A. |

## Project Structure

```
src/videoannotator/batch/types.py              # rerun_of
src/videoannotator/storage/{models,sqlite_backend,file_backend,base}.py  # column, list_reruns
src/videoannotator/api/v1/jobs.py              # rerun endpoint, JobResponse fields, shared create_rerun()
src/videoannotator/api/v1/batches.py           # batch rerun
src/videoannotator/cli.py                      # job rerun
viewer/src/components/RunAgainActions.tsx      # actions for job & batch pages
viewer/src/pages/{JobDetail,BatchDetail,NewJob}.tsx
viewer/src/lib/wizardStart.ts                  # typed navigation state for the wizard
tests/api/test_rerun.py, tests/unit/cli/test_cli_rerun.py, viewer tests
```
