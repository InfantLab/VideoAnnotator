# Implementation Plan: Videos and Projects

**Branch**: `1.7-dev` (spec `029-videos-and-projects`) | **Date**: 2026-10-20 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/029-videos-and-projects/spec.md` (not separately
clarified; its design is spec 023's agreed proposal)

## Summary

The viewer is restructured around videos: one **Videos** page with All videos, Not in a project,
projects and folders on the left, a table of videos with the results each has, and Run…, Export
and Add to project for any folder, project or selection. Saved datasets are shown as projects
and presets as workflows; the API keeps its names. Projects remember their current settings;
workflows are saved and applied only when asked. Jobs, Job detail, Batch detail, Results,
Datasets and Getting started go, their addresses redirecting. The nav becomes Home, Videos,
Ask AI, Pipelines, with an activity indicator, Help and Settings, and help sits where the
question arises.

Approach (from [research.md](research.md)): a video is the group of paths the server can tell
are one file (R1), so reruns of an upload stop looking like new videos. One new request lists
videos per scope with per-pipeline state and the left side's counts, built from a single query
over jobs (R2, R11). A project is a saved dataset whose manifest can now name videos by path,
so it spans folders (R4); it stores its current settings (R5). Last-used settings are stored
per user, and the server decides New run's prefill (R6). Workflows are presets with export
choices (R7). One request starts a run on any readable videos (R8) and one family exports any
set (R10). New run is rebuilt from a target in two steps (R9). A run is a filter of Videos and a
job's details move to the Video page's Runs tab (R12). The local library and the old player go
(R14). A rendered-text test keeps the API's words off the screen (R15); help links are checked
against `docs/` (R16).

## Technical Context

**Language/Version**: Python 3.12/3.13 (server); TypeScript, React 18 (viewer, Vite, Bun)
**Primary Dependencies**: FastAPI, SQLAlchemy, Pydantic (existing); viewer: TanStack Query,
React Router, shadcn/ui (existing). No new dependency.
**Storage**: server SQLite: new columns `saved_datasets.current_settings`,
`saved_pipeline_presets.export_choices` and `.source`; new table `user_preferences`; manifest
entries gain optional `path`. Applied at start-up like specs 018 and 022's columns. Viewer: no
browser storage beyond 028's track preferences.
**Testing**: pytest (unit: video grouping, project membership, run-settings order, run creation,
exports per set, activity; api: each new route and every additive field; a performance test
with 5,000 jobs); Vitest (Videos page, selection actions, New run with targets, run filter,
Runs tab, redirects for every old route, vocabulary, docs links, toasts, no-server states);
manual walk-through [quickstart.md](quickstart.md).
**Target Platform**: the API server on Linux, macOS, Windows; current Chrome, Firefox, Safari.
**Project Type**: web application (Python API + React viewer in one repo).
**Performance Goals**: `GET /videos` under 1 s with 5,000 jobs and 2,000 videos; the Videos
table scrolls smoothly at 2,000 rows (windowed); the activity poll under 100 ms.
**Constraints**: local-first (help links are links to the published docs, nothing loaded from
elsewhere); initial bundle within 300 KB gzipped (Videos and New run are lazy); every existing
route, API field and saved dataset or preset keeps working (Principle V).
**Scale/Scope**: one lab server; hundreds to a few thousand videos; tens of projects and
workflows; a few users.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design: still passes.*

| Gate | Status | How |
|---|---|---|
| **I. Local-first, no telemetry** | ✅ | Everything comes from the user's server; docs are linked, not loaded; no new outbound traffic. Standalone mode and demos work without a server. |
| **II. Stable pipeline contract, open formats** | ✅ | No pipeline or output change. Exports reuse spec 025's writers; a set export is the same table format, with the `run` column it already has. |
| **III. Provenance** | ✅ | No new annotation. Runs keep recording their settings; "What changed?" reads provenance; a project's current settings and a workflow record where they came from. Grouping by `sha256` uses the provenance already recorded. |
| **IV. Modular** | ✅ | No new dependency; no pipeline import in the core. Video grouping and the list live in server modules beside 028's `video_runs`. |
| **V. Backward compatibility** | ✅ | API: new routes and optional fields only; datasets and presets stay as they are and gain fields. `/videos/runs` returns more runs for an uploaded video, in the same shape. Viewer: every old route redirects (R13, tested); old router state shapes are still read; `/viewer`, `/view/:jobId`, `/video` and keys unchanged. Permissions only widen (anyone may edit a project; delete unchanged). |
| **VI. Faithful display** | ✅ | The table shows per-pipeline state from the server, and failed or missing results as such, never hidden; a video whose file is gone stays listed, marked. Grouping never merges by name or size, only where the server can tell (R1). |
| **Engineering standards** | ✅ | Tests per area (above); API typed; docs: a Videos and projects how-to, Help rewritten, API reference for the new routes, CHANGELOG; viewer `tsc`, ESLint, Vitest, bundle check; `viewer_static` rebuilt. |

No violations; Complexity Tracking is empty.

## Project Structure

### Documentation (this feature)

```
specs/029-videos-and-projects/
├── spec.md
├── plan.md              # this file
├── research.md          # what exists; decisions R1–R19
├── data-model.md        # project settings, manifest path, workflow fields, preferences, Video, Run
├── quickstart.md        # walk-through per user story
├── contracts/
│   ├── api.md           # /videos, /videos/run, run-settings, activity, set exports; additive fields
│   └── viewer.md        # nav, routes and redirects, Videos page, New run, words, help
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks (not created here)
```

### Source Code

```
src/videoannotator/
├── video_index.py           # NEW grouping of paths into videos (R1), the list per scope (R2),
│                            #     project membership (R4), runs and activity (R17); cached (R11)
├── video_runs.py            # runs_for over a video's group; projects_for via video_index
├── run_settings.py          # NEW current/last-used settings, prefill order (R5, R6)
├── api/v1/videos.py         # GET /videos, POST /videos/run, GET /videos/run-settings,
│                            #     GET /videos/activity, POST /videos/export/* (or exports.py)
├── api/v1/datasets.py       # current_settings; manifest path; /{id}/videos add/remove; edit by any user
├── api/v1/presets.py        # export_choices, source
├── api/v1/ingest.py, batches.py, datasets.py   # record current/last-used settings on run creation
├── api/v1/exports.py        # set exports (R10)
├── exporters/results_set.py # filter by (job, pipeline)
├── storage/base.py, sqlite_backend.py, file_backend.py   # job_summaries()
└── database/models.py, migrations.py   # columns and user_preferences

viewer/src/
├── pages/Videos.tsx               # NEW (lazy): left side, header, table, selection bar, run filter
├── pages/NewRun.tsx               # NEW (lazy), replacing pages/NewJob.tsx (split into components/run/)
├── pages/Help.tsx                 # replaces GettingStarted.tsx
├── pages/Home.tsx                 # first run and no server only
├── pages/Pipelines.tsx            # + Workflows
├── components/videos/             # NEW Sidebar, ScopeHeader, ProjectHeader, RunHeader, VideosTable,
│                                  #     SelectionBar, AddToProject, NewInFolder
├── components/run/                # NEW ChooseVideos, WhatToFind, StartRun, WorkflowPicker
├── components/video/RunsPanel.tsx # NEW the Runs tab (R12), from JobDetail's parts
├── components/ActivityIndicator.tsx, PageHelp.tsx, Term.tsx, NeedsServer.tsx   # NEW
├── components/AppLayout.tsx       # the new nav
├── components/ExportMenu.tsx      # scopes videos and project
├── hooks/useVideos.ts, useActivity.ts, useRunSettings.ts, useProjects.ts   # NEW
├── lib/docsLinks.ts, glossary.ts, notify.ts, userMessage.ts, runTarget.ts  # NEW
├── App.tsx                        # routes and redirects (R13)
└── deleted: pages/Jobs, JobDetail, BatchDetail, Results, Datasets, GettingStarted, NewJob;
    VideoAnnotationViewer, UnifiedControls, FileViewer, VlmAnnotationPanel, OpenFace3Controls,
    Timeline, useZipDownloader, lib/localLibrary, lib/datasetHandles, DatasetDriftDialog,
    SaveDatasetDialog, PresetBar (once unimported)

tests/
├── unit/test_video_index.py, test_project_membership.py, test_run_settings.py
├── unit/test_video_index_performance.py   # 5,000 jobs, 2,000 videos, under 1 s
├── unit/test_results_set_filter.py
└── api/test_videos_list_api.py, test_videos_run_api.py, test_videos_activity_api.py,
    test_videos_export_api.py, test_datasets_project_fields.py, test_presets_workflow_fields.py

viewer/src/test/
├── vocabulary.test.tsx, docsLinks.test.ts, routes.test.tsx   # SC-002, SC-007, SC-006
└── pages/videos*.test.tsx, newRun*.test.tsx; components/runsPanel.test.tsx, activity.test.tsx

docs/usage/
├── videos_and_projects.md   # NEW how-to (the help links' main target)
├── GETTING_STARTED.md       # rewritten for this structure (Help links here)
└── README.md, video_page.md, exporting_results.md   # words and links updated
```

**Structure Decision**: the existing web-application layout: server code in
`src/videoannotator/`, viewer in `viewer/` built into `viewer_static/` (rebuilt and committed,
CI checks it).

## Order of work

Each stage ships on its own and keeps the suite green; the old pages keep working until stage 5.

1. **Server foundations** (R1, R2, R4, R11): `job_summaries()`, `video_index` (grouping,
   membership, list, counts) with the performance test; `/videos/runs` over the group (fixes the
   Video page for reruns of uploads straight away); `GET /videos`.
2. **Server settings and runs** (R5–R8, R10, R17): the columns and table; current and last-used
   settings recorded on every run route; `/videos/run-settings`; `POST /videos/run`; dataset
   video add/remove and edit by any user; preset fields; set exports; `/videos/activity`.
3. **Viewer foundations** (R15, R16): `userMessage`, `notify`, `glossary`/`Term`, `PageHelp` with
   `docsLinks` and their tests; the vocabulary test (failing pages listed as known until stage
   5, so it ratchets); `NeedsServer`; `useVideos`, `useActivity`, `useRunSettings`.
4. **Videos and New run** (US1, US2, US3, US5): the Videos page (left side, table, selection,
   project header, run filter); New run from targets (R9); the Runs tab (R12); Export scopes;
   the activity indicator and toasts; Home for first run; the root's choice (R18).
5. **Retire and redirect** (R13, R14, US5): routes and redirects with their test; demos through
   `/viewer?demo=`; delete the old pages, the old player and the local library; the nav;
   the vocabulary test with no exceptions.
6. **Workflows and help** (US4, US6, US7): Workflows on Pipelines (save, apply, export, import);
   Help page; docs (`videos_and_projects.md`, Getting started rewritten, links), CHANGELOG,
   roadmap v1.7.0 Phase 1, `CLAUDE.md`; the quickstart walk-through in `checks.md`, with
   screenshots taken in the dev container.

## Risks

- **Grouping mistakes** (R1): joining two different files would merge their results and notes.
  Only same path, same inode, a stored-copy rerun chain, or a recorded `sha256` join; tests
  cover each, and two files with the same name and size stay apart.
- **The list's speed** (R11) on a network share: existence and inode checks stat each path once
  per refresh. If a lab's share is slow, the cache window grows or availability is checked
  lazily; the target is measured on a laptop and in the pilot.
- **New run is the largest rewrite** (1,917 lines today, R9): it is split into components and
  re-tested by target; the upload path (another computer) keeps today's code where it can.
- **Words from the server** (R15): an unmapped error code shows a neutral sentence, which may
  hide a useful detail; the debug panel keeps the raw message.
- **Owner-only datasets in other clients**: widening edits to any signed-in user changes who may
  change a dataset through the API; nothing that worked stops working, but a lab relying on the
  owner rule loses it. Called out in the CHANGELOG; revisit if a pilot lab needs privacy (spec
  Assumptions).
- **Ingest gates** (R3): non-admin or remote users can't browse new folders; they see known
  videos and shared folders' videos, and can upload. Relaxing the gates is a separate decision.
- **Spec 023's journeys** are checked through pilot feedback after release; layout may change
  then, not the server work.

## Complexity Tracking

None.
