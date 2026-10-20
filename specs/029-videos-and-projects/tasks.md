---
description: "Tasks for spec 029: Videos and Projects"
---

# Tasks: Videos and Projects

**Input**: `specs/029-videos-and-projects/` (plan.md, spec.md, research.md, data-model.md,
contracts/, quickstart.md)
**Tests**: included. The constitution requires tests for new API surface; SC-002, SC-006 and
SC-007 are checked by tests (vocabulary, routes, docs links); R11's speed by a performance test.
**Organization**: by user story, in the spec's priority order (US1, US2, US3, US5 are P1; US4,
US6, US7 are P2). Within a story, server before viewer, tests beside the code they test.

**Conventions for every task**: Python passes `ruff check`, `ruff format`, `mypy src/videoannotator`;
viewer passes `bun run lint`, `bun run typecheck`, `bun run test:run`. No comments saying *what*
code does. API changes are additive only (Principle V): new routes, new optional fields. User-visible
words follow spec 023 Decision 1: video, folder, run, project, workflow, track; never job, batch,
dataset, preset. Any change under `viewer/src/` is committed with a rebuilt `viewer_static/`
(`bash scripts/build_viewer.sh`). After `src/videoannotator/api` changes, regenerate
`viewer/src/api/schema.d.ts` (`bash scripts/gen_viewer_api_types.sh`). Commit per task or small
group, `<area>: <summary>`, with the Co-Authored-By trailer; the container has no git identity
(`git -c "user.name=Caspar Addyman" -c user.email=infantologist@gmail.com commit …`). Check UI
changes with a screenshot (Chromium for Playwright is in the dev container) before ticking them.

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Create `tests/fixtures/video_index.py`: helpers that build a temporary jobs database through the real `SQLiteStorageBackend` (`make_job(video_path, *, batch_id=None, dataset_id=None, rerun_of=None, pipelines={name: (status, error, sha256)}, created_at=…)`), write small video files (bytes from `tests/fixtures/synthetic_video.py`), hard-link copies (`os.link`) for reruns, and make saved datasets and presets through `database.crud`; used by every server test in this spec
- [X] T002 [P] Create `viewer/src/test/fixtures/videos.ts`: typed sample responses for `GET /videos` (all, project, folder, run scopes; a missing file; a failed pipeline; a video in two projects), `GET /videos/activity`, `GET /videos/run-settings`, and datasets/presets with the new fields, matching `contracts/api.md`; and a `mockVideosApi()` helper beside `src/test/videoPageHarness.tsx` that stubs `apiClient` methods with them

---

## Phase 2: Foundational (blocks every story)

### Server: storage

- [X] T003 Add to `src/videoannotator/database/models.py`: `SavedDataset.current_settings` (JSON, nullable), `SavedPipelinePreset.export_choices` and `.source` (JSON, nullable), and a `UserPreference` model (`user_preferences`: `user_key` String(64) + `name` String(100) composite primary key, `value` JSON not null, `updated_at` DateTime); in `src/videoannotator/database/migrations.py` add the three columns when missing (the spec 018/022 `ALTER TABLE … ADD COLUMN` pattern in `migrate_to_v1_3_0`); the new table comes from `create_tables()`. Test in `tests/unit/test_migrations_029.py`: an old-schema SQLite file gains the columns and table, existing rows untouched
- [X] T004 [P] Add `job_summaries()` to `src/videoannotator/storage/base.py` (default: loads every job, returning lightweight `JobSummary` dataclasses: job_id, video_path, status, created_at, completed_at, batch_id, batch_name, dataset_id, rerun_of, storage_path, output_dir, selected_pipelines, config, and `pipelines: dict[name, (status, error, sha256)]` with sha256 from provenance `input.sha256`), and an override in `src/videoannotator/storage/sqlite_backend.py` that reads jobs and `pipeline_results` in one query each (no per-job load). Test both backends return the same in `tests/unit/test_job_summaries.py`

### Server: videos as groups of paths (R1, R2, R11)

- [X] T005 Create `src/videoannotator/video_index.py` with `build_index(summaries, extra_paths=()) -> VideoIndex`: resolves each distinct path once, stats it (exists, size, `(st_dev, st_ino)`), and joins paths with a union-find on: same resolved path; same `(dev, ino)`; a job's `rerun_of` whose video was a stored copy (inside the original job's `storage_path`); the same provenance sha256. `VideoIndex.video_for(path)` returns the group (`Video`: `path` = newest existing member, `paths` newest first, `name`, `folder` (or `uploads` for paths under the storage root's jobs and videos folders), `available`, `size_bytes`, `jobs`); `VideoIndex.videos()` lists them. Tests in `tests/unit/test_video_index.py`: an upload and its hard-linked rerun are one video; after deleting the first copy they still are (rerun chain); two uploads with the same recorded sha256 are one; two different files with the same name and size are two; a symlink and its target are one
- [X] T006 Add per-pipeline result state to `video_index.py`: `Video.results` per pipeline that ever ran on any of its jobs: `{state, run_id, finished_at}` from the newest job whose pipeline completed, else the newest job (`state`: done, running, waiting, failed; `run_id` = batch_id or job id), plus `last_run_at`. Add `cached_index(storage)` that rebuilds only when the jobs' row count or newest `updated_at`/`created_at` changes, or after 5 s. Tests in `tests/unit/test_video_index.py`
- [X] T007 [P] Create `tests/unit/test_video_index_performance.py`: 5,000 jobs over 2,000 video files (with reruns and batches) via T001's helpers; `build_index` + listing all videos under 1 s (marked `@pytest.mark.slow` only if over 2 s on CI hardware; record the measured time in the test's output)
- [X] T008 Make `src/videoannotator/video_runs.py` use the group: `runs_for(storage, path)` returns the runs of every path of `video_for(path)` (newest first); `VideoRunsResponse` gains `paths`; `api/v1/videos.py`'s `_video_for_page` accepts any member path. Update `tests/api/test_video_runs_api.py`: the Video page of an upload lists its rerun's run (the 2026-10-20 bug)

### Server: runs and project membership helpers

- [X] T009 Add to `video_index.py`: `runs(index)` grouping jobs into runs (`id` = batch_id, or job id when none; `name` = batch_name, else the run folder's name via `results_folder.run_folder_of`, else "Run of <video name>"; counts total/done/failed/running/waiting/cancelled; started/finished; pipelines; project from `dataset_id`), and `run_videos(index, run_id)`. Tests in `tests/unit/test_video_index.py`
- [X] T010 Create `src/videoannotator/projects.py`: `members(dataset, index, storage) -> set[video key]` per data-model.md's membership table (follows the folder: every video file in `server_folder` (recursive if set), listed with `ingest`'s video-extension rule; selection: manifest entries by `relative_path` joined to the folder or by `path`; uploads: entries by `path`, jobs with this `dataset_id`, and stored copies matched by filename and size as `datasets._stored_copies` does), `new_in_folder(dataset, index)` (selection projects only), and `projects_of(index, datasets)` (video → `[{id, name}]`). Replace `video_runs.projects_for` with it. Tests in `tests/unit/test_project_membership.py` for each case, including a video in two projects and a project spanning two folders

### Viewer: shared pieces

- [X] T011 [P] Create `viewer/src/lib/userMessage.ts`: `userMessage(error)` maps the API error codes these pages meet (`JOB_NOT_FOUND`, `BATCH_NOT_FOUND`, `DATASET_NOT_FOUND`, `NOT_DATASET_OWNER`, `DATASET_NOT_STORED`, `NOTHING_TO_EXPORT`, `NOTHING_TO_RUN`, `PATH_NOT_ALLOWED`, `VIDEO_NOT_FOUND`, `RERUN_VIDEO_MISSING`, `RUN_NOT_FOUND`, `OWNER_IDENTITY_REQUIRED`, network errors) to sentences in the viewer's words, and falls back to the server's message only when it contains none of job/batch/dataset/preset, else to "The server couldn't do that." with the code; test in `viewer/src/test/lib/userMessage.test.ts`
- [X] T012 [P] Create `viewer/src/lib/notify.ts`: `notify({title, body?, action?: {label, to | onClick}, tone?})` over the existing sonner toaster (`components/ui/sonner`), so every toast with a next step looks the same; test in `viewer/src/test/lib/notify.test.tsx`
- [X] T013 [P] Create `viewer/src/lib/glossary.ts` (video, folder, run, project, workflow, track, results table: one plain sentence each, from spec 023's vocabulary) and `viewer/src/components/Term.tsx` (the word with a dotted underline and a `Tooltip` from `components/ui/tooltip`, focusable); test in `viewer/src/test/components/term.test.tsx`
- [X] T014 [P] Create `viewer/src/lib/docsLinks.ts` (`DOCS_BASE = 'https://infantlab.github.io/VideoAnnotator/'`; a map page → `{file: 'usage/videos_and_projects.md', anchor: 'projects'}`; `docsUrl(page)` turning `usage/x.md#a` into `usage/x/#a` as MkDocs with directory URLs does) and `viewer/src/components/PageHelp.tsx` (one sentence and **More in the docs**, opening in a new tab); test `viewer/src/test/docsLinks.test.ts` reads each target under `docs/` (via `fs` in Vitest's node context) and checks the anchor equals the MkDocs slug of a heading (lower-case, punctuation removed, spaces to hyphens)
- [X] T015 [P] Create `viewer/src/components/NeedsServer.tsx`: what this page needs a server for (a prop), **Connect** (to `/settings`), and what works without one: **Open files from this computer** (`/viewer`) and **Demos** (`/viewer?demo=…`, R14); test in `viewer/src/test/components/needsServer.test.tsx`
- [X] T016 Create `viewer/src/test/vocabulary.test.tsx`: renders each routed page (with `mockVideosApi()` and the existing mocks) in its loaded, empty and error states, collects visible text plus `title`, `aria-label` and `placeholder` attributes, and fails on `/\b(jobs?|batch(es)?|datasets?|presets?)\b/i`. Pages still to be replaced are listed in a `KNOWN` set that the test also checks is not stale (a listed page that now passes fails the test, so the list only shrinks); start with every current page that fails
- [X] T017 [P] Add API types and client methods in `viewer/src/api/client.ts` (and `viewer/src/types/videos.ts`): `listVideos(scope, q?)`, `runVideos(body)`, `getRunSettings({project?, workflow?})`, `getActivity()`, `exportVideos(kind, body)` returning a `Blob` and the summary header, `addProjectVideos(id, paths)`, `removeProjectVideos(id, paths)`; dataset and preset types gain `current_settings`, manifest `path`, `export_choices`, `source`; `ExportScope` gains `{kind: 'videos', paths}` and `{kind: 'project', id}`. Regenerate `schema.d.ts` after T020–T024 land; tests in `viewer/src/test/api/videosClient.test.ts` with mocked `fetch`
- [X] T018 [P] Create hooks `viewer/src/hooks/useVideos.ts` (TanStack Query on `listVideos`, key per scope and filter, `staleTime` 5 s, refetch every 5 s while any shown video has a running or waiting result) and `viewer/src/hooks/useActivity.ts` (polls `getActivity` every 5 s while anything is active, else 30 s; exposes `active`, `recent`, and transitions since the last poll: started, finished, failed, stopped); tests in `viewer/src/test/hooks/useActivity.test.tsx` for the transitions

**Checkpoint**: the index, membership and shared viewer pieces exist and are tested; the old pages still work; the Video page now shows reruns of uploads.

---

## Phase 3: User Story 1 – Run pipelines on a folder of videos and get a table (P1) 🎯 MVP

**Goal**: pick a folder on Videos, Run…, keep defaults, start; progress in the header and the
activity indicator; when done, "N videos → results table" with ELAN and methods beside it.
**Independent test**: quickstart US1 on a folder of three videos, without meeting job, batch or
dataset.

### Server

- [X] T019 [US1] Add `GET /videos` to `src/videoannotator/api/v1/videos.py` per `contracts/api.md`: scopes `project`, `folder` (+`recursive`; the folder's video files listed when it passes `_in_readable_folder`, else 403 `PATH_NOT_ALLOWED`; files never run included with empty `results`), `not_in_project`, `run`, none; `q`, `offset`, `limit` (≤ 2,000); `counts` (all, not in a project, projects with `owner_name`, folders: spec 022 places with videos, folders of known videos inside the shared folders, and uploads); `runs` for project and run scopes; `new_in_folder` (T010). Pydantic response models. Tests in `tests/api/test_videos_list_api.py` per scope, the 403, 404s (`DATASET_NOT_FOUND`, `RUN_NOT_FOUND`) and counts
- [X] T020 [US1] Create `src/videoannotator/run_settings.py` with `record_run(db, *, user, dataset_id, selected_pipelines, config, run_id, run_name)` (stores the caller's `last_run_settings` in `user_preferences`, key `anonymous` when no user; project settings are added in T047) and `prefill(db, *, user)` returning last-used, else the defaults (`selected_pipelines` from the registry's default selection the wizard uses today, `config` `{}`), with `source`. Add `GET /videos/run-settings` (no `project`/`workflow` yet). Tests in `tests/unit/test_run_settings.py` and `tests/api/test_videos_run_settings_api.py`
- [X] T021 [US1] Add `POST /videos/run` to `api/v1/videos.py` per `contracts/api.md`: validate with `jobs.validate_pipeline_selection`; for each path, `readable_video`'s rule; a path inside a job's `storage_path` (a stored upload, found through `video_index`) → `jobs.job_from_stored_video` (hard link); else a job reading in place with a spec 022 run folder, as `ask.create_run` does (`RunFolder.create(run_name(...), batch_id=…)`, `run.add_video`); one `batch_id`, `batch_name` = `name` or the folder's or project's name; `dataset_id` = `project_id`; `skipped` with reasons; 422 `NOTHING_TO_RUN` when none; calls `run_settings.record_run`. Any signed-in user. Tests in `tests/api/test_videos_run_api.py`: a folder video in place, an upload hard-linked, a mix in one run, a missing path skipped, nothing usable, settings recorded
- [X] T022 [P] [US1] Add `GET /videos/activity` to `api/v1/videos.py` from `video_index.runs` (active: any video waiting or running; recent: up to 5 finished in the last 24 h); tests in `tests/api/test_videos_activity_api.py`
- [X] T023 [US1] Add a filter to `src/videoannotator/exporters/results_set.py`: `ResultsSet.only(pairs: set[tuple[job_id, pipeline]])` keeping only those files (and dropping videos left with none), and `latest_pairs(index, video_keys)` in `video_index.py` (per video and pipeline, the newest job whose pipeline completed). Tests in `tests/unit/test_results_set_filter.py`
- [X] T024 [US1] Add `POST /videos/export/table`, `/videos/export/elan`, `/videos/export/methods` to `src/videoannotator/api/v1/exports.py` per `contracts/api.md` (exactly one of `paths`, `project_id`, `run_id`; `results` `latest` | `all`; reuse `_table`, `write_videos`, `_methods`, the summary header and errors; file names from the project, run or "Videos"). Tests in `tests/api/test_videos_export_api.py`: latest vs all, a project, a run, nothing to export, parquet missing

### Viewer

- [X] T025 [US1] Create `viewer/src/pages/Videos.tsx` (lazy route `/videos`) with `viewer/src/components/videos/Sidebar.tsx` (All videos, Not in a project, Projects, Folders with counts from `useVideos`, the scope in the address per `contracts/viewer.md`) and `viewer/src/components/videos/VideosTable.tsx` (checkbox, Video, Folder, Projects chips, one column per pipeline in the scope with ✓ ● ◌ ✕ – and the reason on hover, "not on the server" rows, windowed past 200 rows as `components/video/DataTable.tsx` does, a name filter, `/` focuses it, `Enter` opens, `Space` selects); a row opens `/video?path=…&from=<scope>` (extend `lib/listNavigation.ts` with `all` and `unassigned` kinds and their list loading). `PageHelp` at the top. Tests in `viewer/src/test/pages/videos.test.tsx`
- [X] T026 [US1] Create `viewer/src/components/videos/ScopeHeader.tsx`: the scope's name and count, **Run…** (opens `/run` with a `RunTarget` for the folder, project or what's shown) and **Export ▾** (results table, ELAN, methods for the scope, through `ExportMenu` with the new scopes); and `viewer/src/components/videos/SelectionBar.tsx`: "n selected", **Run…** and **Export ▾** for the selection (Add to project comes in US2). Tests in `viewer/src/test/pages/videosActions.test.tsx`
- [X] T027 [US1] Extend `viewer/src/components/ExportMenu.tsx` and `MethodsDialog.tsx` for `videos` and `project` scopes (POST bodies; **Latest results** / **Every run** choice; download through the existing blob path); tests in `viewer/src/test/components/ExportMenu.test.tsx`
- [X] T028 [US1] Create `viewer/src/lib/runTarget.ts` (the `RunTarget` type from data-model.md; `runTargetOf(state)` reading it and the old `WizardStart` shapes from `lib/wizardStart.ts`) and `viewer/src/hooks/useRunSettings.ts` (on `getRunSettings`); tests in `viewer/src/test/lib/runTarget.test.ts`
- [X] T029 [US1] Create `viewer/src/pages/NewRun.tsx` (lazy route `/run`) and `viewer/src/components/run/`: `WhatToFind.tsx` (pipeline choice with the 026 guide, as the wizard's Select Pipelines step, and `DynamicPipelineParameters`; "Settings from: <source>" at the top from `useRunSettings`), `StartRun.tsx` (videos count and names, pipelines, estimate from `lib/estimate.ts`, run name, the project if the target has one; **Start run on N videos**), and `ChooseVideos.tsx` for no target (a shared folder via `ServerFolderPicker`, this computer via upload, a project, a video ran before). Starting with a target calls `runVideos` (resolving a folder or project target to paths through `listVideos`); an upload keeps today's `POST /jobs` per file with one `batch_id` (move that code from `pages/NewJob.tsx` into `components/run/uploadRun.ts`). With a target and nothing changed, Start is the only confirmation (FR-017). After starting: `notify` "Run started on N videos. You can close this tab; results will be here." with **Show run** → `/videos?run=<id>`. Tests in `viewer/src/test/pages/newRun.test.tsx` for each target, the upload path, and the one-confirmation rule
- [X] T030 [US1] Create `viewer/src/components/videos/RunHeader.tsx` for `?run=`: name, project, progress bar and counts, **Cancel** (`/batches/{id}/cancel` or `/jobs/{id}/cancel`), **Retry failed videos** (`/batches/{id}/retry`), **Run again…** (`/run` with `{kind:'rerun'}`), **Delete run** (existing `RunDeleteButton`, reworded), and when finished the headline "**N videos → results table**" with **Download**, **ELAN**, **Methods** beside it (the run's existing batch export routes); VLM runs add **Compare answers** (`/compare?…` as `CompareWith` builds it). Tests in `viewer/src/test/pages/videosRun.test.tsx`
- [X] T031 [US1] Create `viewer/src/components/ActivityIndicator.tsx` (in the nav: "n runs · done/total" with a popover listing active and recent runs, each linking to `/videos?run=`), raising `notify` on `useActivity` transitions: started, finished (+ **Results table**), failed (+ **Show run**), stopped; mount it in `components/AppLayout.tsx` beside Settings (the nav itself changes in T060). Tests in `viewer/src/test/components/activity.test.tsx`
- [X] T032 [US1] Show progress in the folder or project header while one of its runs is active (`runs` from `GET /videos`), and rows filling in as videos finish (`useVideos` refetch); test in `viewer/src/test/pages/videos.test.tsx`

**Checkpoint**: a folder can be run and its table downloaded from Videos; the old pages still work beside it.

---

## Phase 4: User Story 2 – Projects after the fact (P1)

**Goal**: select videos → Add to project (existing or new); a project spans folders; a video in
two projects shows both chips; datasets appear as projects; new videos in a folder are offered;
removing a project leaves videos and results.
**Independent test**: quickstart US2.

### Server

- [X] T033 [US2] In `src/videoannotator/api/v1/datasets.py`: `VideoManifestEntry.path` (optional, resolved on write); `current_settings` in `DatasetResponse`; `PUT` open to any signed-in user for `name`, `description`, `video_manifest`, `server_selection`, `current_settings` (delete keeps `_require_owner_or_admin`); new `POST /datasets/{id}/videos` and `DELETE /datasets/{id}/videos` (`{"paths": [...]}`; adding to a project that follows its folder turns it into a selection of the folder's current videos plus the new paths, per R4; removing never touches files or jobs). Tests in `tests/api/test_datasets_project_fields.py`: add from two folders, follow-folder → selection, remove, another user may edit but not delete, export/import round-trips `path`
- [X] T034 [P] [US2] Make `POST /datasets` accept entries with `path` and no folder (a project made from a selection), and keep the unique-name-per-owner rule with a 409 the viewer words; tests in `tests/api/test_datasets_project_fields.py`

### Viewer

- [X] T035 [US2] Create `viewer/src/hooks/useProjects.ts` (create, rename, add videos, remove videos, delete; invalidates `useVideos` queries) and `viewer/src/components/videos/AddToProject.tsx` (**Add to project ▾**: each project, **New project…** with a name field); add it, and **Remove from project** inside a project, to `SelectionBar`. `notify` on each ("2 videos added to QS"). Tests in `viewer/src/test/pages/videosProjects.test.tsx`
- [X] T036 [US2] Create `viewer/src/components/videos/ProjectHeader.tsx` (extends `ScopeHeader` for a project: name with **Rename**, owner, count, runs line `● running 31/48 · ✓ 12 Oct` linking to `?run=`, **Remove project** with a confirmation saying the videos and their results stay; folder-following projects say "All videos in <folder>, including new ones") and `viewer/src/components/videos/NewInFolder.tsx` ("**3 new videos in /data/pk** · **Add them**" from `new_in_folder`); **+ New project** on the left (an empty project, or from the current selection). Tests in `viewer/src/test/pages/videosProjects.test.tsx`
- [X] T037 [US2] Show project chips on rows and, in the Video page header (`components/video/Header.tsx`), the projects holding the video from `GET /videos/runs` (already there) linking to `/videos?project=`; projects with clashing names show their owner. Test in `viewer/src/test/pages/videosProjects.test.tsx`

**Checkpoint**: projects are made, filled and removed from Videos; old datasets show as projects.

---

## Phase 5: User Story 3 – Projects remember their settings (P1)

**Goal**: a run on a project sets its settings; Edit changes them without running; New run says
where its settings came from; What changed? between runs.
**Independent test**: quickstart US3.

### Server

- [X] T038 [US3] In `run_settings.py`: `record_run` also sets the project's `current_settings` (`source: run`) when `dataset_id` is given; `prefill(db, user, project_id=None)` in FR-012's order (project's current settings; else the settings of the project's newest run through `list_jobs_by_dataset`; else last-used; else defaults), with `source` and `changed_at`. Call `record_run` from `POST /ingest` (`api/v1/ingest.py`), `POST /datasets/{id}/run`, `POST /batches/{id}/rerun` and `POST /jobs/{id}/rerun` and the plain `POST /jobs` (last-used only), and `POST /videos/run`. `GET /videos/run-settings` takes `project`. Tests in `tests/unit/test_run_settings.py` (each source in order) and `tests/api/test_videos_run_settings_api.py` (a run through each route updates them)

### Viewer

- [X] T039 [US3] In `ProjectHeader.tsx`: the **Settings** line (pipelines by their 026 titles, and Ask AI's question when `vlm_annotation` is in them) with **Edit**, opening a sheet with `WhatToFind` that saves through `PUT /datasets/{id}` (`source: edit`) without starting a run; `notify` "Settings saved for QS. The next run starts from them." Tests in `viewer/src/test/pages/videosSettings.test.tsx`
- [X] T040 [US3] In `NewRun.tsx`: pass the target's project to `useRunSettings`; the top line reads "Settings from: this project's settings, last changed 12 Oct" / "your last run" / "the defaults" (FR-012); test in `viewer/src/test/pages/newRun.test.tsx`
- [X] T041 [US3] **What changed?** in `ProjectHeader` and `RunHeader`: pick two of the project's runs (or this run and another on the same videos) and list the settings that differ, from each run's provenance via `lib/settingsDiff.ts` ("Faces · smallest face: 30 → 40"); test in `viewer/src/test/pages/videosSettings.test.tsx`

**Checkpoint**: settings follow the project without anyone saving them.

---

## Phase 6: User Story 5 – One video, one place (P1)

**Goal**: everything about a video is on its page or row; every old address leads somewhere;
a run outside a project is a filter of Videos.
**Independent test**: quickstart US5; the routes test.

- [X] T042 [US5] Create `viewer/src/components/video/RunsPanel.tsx` for the Video page's **Runs** tab (replacing `RunsList` in `components/video/VideoView.tsx`): per run, newest first, its status (`JobStatusBadge`, reworded), progress and queue position while waiting (as `pages/JobDetail.tsx` computes it), each pipeline's state and error, its settings (`DynamicPipelineParameters` read-only or a settings list), `ResultsLocation`, **Download all files** (`/jobs/{id}/artifacts`), **Cancel**, **Delete** (`JobDeleteButton`, reworded), **Run again…** (`RunAgainActions` with a `RunTarget`), and "part of <run name>" linking to `/videos?run=`. Needs a `jobs` list per run in `GET /videos/runs`' response (already has job ids and statuses; add `progress` and `queue_position` additively if missing). Tests in `viewer/src/test/components/runsPanel.test.tsx`
- [X] T043 [US5] Make `RunAgainActions.tsx` work from a `RunTarget` (**Run again…** → `/run` with `{kind:'rerun'}`; **Run again unchanged** as now) and drop its job/batch wording; update `viewer/src/test/components/RunAgainActions.test.tsx`
- [X] T044 [US5] Routes and redirects in `viewer/src/App.tsx` per `contracts/viewer.md`: `/videos`, `/run`, `/help`, `/home`; `/jobs` → `/videos`; `/jobs/new` → `/run` (state passed on); `/jobs/:jobId` → a small `JobRedirect` (fetch the job → `/video?path=…&run=…`; not found → `/videos` with a `notify`); `/batches` → `/videos`; `/batches/:batchId` → `/videos?run=`; `/datasets`, `/results`, `/library` → `/videos`; `/getting-started` → `/help`. Test `viewer/src/test/routes.test.tsx` visits every old and new address and checks where it lands (SC-006)
- [X] T045 [US5] Demos without the local library (R14): `/viewer?demo=<key>` in `viewer/src/pages/Index.tsx` fetches the demo's bundled files (`utils/debugUtils.ts: DEMO_DATA_SETS`) into `File`s and opens them in the standalone Video page (the local-files path `/viewer` uses); Home and Help link demos there. Test in `viewer/src/test/pages/demo.test.tsx`
- [X] T046 [US5] Delete, once nothing imports them (`grep -rn` each): `pages/Jobs.tsx`, `pages/JobDetail.tsx`, `pages/BatchDetail.tsx`, `pages/Results.tsx`, `pages/Datasets.tsx`, `pages/NewJob.tsx`, `components/VideoAnnotationViewer.tsx`, `UnifiedControls.tsx`, `FileViewer.tsx`, `VlmAnnotationPanel.tsx`, `OpenFace3Controls.tsx`, `Timeline.tsx`, `VideoControls.tsx`, `JobsTable.tsx`, `BatchCard.tsx`, `BatchVideosTable.tsx`, `VideosAndResultsCard.tsx`, `DatasetDriftDialog.tsx`, `SaveDatasetDialog.tsx`, `DatasetPicker.tsx` (if New run's project choice replaced it), `StartFromRecent.tsx` (if `ChooseVideos` replaced it), `hooks/useZipDownloader.ts`, `hooks/useBatches.ts`, `lib/localLibrary/`, `lib/datasetHandles.ts`, and their tests; move anything still needed (queue position, status classes) into the components that use it. Ticks spec 028's T056. Bundle check passes

**Checkpoint**: the old pages are gone; every old address lands; the vocabulary test's `KNOWN` list covers only Home, Pipelines and Help.

---

## Phase 7: User Story 4 – Workflows, only when wanted (P2)

**Goal**: save a workflow from a run or project, apply it to a project (copied), share as a file;
presets appear as workflows.
**Independent test**: quickstart US4.

- [X] T047 [US4] In `src/videoannotator/api/v1/presets.py`: `export_choices` and `source` on create, update, responses and export/import JSON; `run_settings.prefill` takes `workflow_id` first in the order (FR-012) and `GET /videos/run-settings` takes `workflow`; applying is `PUT /datasets/{id}` with `current_settings.source = {kind: workflow, id, name}` (no new route). Tests in `tests/api/test_presets_workflow_fields.py` and `tests/unit/test_run_settings.py` (a workflow edited after applying leaves the project unchanged)
- [X] T048 [US4] Create `viewer/src/hooks/useWorkflows.ts` and `viewer/src/components/workflows/`: `SaveAsWorkflow.tsx` (from a project's current settings or a run's settings; name, export choices), `WorkflowPicker.tsx` (in New run's `WhatToFind`, replacing `PresetBar`), `ApplyWorkflow.tsx` (to a project: copies, `notify` "QS now uses Faces 40's settings"); add **Save as workflow** and **Apply workflow** to `ProjectHeader` and **Save as workflow** to `RunHeader` and the Runs tab. Delete `components/PresetBar.tsx` when unimported. Tests in `viewer/src/test/components/workflows.test.tsx`
- [X] T049 [US4] Add **Workflows** to `viewer/src/pages/Pipelines.tsx`: each workflow's name, pipelines, where it came from, **Apply to project…**, **Use in a new run** (`/run` with the workflow chosen), **Export** (JSON file), **Rename**, **Delete**, and **Import workflow**; `PageHelp`. Tests in `viewer/src/test/pages/pipelinesWorkflows.test.tsx`
- [X] T050 [US4] Use a workflow's `export_choices` in `RunHeader`'s headline (the table format offered first; ELAN and methods shown or not); test in `viewer/src/test/pages/videosRun.test.tsx`

---

## Phase 8: User Story 6 – Help where the question arises (P2)

**Goal**: a line of help and a docs link on every page; tooltips on terms and settings; toasts
with a next step; pages that need a server say so.
**Independent test**: quickstart US6; the docs-links test.

- [X] T051 [P] [US6] Write `docs/usage/videos_and_projects.md` (Videos page, folders, projects and how membership works, settings that projects remember, workflows, runs as filters, exporting a selection) with headings matching `lib/docsLinks.ts`' anchors; link it from `docs/usage/README.md` and `mkdocs.yml`'s nav
- [X] T052 [P] [US6] Rewrite `docs/usage/GETTING_STARTED.md` for this structure (Home's three start points, Videos, New run, the Video page, Ask AI, Help) with the anchors Help links to
- [X] T053 [US6] Create `viewer/src/pages/Help.tsx` (`/help`, replacing `pages/GettingStarted.tsx`): what VideoAnnotator does in three sentences, the vocabulary from `glossary.ts`, the start points, demos, and links into the docs sections; delete `GettingStarted.tsx`. Test in `viewer/src/test/pages/help.test.tsx`
- [X] T054 [US6] Add `PageHelp` to every routed page (Home, Videos, Video page's ⋯ or About tab, Ask AI, Pipelines, New run, Settings, Help, Compare) with its `docsLinks` entry, and `Term` to the first use of each vocabulary word on Videos, New run and Pipelines; `docsLinks.test.ts` covers every entry
- [X] T055 [US6] Setting tooltips in `viewer/src/components/DynamicPipelineParameters.tsx`: each label gets a `Tooltip` with the description and "Default: <value> <unit>" (FR-024); test in `viewer/src/test/components/DynamicPipelineParameters.test.tsx`
- [X] T056 [US6] Toasts for project and workflow actions not yet covered, and for run cancel and delete (FR-025), all through `notify` with a next step; `NeedsServer` on Videos, New run and Pipelines' install actions when `useServerCapabilitiesContext` reports no server (FR-026). Tests in `viewer/src/test/pages/noServer.test.tsx`
- [X] T057 [US6] Replace every place a server error is shown on these pages with `userMessage(error)` (`grep -rn "error.message\|parseApiError" viewer/src/pages viewer/src/components`); the debug panel keeps the raw message

---

## Phase 9: User Story 7 – First run (P2)

**Goal**: a new user's Home shows the three start points and the checklist; with results, the
viewer opens on Videos.
**Independent test**: quickstart US7.

- [X] T058 [US7] Rework `viewer/src/pages/Home.tsx` (also at `/home`): three start points, **Ask AI about a video** (`/ask`), **Run pipelines on videos** (`/run`), **Open files from this computer** (`/viewer`), plus demos and the existing setup checklist; recent work moves to Videos (remove Home's runs and jobs lists). Test in `viewer/src/test/components/Home.test.tsx`
- [X] T059 [US7] The root's choice (R18) in `App.tsx`: `/` renders a `RootRoute` that asks `listVideos({}, limit 1)` for any video with a done result and shows Videos, else Home; without a server, Home. Test in `viewer/src/test/routes.test.tsx`
- [X] T060 [US7] The nav in `viewer/src/components/AppLayout.tsx`: **Home · Videos · Ask AI · Pipelines**, `ActivityIndicator`, **Help**, **Settings** (SC-004); logo → `/home`; remove New job, Jobs, Results, Datasets. Update `viewer/src/test/AppRender.test.tsx`

---

## Phase 10: Polish & cross-cutting

- [X] T061 Empty the vocabulary test's `KNOWN` list (fix what's left on Home, Pipelines, Settings, Compare, Ask AI, the Video page) so SC-002 holds for every route
- [X] T062 [P] Update `docs/usage/video_page.md`, `docs/usage/exporting_results.md`, `docs/usage/ask_ai.md` and `docs/usage/README.md` for the new pages and words (links to Videos instead of Jobs/Results/Datasets); API reference for the new routes in the docs where 028's are; `CHANGELOG.md` Unreleased: Videos and projects, workflows, New run, the nav, help, the new endpoints and fields, and that any signed-in user may now edit a saved dataset (R4)
- [X] T063 [P] Check the bundle stays within 300 KB gzipped (`bash scripts/build_viewer.sh` reports it); Videos and New run lazy
- [ ] T064 Walk through `quickstart.md` in a browser (screenshots from the dev container), measure SC-001, record results in `specs/029-videos-and-projects/checks.md`
- [ ] T065 Update `docs/development/roadmap_v1.7.0.md` Phase 1 (029 built, link checks.md; 028's T056 done), `specs/028-video-page/tasks.md` T056 ticked, `CLAUDE.md` Current Plan, and the spec's Status to "Built"

---

## Dependencies and execution order

- **Phase 1 → Phase 2 → stories.** T003–T010 (server) and T011–T018 (viewer) are independent of
  each other; within the server, T005 → T006 → T008/T009 → T010; T004 before T005.
- **US1** (T019–T032) needs Phase 2. **US2** needs T019 (list) and T025–T026 (page). **US3**
  needs US1's New run (T029) and US2's project header (T036). **US5** needs US1–US3 so nothing
  is deleted before its replacement exists (T046 last). **US4** needs US3 (prefill order) and
  US2 (project header). **US6** can start after Phase 2 for docs (T051, T052) and after US5 for
  pages (T054). **US7** needs US1 (Videos) and T031 (indicator).
- Polish after all stories; T061 last among code tasks.

## Parallel opportunities

- Phase 2: T004 with T011–T015 and T017–T018; T007 once T005–T006 exist.
- US1: T022 and T023 beside T021; viewer T027 and T028 beside T025.
- US2: T034 beside T033.
- US6: T051 and T052 (docs) beside any viewer work.
- Polish: T062 and T063 together.

## Implementation strategy

1. **MVP = Phase 2 + US1**: Videos with folders, Run…, the activity indicator and the headline
   table, beside the old pages. The Video page's rerun fix (T008) ships in Phase 2 already.
2. **US2 + US3**: projects and their settings; then **US5** retires the old pages in one step
   once every replacement exists.
3. **US4, US6, US7**: workflows, help and first run; then polish and the walk-through.
