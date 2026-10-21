# Research: Videos and Projects

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Date**: 2026-10-20

What the code does today, and the decisions this plan takes. Each decision says what was chosen,
why, and what else was considered.

## What exists (read 2026-10-20, at `0cec2d7`)

- **Jobs** (`storage/sqlite_backend.py`, the default backend): table `jobs` with `video_path`,
  `output_dir`, `batch_id`, `batch_name`, `dataset_id`, `rerun_of`, status and times;
  per-pipeline results in their own table (`pipeline_results`: status, error, provenance).
  Listing helpers load one job at a time (`list_jobs`, then `load_job_metadata`).
- **One video, several paths.** An upload is stored in its job's folder
  (`storage/jobs/<id>/<name>`). "Run again" and a dataset run hard-link that copy into the new
  job's folder (`api/v1/jobs.py: job_from_stored_video`), so every rerun of an uploaded video has
  a new path. Spec 028 keys the Video page by resolved path (`video_runs.runs_for`), so the
  Video page of an uploaded video does not show its reruns, and a list by path would show the
  video once per run. Videos in shared folders (spec 022) keep one path. Provenance records the
  input's `sha256` (spec 017) for every result.
- **Batches** are not a table: a batch is the jobs sharing a `batch_id` (`api/v1/batches.py`),
  with a computed summary, cancel, retry, rerun (with relocation, spec 022), delete, a zip.
- **Datasets** (`saved_datasets`, `api/v1/datasets.py`): name, description, owner, a
  `video_manifest` of `{filename, size_bytes, relative_path?}`, an optional `server_folder`
  (+ `server_folder_recursive`) and `server_selection` (the manifest is a chosen subset of the
  folder, spec 022). Unique name per owner. Only the owner or an admin may change or delete
  one. `/stored-videos` and `/run` reuse jobs' stored copies for upload datasets. Jobs record
  `dataset_id`; `video_runs.projects_for` finds a video's datasets by that or by folder and
  manifest.
- **Presets** (`saved_pipeline_presets`, `api/v1/presets.py`): name, description, owner,
  `selected_pipelines`, `config` (which already holds VLM prompts and sampling), `tags`,
  `last_used_at`. Export and import as JSON in the wizard's `PresetBar`.
- **Starting work on server videos**: `POST /ingest` (a folder or chosen files in it; admin and
  same-machine only, spec 022), `POST /datasets/{id}/run` (stored copies), `POST /jobs/{id}/rerun`,
  `POST /batches/{id}/rerun`, and Ask AI's `POST /ask/runs`, which creates a job on any path
  `readable_video` allows (shared folders, `storage/videos` uploads, any path a job read) for any
  signed-in user, with a spec 022 run folder.
- **Exports** (`api/v1/exports.py`, spec 025): table per job, batch and dataset; methods and
  ELAN per job and batch. All resolve to jobs and call `exporters.results_set.from_jobs`.
  `VideoResults.files` carry their pipeline, so a set can be narrowed per pipeline.
- **Additive schema changes** are applied at start-up (`database/migrations.py`:
  `ALTER TABLE … ADD COLUMN` when missing, as spec 018 and 022 did for `saved_datasets`).
- **Viewer pages to retire**: Jobs (397 lines), Job detail (534: status, progress and queue
  position, run again, video information, settings, results folder, download zip, export,
  cancel, delete), Batch detail (325), Results (235: the local library, demos, open files),
  Datasets (304: list, import/export, drift dialog, delete), Getting started (249). New job is
  1,917 lines in `pages/NewJob.tsx`. Home (448) has start points, the setup checklist and recent
  work. Pipelines (38) wraps the pipeline guide.
- **Old player**: `VideoAnnotationViewer` with `UnifiedControls`, `FileViewer`,
  `VlmAnnotationPanel`, `OpenFace3Controls`, `Timeline` still serves demo results from the local
  library (`lib/localLibrary`, `hooks/useZipDownloader.ts`) and `/viewer`'s file screen
  (spec 028 T056, waiting on this spec).
- **Following work**: no activity endpoint; pages poll their own queries (Jobs, Job detail,
  `useBatches`, Home). The SSE provider is mounted with `enabled={false}`.
- **Words**: DeepWiki links are already gone from `viewer/src`. The nav and pages still say
  New job, Jobs, Results, Datasets; many strings say job, batch, dataset, preset.
- **Docs**: `docs/` is published with MkDocs at `https://infantlab.github.io/VideoAnnotator/`
  (`mkdocs.yml`); `docs/usage/` has the how-tos (Ask AI, Video page, exporting results, …).
- **Reusable viewer pieces**: `lib/settingsDiff.ts` (028, settings that differ between runs),
  `ExportMenu` and `MethodsDialog` (025), `DynamicPipelineParameters` and the pipeline guide
  (026), `ServerFolderPicker`, `ListNavigation`/`lib/listNavigation.ts` (`from=run|project|folder`),
  `RunAgainActions`, `ResultsLocation`, `JobStatusBadge`, `lib/videoLinks.ts`.

## Decisions

### R1. A video is a group of paths the server can tell are one file

**Decision**: A video is identified by its paths: every path a job, upload or project names,
resolved. Paths are joined into one video when the server can tell they hold the same file:
the same resolved path; hard links of one file (same device and inode, which is what every
rerun of an upload is); a job's `rerun_of` whose video was a stored copy (so the link holds
after one copy is deleted); or the same `sha256` recorded in their provenance (two uploads of
one file). Nothing else joins them: not the name, not the size. The video's address stays
`/video?path=<any of its paths>`; the server maps it to the group, and the list links each video
by its newest existing path. `GET /videos/runs` returns the runs of the whole group (additive:
same shape, more runs).

**Rationale**: Without it, "Run again" on an uploaded video makes a second video, and the
Video page of the first doesn't show the rerun: the case the maintainer hit on 2026-10-20.
Inode and `rerun_of` need no hashing; `sha256` is already recorded. The spec's edge case
("one video when the server can tell, otherwise two, never silently merged") is exactly this.

**Amended 2026-10-21 (walk-through, `checks.md`)**: the same `sha256` joins paths only when one
of them is an upload (under the storage root, or in a job's own folder). The quickstart's three
byte-identical clips in one shared folder were shown as one video; files a researcher sees as
separate in a folder stay separate videos.

**Alternatives**: a `videos` table with an id (a second identity to keep in step with jobs and
manifests, and a migration for every existing job; worth it only if grouping turns out too slow,
R11); hashing every file (tens of seconds for a large video, on first open); joining by name and
size (merges different recordings of the same length).

### R2. One list request: `GET /videos`

**Decision**: A new endpoint lists the videos the server knows, for one scope: all, not in a
project, a project, a folder, or a run. Each video has its name, path, folder, whether the file
is there, size, its projects, and per pipeline the state of its newest run (done, running,
failed, none). The response also carries the left side's counts (all, not in a project, per
project, per folder). Built from one query over jobs and their pipeline results (a new backend
method, `job_summaries()`, with a slow fallback for the file backend), the saved datasets, and,
for a folder scope, the folder's video files. Contract: [contracts/api.md](contracts/api.md).

**Rationale**: FR-004. The existing endpoints stay as they are (Principle V). Per-pipeline state
comes from the database, never from statting result files, so the list stays fast; a missing
result file is the Video page's to report (as it does since 2026-10-20).

**Alternatives**: building the list in the browser from `/jobs`, `/batches` and `/datasets`
(three paginated APIs and the grouping of R1 in TypeScript, repeated on every client); a
server-side search index (not needed at one lab's scale, R11).

### R3. The left side: All, Not in a project, projects, folders

**Decision**: All videos; Not in a project; each project (with its owner when two share a
name); each folder: the shared places that have videos (spec 022's `places`), the folders of
known videos inside the shared folders, and "Uploaded videos" for uploads. Runs are not on the
left: a run is a filter of the list, reached from a project's runs, the activity indicator, a
toast, or an old `/batches/…` address (R12).

**Rationale**: Spec 023's sketch and FR-001. Listing a folder's videos uses the same rule as
`readable_video` (inside a shared folder or the uploads folder), so any signed-in user sees the
videos they could already open in Ask AI. The admin and same-machine gates on `/ingest/browse`
and `/ingest/scan` are unchanged: browsing for a new folder still goes through them.

**Alternatives**: browsing any folder from the left (needs the ingest gates relaxed, which is
the maintainer's call, not this spec's); runs as a fourth group on the left (spec 023 Decision 5:
a run is not a level).

### R4. A project is a saved dataset, and can hold videos from any folder

**Decision**: The viewer shows every saved dataset as a project (FR-005); the API keeps its
name. Membership is worked out from what a dataset already records, plus one additive field:

- a manifest entry may now carry `path`, the video's server path; "Add to project" writes
  entries with it, so a project can hold videos from several folders;
- a dataset with a folder and no selection (`server_selection` false) holds every video in the
  folder, including new ones, as today ("follows the folder");
- a dataset with a folder and a selection holds its manifest entries (`relative_path` within the
  folder, or `path`); videos added to the folder since are offered in the project's header
  ("3 new videos in /data/pk: Add them"), which replaces the drift dialog (FR-007);
- an upload dataset (no folder) holds the videos of the jobs that ran from it (`dataset_id`)
  and, by filename and size, the stored copies `/stored-videos` already finds.

Adding videos from another folder to a project that follows its folder turns it into a
selection of that folder plus the new paths, and its header says so. Removing a project deletes
the dataset only; jobs keep their `dataset_id` (an orphan id shows nothing) and results stay
(FR-006). Any signed-in user may add or remove videos and edit a project's settings; deleting a
project stays with its owner or an admin, as now.

**Rationale**: Spec 023 Decisions 5–7: a project is the set of videos, and a dataset already is
one. One optional manifest field is the smallest change that lets a project span folders.
Opening edits to everyone follows the server's shared-projects assumption (spec Assumptions)
and the maintainer's "no admin friction" preference; delete stays guarded because it is the one
destructive action.

**Alternatives**: a new projects table with a dataset compatibility layer (two models of one
thing); a separate membership table (`project_videos`) instead of manifest entries (cleaner
queries, but the dataset export/import format, FR-006 of spec 007, would no longer round-trip
a project).

### R5. A project remembers its current settings, on the server

**Decision**: `saved_datasets.current_settings` (JSON, nullable, additive): pipelines, config,
where they came from (`run`, `edit` or `workflow`, with its name), when, and by whom. It is set
on the server whenever a run starts on the project (any route that records `dataset_id`), and by
`PUT /datasets/{id}` when edited in the project's header. A project with none (every dataset
before this spec) starts from the settings of its newest run, found through its jobs, and
otherwise from the last-used settings (R6).

**Rationale**: FR-009, FR-010. Setting it where runs are created means the CLI and API keep it
current too, not only the viewer. Old datasets need no migration: their newest run already is
their current settings.

**Alternatives**: computing it from the newest run only (loses settings edited without
running, US3 scenario 2); storing it in the browser (other users and machines wouldn't see it).

### R6. Last-used settings per user, and one place that decides the prefill

**Decision**: A small table `user_preferences (user_key, name, value JSON, updated_at)`; run
creation stores `last_run_settings` for the caller (`anonymous` when the server runs without
sign-in). `GET /videos/run-settings?project=&workflow=` returns the settings New run starts
from, in FR-012's order (the chosen workflow, the project's current settings, the caller's
last-used settings, the defaults), with a `source` saying which and when ("This project's
settings, last changed 12 Oct").

**Rationale**: FR-011, FR-012. Deciding the order on the server keeps the CLI and viewer the
same and lets the viewer only show it. A generic preferences table avoids a column per future
preference.

**Alternatives**: browser storage (per machine, not per user, and invisible to the CLI); a
column on `users` (no row when sign-in is off).

### R7. Workflows are presets, with export choices

**Decision**: The viewer shows every preset as a workflow (FR-015). The API keeps `presets` and
gains `export_choices` (JSON, nullable: the table format, ELAN, methods) and `source` (the run or
project it was saved from). Ask AI questions and sampling are already in `config.vlm_annotation`.
"Save as workflow" on a project or run creates a preset; applying one to a project copies its
pipelines and settings into `current_settings` with `source: workflow` (FR-014); later edits to
the workflow change nothing else. Export and import as a file stay presets' JSON, with the new
fields. Workflows are listed on Pipelines (FR-016) and offered in New run.

**Rationale**: Spec 023 settings layer 3 and Decision 7's twin: a preset already is a workflow
minus export choices.

**Alternatives**: a workflows table (duplicates presets); linking projects to workflows instead
of copying (a later edit would silently change a project's settings, which spec 023 rules out).

### R8. One request to start a run on any videos: `POST /videos/run`

**Decision**: A new endpoint takes server paths (any mix of folders and uploads), pipelines,
settings, an optional name and an optional project, and creates one run (one `batch_id`): a
video in a shared folder is read in place, an uploaded one is hard-linked from its stored copy
(`job_from_stored_video`), each path checked by `readable_video`'s rule. Videos it can't use
are listed in `skipped`. Any signed-in user may call it, as Ask AI's runs (R3). It records the
project's current settings (R5) and the caller's last-used settings (R6). Videos from another
computer are still uploaded by New run with `POST /jobs`, one per video, under one `batch_id`.

**Rationale**: FR-017 ("on exactly those videos"): a selection can span folders and uploads,
which no current endpoint takes in one request. Ask AI's run already sets the precedent for
access.

**Alternatives**: one `/ingest` call per folder plus `/datasets/{id}/run` for uploads, from the
viewer (several requests that can half-succeed; ingest's admin gate would block non-admin
users from running on a selection they can see).

### R9. New run: from a target, two steps, one confirmation

**Decision**: The wizard becomes **New run** at `/run` (`/jobs/new` keeps working and opens
it). It is opened with a target: videos (paths), a project, a folder, a run to repeat, or
nothing (then its first step chooses videos as today: a shared folder, an upload, a project).
With a target, the steps are **What to find** (pipelines with their guide, spec 026, and their
settings, prefilled per R6 with the source said at the top and a workflow picker) and **Start**
(a summary: N videos, pipelines, estimated time, the run's name, the project it belongs to).
Keeping the prefilled settings and pressing Start is the one confirmation (FR-017). The
settings step is the redesigned Configure (2026-10-20); PresetBar becomes the workflow picker
and "Save as workflow". "Save as dataset" goes: projects are made after the fact (US2).
`pages/NewJob.tsx` is split into `pages/NewRun.tsx` and components under `components/run/`.

**Rationale**: Spec 023's "simplified wizard from a video, a project, a selection or Home";
1,900 lines in one page can't be changed safely.

**Alternatives**: keeping the four-step wizard and hiding steps (the target's videos would
still be re-chosen on the way, the 2026-10-20 complaint about "Start from what you used before").

### R10. Export for any set of videos

**Decision**: `POST /videos/export/table`, `/elan` and `/methods` take a set (paths, a project or
a run) and which results: `latest` (default: per video and pipeline, the newest completed
result, the same rule the Video page starts from, ported from `lib/videoRuns.ts`) or `all` (every
run, told apart by the `run` column, as the dataset table does now). `ResultsSet` gains a filter
by (job, pipeline). The viewer's `ExportScope` gains `videos` and `project`; the existing job,
batch and dataset routes stay. After a run finishes, its header offers the run's own table as
the headline result ("48 videos → results table"), with ELAN and methods beside it (US1).

**Rationale**: FR-003 (Export for any selection, folder or project) and US1 scenario 4. POST,
because a selection can be hundreds of paths; the viewer already downloads exports with `fetch`.

**Alternatives**: a GET with repeated `path=` (URL length limits); exporting only per run
(a project's videos ran at different times; the researcher wants one table).

### R11. Fast enough without an index

**Decision**: `job_summaries()` reads jobs and pipeline results in one query; grouping (R1)
stats each distinct path once (for inode and existence) and is cached for a few seconds, keyed
by the jobs table's row count and newest update and the datasets' newest update. Target: `GET
/videos` under 1 s with 5,000 jobs and 2,000 videos on a laptop; the viewer windows the table
past 200 rows as 028's Data tab does (no new dependency). A test builds that many jobs.

**Rationale**: The spec's edge case ("hundreds of videos: fast"). 028 flagged the job scan as
029's to solve; one query instead of one per job is the solve.

**Alternatives**: a `videos` table maintained on every job write (R1's alternative), kept in
reserve if the measured time misses the target.

### R12. A run is a filter, and the Video page holds a job's details

**Decision**: `/videos?run=<batch id or job id>` lists the run's videos, with the run's header:
progress, Cancel, Retry failed videos, Run again… (R9), Delete run, Export (headline table),
"What changed?" against another run (FR-013, `lib/settingsDiff.ts`), and "Compare answers" for
VLM runs (the existing Compare page, kept at `/compare` without a nav entry). What Job detail
showed for one video moves to the Video page's **Runs** tab: each run's status, progress and
queue position, pipelines with their errors, settings, results folder, "Download all files",
Cancel, Delete, Run again…. Its log was never a separate view (Job detail had none).

**Rationale**: US5 scenario 3 and spec 023's page table (Batch detail and Job detail replaced).
Nothing Job or Batch detail could do is lost.

### R13. Every old address leads somewhere

**Decision** (SC-006): `/jobs` → `/videos`; `/jobs/:id` → the Video page on that run
(`/video?path=…&run=:id`, as `/view/:jobId` already does); `/jobs/new` → `/run` (router state
passed on); `/batches` → `/videos`; `/batches/:id` → `/videos?run=:id`; `/datasets` → `/videos`;
`/results` and `/library` → `/videos`; `/getting-started` → `/help`; `/view/:jobId`, `/viewer`,
`/video`, `/ask`, `/compare`, `/prompts`, `/workbench`, `/pipelines`, `/settings` stay. A test
visits every one.

### R14. No local library; demos open as files

**Decision**: The local library goes (FR-021): nothing is copied into the browser, so there is
nothing to show. Demo results (`utils/debugUtils.ts: DEMO_DATA_SETS`) open in the standalone
Video page, `/viewer?demo=<key>`, by fetching their bundled files into the same local-files
path `/viewer` uses for dropped files. Then `VideoAnnotationViewer`, `UnifiedControls`,
`FileViewer`, `VlmAnnotationPanel`, `OpenFace3Controls`, the old `Timeline`, `useZipDownloader`,
`lib/localLibrary` and `lib/datasetHandles` (browser folder handles, used by the drift dialog)
are deleted, finishing spec 028's T056.

**Rationale**: FR-021, and the last users of the old player. An upload project whose stored
copies were deleted is run again by uploading the videos in New run; the drift dialog's other
job (new videos in a folder) is R4's offer.

### R15. The words, checked by a test

**Decision** (SC-002): a Vitest test renders every route with mocked API responses, in its
loaded, empty and error states, and fails on `job(s)`, `batch(es)`, `dataset(s)` or `preset(s)`
in visible text, titles, `aria-label`s and placeholders. Server error messages shown to the user
go through one `userMessage(error)` that maps the error codes these pages can meet to the
viewer's words, and falls back to a neutral sentence instead of the server's text when the text
uses an API word. The API, its docs and the CLI keep their words.

**Rationale**: A rendered-text test catches what a source grep misses (composed strings) and
ignores what it shouldn't flag (identifiers, API paths).

**Alternatives**: a source-scan lint rule (flags `jobId` and API paths; misses composed text).

### R16. Help in context

**Decision**: Each page starts with a `PageHelp` line: one sentence and "More in the docs",
linking into the published docs at a section (`lib/docsLinks.ts`: page → `usage/<file>/#<anchor>`).
A test checks every link's file exists in `docs/` and its anchor matches a heading, using
MkDocs' slug rule (SC-007). A `Term` component gives the vocabulary (video, folder, run,
project, workflow, track) a tooltip from one glossary (`lib/glossary.ts`); each setting's label
gets a tooltip with its description and default (FR-024). Toasts go through one `notify()` with
an optional next-step action; the activity indicator (R17) raises them when a run starts,
finishes, fails or is stopped, and project and workflow actions raise their own (FR-025).
Getting started becomes **Help** (`/help`, FR-027). Server pages without a server show
`NeedsServer`: what needs one, how to connect (Settings), what works without (open files,
demos) (FR-026).

**Rationale**: Spec 023 theme 8. Linking to the published docs (a link, not a runtime asset)
keeps Principle I: the viewer loads nothing from elsewhere.

**Alternatives**: serving `docs/` from the API server (bigger install, a second docs build).

### R17. The activity indicator

**Decision**: `GET /videos/activity` lists runs with videos waiting or running: id (the batch,
or the job for a run of one video), name, total, finished, failed, running, project, started.
The nav's indicator polls it every 5 s while anything is active and every 30 s otherwise, shows
"2 runs · 31/48" with a popover of runs linking to their filters, and raises the run toasts
(R16) by comparing polls.

**Rationale**: FR-018, US1 scenario 3. One small request instead of every page polling its own.
SSE stays off as now; switching to it later changes only the hook.

### R18. The root opens Videos once there are results

**Decision**: `/` shows Videos when the server has any video with a finished run, else Home;
Home is also at `/home`, in the nav and behind the logo. Without a server, `/` is Home (open
files, demos).

**Rationale**: FR-022, US7.

### R19. Spec 010 (corpus analysis) is not reconciled here

**Decision**: The Videos page's per-pipeline columns are the only corpus overview in 029;
spec 010's charts stay where the roadmap puts them (v1.7.0 Phase 1 asks for reconciliation,
which is a separate decision for the maintainer).
