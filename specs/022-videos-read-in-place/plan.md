# Implementation Plan: Videos and Results Where You Expect Them

**Branch**: `022-videos-read-in-place` (work lands on `1.6-dev`) | **Date**: 2026-10-06 |
**Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/022-videos-read-in-place/spec.md`

## Summary

On a laptop install, videos are read where they are, and results go to one visible folder.

**Inputs**:
- **"My folders" becomes the wizard's default.** It is the existing server-folder picker (spec 008),
  extended so researchers can tick individual videos.
- **The viewer asks the server whether this browser counts as the same machine**, using a new
  per-caller access endpoint.
- **Docker gets a supported setup**: the port is published on the host's loopback only, the video
  folder is mounted read-only, and a flag tells the server that every caller is therefore local.

**Outputs**:
- **Every new job's existing `output_dir` field is set at creation** to
  `~/VideoAnnotator/<run> (<date>)/<video>/`. The runner already writes there, so no pipeline
  changes.
- **Each run folder gets a `run.json`** index.
- **The viewer shows the location**, opens it on this computer, and downloads a whole run as one
  zip without videos.

## Technical Context

- **Language/Version**: Python 3.12 and 3.13 (server, CLI); TypeScript with React 18 (viewer).
- **Primary Dependencies**: FastAPI, SQLAlchemy and Pydantic, already in core. Zip streaming and
  folder opening use the standard library (`zipfile`, `os.startfile`, `subprocess`). Viewer:
  existing shadcn/ui components. No new dependencies.
- **Storage**:
  - SQLite: one new column, `saved_datasets.server_selection`.
  - Filesystem: a new results root (default `~/VideoAnnotator`), with run and video folders and
    `run.json`.
  - Internal job folders stay where they are (per-user data directory, `f71d012`).
- **Testing**: pytest (`tests/api`, `tests/unit`, `tests/integration`); Vitest for the viewer; the
  quickstart walkthrough, including Docker.
- **Target Platform**: a researcher's laptop on Linux, macOS or Windows, with a plain install or the
  documented Docker setup. A lab server (another machine) keeps upload.
- **Project Type**: web application (Python API and CLI, plus the bundled React viewer).
- **Performance Goals**:
  - A 100-video run is created in one request, in under 5 s (SC-001).
  - A folder listing returns in under 1 s for 1,000 files, without reading durations.
  - The run zip streams, rather than being built in memory.
- **Constraints**:
  - Never write into allowed (video) folders.
  - Never overwrite results.
  - Same-machine checks are decided by the server only.
  - Docker publishes on loopback only.
  - Windows-safe folder names.
- **Scale/Scope**:
  - Runs of 1–1,000 videos, and folders of up to a few thousand files.
  - About 14 server files touched, plus the viewer's job wizard, the batch and job pages, and the
    settings page.

## Constitution Check

*GATE: checked before research and again after design (below).*

| Principle | Assessment |
|---|---|
| **I. Local-first, no telemetry** | **Pass, and strengthened.** Videos are no longer copied, and nothing new leaves the machine. The results default is `~/VideoAnnotator`, deliberately not `~/Documents`, which is cloud-synced by default on many Windows installs (research.md R4). Docker now publishes on loopback only. |
| **II. Stable pipeline contract, open formats** | **Pass.** No pipeline changes: results move via the existing `output_dir`, and file names are unchanged. `run.json` is a new JSON file. The viewer reads the same formats. |
| **III. Provenance** | **Pass.** Provenance stays in each output (spec 017). `run.json` adds a per-run index of pipelines, config, versions, sources and models. |
| **IV. Modular by construction** | **Pass.** Core only. No pipeline imports, and no new dependencies. Viewer changes are in the job wizard and pages, not in annotation layers. |
| **V. Backward compatibility** | **Pass, with two default changes justified below.** Existing API requests, config, CLI invocations, datasets, routes and output files keep working. Old jobs stay where they are. |
| **VI. Faithful display** | **Pass.** Display of annotations is unchanged. A missing video shows a message in place of playback; results stay fully viewable. |
| **Engineering standards** | Tests for every new endpoint and rule. Docs updated: install (plain and Docker), getting started, the CHANGELOG, and the rc1 walkthrough. |

**Post-design re-check**: unchanged. The design adds no dependencies and keeps every interface
additive, apart from the two default changes in Complexity Tracking.

## Project Structure

### Documentation (this feature)

```
specs/022-videos-read-in-place/
├── spec.md
├── plan.md              # this file
├── research.md          # R1–R14 decisions
├── data-model.md
├── quickstart.md        # end-to-end walkthrough, including Docker
├── contracts/
│   ├── api.md
│   └── docker.md
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks
```

### Source Code (repository root)

```
src/videoannotator/
├── results_folder.py            # NEW: results root, run/video folder naming (R2), run.json (R3),
│                                #      overlap checks, display paths (R7)
├── config_env.py                # RESULTS_DIR, PUBLISHED_LOCALLY, HOST_PATHS
├── api/v1/ingest.py             # access endpoint (R5, R6); `files` selection (R8); results folder
├── api/v1/jobs.py               # upload and rerun set output_dir; artifacts include_video (R12);
│                                #   results_folder and video_available in responses
├── api/v1/batches.py            # results.zip (R12); rerun ?check=true (R10); results_folder
├── api/v1/datasets.py           # dataset run sets output_dir; server_selection (R9)
├── api/v1/results.py            # NEW: POST /results/open (R11)
├── batch/job_execution.py       # missing-video check at job start (R10); run.json update on finish
├── batch/local_job.py           # CLI process: in place, results folder (R14)
├── storage/sqlite_backend.py    # delete_job also removes the video folder and an empty run folder
├── database/migrations.py       # saved_datasets.server_selection
└── cli.py                       # `server` prints the results folder

viewer/src/
├── pages/NewJob.tsx             # My folders default, upload link, source by access (R13)
├── components/ServerFolderPicker.tsx  # → My folders: file checkboxes, select all, subfolders
├── components/ResultsLocation.tsx     # NEW: location, Open folder, Copy, Download run
├── pages/BatchDetail.tsx, pages/JobDetail.tsx  # ResultsLocation; job download without video
├── components/DatasetPicker.tsx # selection datasets (server_selection)
├── hooks/useIngestAccess.ts     # NEW: GET /ingest/access
└── pages/Settings.tsx           # shows results folder and how to change it

docker-compose.yml, docs/installation/INSTALLATION.md   # contracts/docker.md
tests/api/test_results_folder.py, test_ingest_access.py, test_results_open.py   # NEW
tests/unit/test_results_folder_names.py                                          # NEW
viewer/src/test/components/MyFolders.test.tsx, ResultsLocation.test.tsx         # NEW
```

**Structure decision**: the existing web-app layout. The one new server module,
`results_folder.py`, holds every naming and layout rule, so the five job-creation paths share them.

## Delivery Order

Each step is testable on its own and is committed when green (per the project's
push-finished-work practice).

1. **Results folder, server side**: `results_folder.py`; `output_dir` set on all creation paths;
   `run.json`; delete cleanup; the writability check. (Stories 6 and 8.)
2. **Access and selection**: `GET /ingest/access`, ingest `files`, `PUBLISHED_LOCALLY`. (Server
   side of Stories 1 and 2.)
3. **Viewer, My folders**: the default tab, file checkboxes, the upload link. (Stories 1 and 2.)
4. **Viewer, results location**: location, Open folder, copy, run zip, and the job zip without the
   video. (Story 7.)
5. **Missing videos**: the check at job start, rerun `?check`, `video_available`, and the player
   message. (Story 4.)
6. **Selection datasets**: the migration and `server_selection` through to the picker. (Story 5.)
7. **Docker**: compose, `docker run` docs, and host paths; run the quickstart's Docker section.
   (Story 3.)
8. **CLI `process`** in place, then docs, CHANGELOG, and the rc1 walkthrough update.

## Complexity Tracking

| Deviation | Why needed | Simpler alternative rejected because |
|---|---|---|
| The job artifacts zip no longer includes the video by default (opt back in with `include_video=true`). | FR-029. The zip was itself an extra copy of sensitive video. | Keeping the old default leaves a copy in every download, against the feature's purpose. The change is announced in the CHANGELOG, with an opt-in. |
| Docker compose publishes on `127.0.0.1` instead of all interfaces. | R6. "Same machine" under Docker is only true if nothing else can reach the port. | Keeping all-interface publishing would make in-place reading readable from the network, or leave Docker without in-place reading. Sharing a server with a colleague is a lab setup, documented as removing the flag. |
