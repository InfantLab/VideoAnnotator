# Implementation Plan: Video Page

**Branch**: `1.7-dev` (spec `028-video-page`) | **Date**: 2026-10-19 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/028-video-page/spec.md` (not separately clarified:
its one open choice, how a `<video>` element authenticates, was left to this plan and is settled
by spec 027's stream token)

## Summary

The results player becomes the **Video page**: one video, every run's results on labelled lanes
that are also the controls, the spec 025 table beside it, notes of the researcher's own, and the
video streamed from the server instead of downloaded in a zip after a folder question.

Approach (from [research.md](research.md)): the page is `/video?path=…`, keyed by the video's
server path (R1); `/view/:jobId` redirects there. One new request, `GET /videos/runs`, lists
every run on the video with its files, and the projects that hold it (R2). The video plays
through 027's stream token (R3); result files are fetched one at a time and parsed by today's
parsers (R4). A TypeScript twin of the spec 025 row builder, checked against the exporter on a
shared fixture, feeds both the Data tab and the lanes, so they can't disagree (R5). Lanes are
canvas-drawn per kind with span, moment, numeric and presence forms, zoom and pan, density when
crowded (R6, R7). Notes are a new `video_notes` table and API, keyed to the video, and enter the
table (format version 2, additive) and ELAN exports (R9, R10).

## Technical Context

**Language/Version**: Python 3.12/3.13 (server); TypeScript, React 18 (viewer, Vite, Bun)
**Primary Dependencies**: FastAPI, SQLAlchemy, Pydantic (existing); viewer: TanStack Query,
React Router, shadcn/ui, existing parsers (existing). No new dependency.
**Storage**: server SQLite: new table `video_notes`; result folders (spec 022) unchanged; lane
preferences in the browser (`idbKv`).
**Testing**: pytest (unit: notes, runs lookup, table v2, ELAN notes tier; api: `/videos/runs`,
notes; a fixture writer for the twin's expected rows); Vitest (row twin vs fixture, lane
building, timeline maths, Data tab, keys, notes, routes); manual walk-through
[quickstart.md](quickstart.md).
**Target Platform**: the API server on Linux, macOS, Windows; current Chrome, Firefox, Safari.
**Project Type**: web application (Python API + React viewer in one repo).
**Performance Goals**: playback within 3 s on a local network whatever the size (SC-001); zoom,
pan and play smooth with 50,000 events over an hour (SC-004: no frame over 50 ms while panning);
`/videos/runs` under 1 s with 5,000 jobs.
**Constraints**: local-first (the stream and files come from the user's server only); initial
bundle within 300 KB gzipped (the Video page is code-split); existing keys, routes and output
formats keep working (Principle V); nothing drawn that the pipeline didn't produce (Principle VI).
**Scale/Scope**: one lab server; videos up to hours long and several GB; a video with up to
tens of runs; up to 50,000 events per video.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design: still passes.*

| Gate | Status | How |
|---|---|---|
| **I. Local-first, no telemetry** | ✅ | The video streams from the user's own server (027's token); results come from the same server; standalone mode keeps working offline. No new outbound traffic. |
| **II. Stable pipeline contract, open formats** | ✅ | No pipeline or output change. The viewer reads the same files with the same parsers. The table gains a column at the end and a measure (format version 2), which the spec 025 contract allows; the viewer contract test gains the expected-rows fixture. |
| **III. Provenance** | ✅ | No new annotation from a pipeline. About shows each lane's provenance; compare lists differing settings from provenance. Notes record author and times and are exported marked as notes, never as pipeline output. |
| **IV. Modular** | ✅ | No new dependency. Each lane kind is a self-contained layer (row mapping, form, overlay) talking to the timeline through the Lane interface; an unknown measure gets a generic lane. The page is code-split. |
| **V. Backward compatibility** | ✅ | `/view/:jobId` redirects with its job as `run=`; `/viewer` stays. Every existing key keeps its meaning (R8). Old overlay preference keys map onto lane kinds. Table v1 columns unchanged in name, order and meaning; exports and APIs only add. |
| **VI. Faithful display** | ✅ | Lanes and table come from one row builder twinned with the exporter (R5). Presence is never bridged beyond the sampling interval; density is labelled as density; the held VLM answer says when it was sampled; the box-area lane keeps its "not movement" label. Notes have their own lane and style, never inside a pipeline's. Every lane names its pipeline and run, and About gives version and settings. |
| **Engineering standards** | ✅ | Tests per area (below); API typed; docs: a Video page how-to replacing the results player pages, the table contract v2, the API reference, CHANGELOG; viewer `tsc`, ESLint, Vitest, bundle check; `viewer_static` rebuilt. |

No violations; Complexity Tracking is empty.

## Project Structure

### Documentation (this feature)

```
specs/028-video-page/
├── spec.md
├── plan.md              # this file
├── research.md          # what exists; decisions R1–R17
├── data-model.md        # video_notes, /videos/runs, table v2, Lane, preferences, URL
├── quickstart.md        # walk-through per user story
├── contracts/
│   ├── api.md           # /videos/runs, /videos/notes, export additions
│   ├── table-v2.md      # spec 025 table, format version 2
│   └── viewer.md        # routes, lane kinds, behaviour the tests check
├── checklists/requirements.md
└── tasks.md             # /speckit-tasks (not created here)
```

### Source Code

```
src/videoannotator/
├── api/v1/videos.py         # GET /videos/runs; notes routes (or video_notes.py beside it)
├── video_runs.py            # NEW runs and projects for a path (R2)
├── video_notes.py           # NEW create, list, update, delete; lookup for exports (R9)
├── database/models.py       # VideoNote
├── exporters/readers.py     # TABLE_FORMAT_VERSION 2, COLUMNS + author, note rows
├── exporters/table.py       # notes once per video per export (R10)
├── exporters/elan.py        # Notes (researcher) tier
└── api/v1/exports.py        # pass notes; summary line when no database

viewer/src/
├── pages/VideoPage.tsx              # NEW (lazy): /video; also renders /viewer's standalone mode
├── pages/JobResultsViewer.tsx       # becomes the /view/:jobId redirect
├── components/video/                # NEW Header (context, ‹ ›, projects), SidePanel (Data,
│                                    #     Notes, Runs, About), DataTable (windowed), NotesList,
│                                    #     AboutPanel, KeysHelp
├── components/lanes/                # + LaneCanvas, LaneHeader (label, eye, run menu),
│                                    #   LayersMenu, RangeBar, Timeline (zoom/pan); 027's parts kept
├── components/VideoPlayer.tsx       # src from a URL as well as a File; overlays by lane kind
├── lib/table/rows.ts                # NEW twin of the spec 025 row builder (R5)
├── lib/lanes.ts                     # + view range, lane building from rows, density (R6, R7)
├── lib/laneKinds.ts                 # NEW kind table (contracts/viewer.md)
├── lib/settingsDiff.ts              # NEW (R11; reused by 029)
├── lib/lanePrefs.ts                 # NEW (R13)
├── hooks/useVideoRuns.ts, useVideoStream.ts, useRunFiles.ts, useNotes.ts, useVideoKeys.ts
├── components/BatchNavigation.tsx   # → ListNavigation (run, project, folder)
├── App.tsx                          # /video; /view/:jobId redirect
└── removed from the page: UnifiedControls, OverlayControls, TimelineControls, FileViewer,
    VlmAnnotationPanel (files deleted once nothing imports them)

tests/
├── fixtures/viewer_contract/expected_table.csv   # NEW, written by the exporter
├── fixtures/video_page/long_run/                 # NEW synthetic hour, 50,000 events (generated)
├── unit/test_video_notes.py, test_video_runs.py, test_table_v2.py, test_elan_notes.py
├── unit/test_viewer_table_fixture.py             # regenerates expected_table.csv; fails on drift
└── api/test_video_runs_api.py, test_video_notes_api.py, test_exports_notes.py
```

**Structure Decision**: the existing web-application layout: server code in
`src/videoannotator/`, viewer in `viewer/` built into `viewer_static/` (rebuilt and committed,
CI checks it).

## Order of work

Each stage ships on its own and keeps the suite green.

1. **Server** (R2, R9, R10): `video_runs` and `GET /videos/runs`; `VideoNote`, `video_notes` and
   the notes API; table v2 (column, measure, version) with its contract; ELAN notes tier; exports
   pass notes and say when they can't. The expected-rows fixture and its test.
2. **Rows and lanes** (R5, R6, R13): `lib/table/rows.ts` passing the fixture; `laneKinds`; lane
   building; preferences. Pure code, unit-tested before any UI.
3. **The page** (R1, R3, R4, R7, R8, R12, R14–R16), in user-story order: loading and streaming
   (US4) with the `/view/:jobId` redirect; the timeline with lanes as controls (US1, US2); Data
   tab (US3); runs and compare (US5); notes (US6); context and header (US7). Standalone `/viewer`
   on the same page. Remove the old controls, modal and panel. Contract fixture; rebuild
   `viewer_static`.
4. **Docs and checks**: Video page how-to (replacing the results-player docs), table contract
   v2 in the docs, API reference, CHANGELOG, roadmap v1.7.0 Phase 1 tick; the quickstart
   walk-through recorded in `specs/028-video-page/checks.md`, including SC-001, SC-003, SC-004.

## Risks

- **Twin drift** (R5): a reader change in Python without the TS twin breaks the fixture test,
  which is the point; the fixture covers every reader's demo-clip output, not the long tail of
  older formats (those still have the existing contract test).
- **Long videos in the browser**: the parsers hold a run's files in memory; an hour of person
  tracking at full rate is tens of MB of JSON. Acceptable for 028; if it isn't in the pilot,
  the server can serve rows per time window later without changing the page's model.
- **`/videos/runs` scans jobs**: linear in jobs, as `has_video_path` is; spec 029's video list
  adds the index.
- **Path as identity** (R1): a file uploaded in two jobs is two videos until spec 029 joins them
  by hash; their notes stay apart, never merged silently (029's edge case).
- **Safari and range requests**: as for 027, to verify; H.265 videos get the codec message.
- **Spec 023's journeys** are checked through pilot feedback after release (023, clarified
  2026-10-19); layout may change then, not the server or row work.

## Complexity Tracking

None.
