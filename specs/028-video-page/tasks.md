---
description: "Tasks for spec 028: Video Page"
---

# Tasks: Video Page

**Input**: `specs/028-video-page/` (plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md)
**Tests**: included. The constitution requires tests for new API surface; the row twin needs a
shared fixture (research R5); SC-004 and SC-005 are checked by tests.
**Organization**: by user story, in the spec's priority order (US1–US4 are P1, US5–US7 P2).
Within a story, server before viewer, tests beside the code they test.

**Conventions for every task**: Python passes `ruff check`, `ruff format`, `mypy src/videoannotator`;
viewer passes `bun run lint`, `tsc --noEmit -p tsconfig.app.json`, `bun run test:run`. Note text and
pipeline labels are user- or pipeline-supplied strings: never `dangerouslySetInnerHTML`. No comments
saying *what* code does. User-visible words follow spec 023 Decision 1: "video", "run", "notes",
"lane"; never "job", "batch", "dataset", "annotation file" (029 removes the rest). Commit per task
or small group, `<area>: <summary>`, with the Co-Authored-By trailer; the container has no git
identity (`git -c user.name=… -c user.email=…`).

## Format: `[ID] [P?] [Story] Description`

---

## Phase 1: Setup

- [X] T001 Create `tests/unit/test_viewer_table_fixture.py`: builds rows with `exporters.table.video_rows` for the demo-clip files in `tests/fixtures/viewer_contract/` (one `VideoResults` with `run=""`, `video="demo_clip"`), writes them as CSV (`table.write_csv` format) to a temp file, and compares with the committed `tests/fixtures/viewer_contract/expected_table.csv`; with `UPDATE_FIXTURES=1` it rewrites the committed file instead. Run it once with `UPDATE_FIXTURES=1` and commit the CSV
- [X] T002 [P] Create `scripts/make_long_run_fixture.py` that writes `tests/fixtures/video_page/long_run/` deterministically (seeded): an hour-long COCO person-tracking file with 4 tracks sampled at 5 fps (≈ 40,000 detections over a 3,600 s video), a WebVTT with 2,000 cues, an RTTM with 3 speakers and 6,000 turns, and a `README.md` saying how it was made; the files are generated in CI by the Vitest setup if absent (do not commit files over 5 MB; commit the script and README)

---

## Phase 2: Foundational (blocks every story)

### Server: the runs on a video

- [X] T003 Create `src/videoannotator/video_runs.py`: `runs_for(storage, path: Path) -> list[RunInfo]` (every job whose `Path(job.video_path).resolve() == path`, newest first, with `job_id`, `status`, `created_at`, `completed_at`, `batch_id`, `dataset_id`, `run_name` (spec 022 run folder name from `job.output_dir`'s parent when it has a run record, else None), and per pipeline `name`, `status`, `error`, `files` (from `jobs.pipeline_result_files`; `kind` = `provenance` for `*.provenance.json`, `result` for files `exporters.readers.classify` knows, else `other`; `download_url` as `GET /jobs/{id}/results` builds it)); `projects_for(db, path, runs) -> list[{id, name}]` (datasets named by the runs' `dataset_id`, plus datasets whose `server_folder` contains the path, or whose manifest lists its file name when `server_folder` is set and the file is in it); `sha256_for(runs) -> str | None` (the first `input.sha256` in any of the runs' provenance files)
- [X] T004 In `src/videoannotator/api/v1/videos.py`: `GET /videos/runs?path=` per contracts/api.md (response models `VideoRunsResponse`, `RunInfo`, `PipelineRunInfo`, `RunFile`, `ProjectRef`); resolve the path like `readable_video` but return 200 with `available: false` and `unavailable_reason` when the file is missing and some job read it; 403 `PATH_NOT_ALLOWED` when neither readable nor read by a job; `duration_sec` from `video_duration` when available, else the newest job's `video_duration_seconds`
- [X] T005 [P] Tests: `tests/unit/test_video_runs.py` (two jobs on one path via a symlink resolve to one list, newest first; a job on another path is excluded; failed pipeline carries its error; provenance file kind; project by `dataset_id` and by folder; `sha256_for` reads a provenance fixture) and `tests/api/test_video_runs_api.py` (200 with runs; 200 `runs: []` for a readable video with none; 200 `available: false` for a moved video a job read; 403; 422 empty path; needs the API key)

### Viewer: rows, lanes, the page shell

- [X] T006 Create `viewer/src/lib/table/rows.ts`: `type Row` with the 19 version-1 columns plus `author` (contracts/table-v2.md), and `rowsFromParsed(parsed: StandardAnnotationData, files: {name: string; provenance?: Provenance}[], video: string, run = ''): Row[]` producing exactly what `exporters/readers.py` + `table.video_rows` produce: per-reader mapping per spec 025's contract table (person, face, emotion, emotion_score, age, gender, OpenFace 3 measures, scene, scene_score, speaker_turn, speech, vlm_label, vlm_reasoning), `event` numbering, `model`/`videoannotator_version` from provenance (`not recorded` without), `file`, and the contract's sort order. Read `readers.py` and `table.py` side by side while writing it; keep numbers unrounded
  - *Built as `rowsFromFiles(files: {name, text}[], video, run)`*: the parsed structures are lossy (OpenFace 3 parsing zero-fills absent action units and drops `track_id`/`frame_number`; `openface3.timestamp` is replaced by the image's), so the twin ports the readers over the files' own text. `useRunFiles` passes both the parsed data (overlays) and the text (rows).
- [X] T007 [P] Create `viewer/src/test/lib/tableRows.test.ts`: parse the files in `tests/fixtures/viewer_contract/` with the existing parsers (as `videoannotator-outputs.test.ts` does), build rows with `rowsFromParsed`, serialise as CSV, and compare cell by cell with `tests/fixtures/viewer_contract/expected_table.csv` (empty `author` column ignored until T047 adds it to the fixture); on mismatch print the first differing row and column
- [X] T008 Create `viewer/src/lib/laneKinds.ts`: the kind table from contracts/viewer.md (`kind`, `label`, measures, `form`, row key, `overlay` key, order) and `kindOf(row): LaneKind` with the `other:<pipeline>/<measure>` fallback; `person_area` is derived (not from a measure) and labelled "People · box area (not movement)"
- [X] T009 Extend `viewer/src/lib/lanes.ts`: `buildLanes(rows: Row[], runs: RunRef[], failed: Record<string,string>): Lane[]` per data-model.md (group by kind, pipeline and run: a kind written by two pipelines in one run, such as `face` from face_analysis and OpenFace 3, makes two lanes, each naming its pipeline; rows by key with `categoryColors` colour, events sorted by start, each event keeps its row index; failed pipelines with no rows give a `failed` lane; a face `count` numeric row = faces per sampled frame time; presence events per track joined only while consecutive samples are within the sampling interval (Principle VI)); add view-range variants `viewPercent(t, view)`, `timeAtClickInView(clientX, el, view)`, `visibleSlice(events, view)` (binary search) and `densityBins(events, view, columns)`; keep the existing exports working for 027's lanes
- [X] T010 [P] Create `viewer/src/test/lib/buildLanes.test.ts`: from the demo-clip rows, one lane per kind present and none for absent; speakers lane has a row per speaker; failed pipeline lane; presence not bridged across a gap larger than the interval; unknown measure → `other:` lane; `visibleSlice` and `densityBins` on 50,000 synthetic events return in < 20 ms
- [X] T011 Create `viewer/src/lib/lanePrefs.ts`: load/save per data-model.md (`idbKv`, key `lanePrefs:<server url>:<user id>`, `lanePrefs:local` standalone), with defaults for unknown kinds and the one-time mapping from old `OverlaySettings` keys; test in `viewer/src/test/lib/lanePrefs.test.ts` (round trip, unknown kind defaults, old keys mapped, storage unavailable → defaults without throwing)
- [X] T012 In `viewer/src/api/client.ts`: `getVideoRuns(path)`, `getStreamToken(path)` (exists from 027: reuse), `downloadResultFile(url): Promise<File>`; regenerate `viewer/src/api/schema.d.ts` from the running server's OpenAPI (the command 027 used: see `viewer/package.json` scripts)
- [X] T013 Create `viewer/src/hooks/useVideoRuns.ts` (TanStack Query on `getVideoRuns`) and `viewer/src/hooks/useRunFiles.ts` (given the chosen job per pipeline, fetch only those pipelines' `result` and `provenance` files, parse with `mergeAnnotationData`, return parsed data per run plus rows from `rowsFromParsed`; cache per job and file name for the session; no write to the local library)
- [X] T014 In `viewer/src/components/VideoPlayer.tsx`: accept `videoUrl?: string` as well as `videoFile` (URL used directly as `src`; codec sniffing skipped for URLs, playback errors still give the codec message); overlays read visibility from a `overlays: Record<string, boolean>` keyed by lane kind instead of `OverlaySettings` (map inside the component so `/viewer` keeps working until T047)
- [X] T015 Create `viewer/src/pages/VideoPage.tsx` (lazy-loaded) and the route `/video` in `viewer/src/App.tsx`: reads `path`, `from`, `run`, `t`; layout per spec (header row; video left; side panel right with tabs Data, Notes, Runs, About, collapsible; timeline below); uses `useVideoRuns`, a stream URL from `getStreamToken`, `useRunFiles` with the default run per pipeline (newest whose pipeline status is `completed`), `buildLanes`, `lanePrefs`. Tabs may be placeholders until their story. Check the initial bundle stays within budget (`bash scripts/build_viewer.sh --check`)

**Checkpoint**: `/video?path=<demo clip path>` plays the streamed video with overlays; suite green.

---

## Phase 3: User Story 1 – Read the timeline like ELAN (P1) 🎯 MVP

**Goal**: labelled lanes, a row per person or speaker, zoom, hover, next/previous event, numeric charts.
**Independent test**: open the demo video; every lane and row is named; zoom to ten seconds; `]` jumps to the next utterance.

- [ ] T016 [US1] Create `viewer/src/components/lanes/LaneCanvas.tsx`: draws one lane's rows into a `<canvas>` for the current view per form (span bars, moment ticks, presence bars, numeric line with min/max labels and unit, density bars labelled "density" when events exceed visible columns), device-pixel-ratio aware, redrawn on view/data/size change only; exposes hit testing (`eventAt(x, y)`) via the same `visibleSlice`
- [ ] T017 [US1] Create `viewer/src/components/lanes/LaneHeader.tsx` (label with kind, pipeline when another lane of the kind comes from a different pipeline, and run; row labels with colour swatches; a tooltip on every lane and row label saying what it shows and which pipeline made it (SC-002); selection state) and `viewer/src/components/lanes/Timeline.tsx` (name it `LaneTimeline` if `components/Timeline.tsx` is still imported): the axis (027's `TimeAxis` given the view), lanes, `Playhead`, hover card (label, text, start–end, confidence) and click-to-seek
- [ ] T018 [US1] Create `viewer/src/components/lanes/RangeBar.tsx` and zoom/pan in `LaneTimeline`: Ctrl/⌘+wheel zoom around the pointer, wheel/drag pan, `+`/`−` around the playhead, `0` fit; the view pages forward when the playhead leaves it while playing; minimum view 1 s
- [ ] T019 [US1] Create `viewer/src/hooks/useVideoKeys.ts`: page-wide keys per research R8 (existing Space, ←/→, Shift+←/→, ↑/↓, Ctrl+Shift+D; new `[`/`]` previous/next event of the selected lane, `+`/`−`/`0`); ignored when focus is in an input, textarea, select or contenteditable; and a `KeysHelp` popover (`viewer/src/components/video/KeysHelp.tsx`) behind a `?` button
- [ ] T020 [US1] In the player readout (`viewer/src/components/VideoControls.tsx` or the page's control row): show the frame number (`Math.round(t × fps)`, fps from the parsed `video_info` or the first result's metadata; omit when unknown) and ⏮ ⏭ event buttons wired to the selected lane
- [ ] T021 [P] [US1] Tests `viewer/src/test/components/laneTimeline.test.tsx`: one lane per kind present; row labels; zoom changes the range bar; `]` seeks to the next event of the selected lane and not another's; hover shows text and times; click seeks; keys ignored in an input; every lane and row label has a tooltip; a run with both face pipelines gives two Faces lanes, each naming its pipeline
- [ ] T022 [P] [US1] Performance test `viewer/src/test/lib/lanePerformance.test.ts` on the long-run fixture (T002): building lanes < 500 ms; `visibleSlice` + `densityBins` for 200 successive pan steps < 1 s total (a proxy for SC-004; the real check is quickstart)

**Checkpoint**: US1's acceptance scenarios pass on the demo video.

---

## Phase 4: User Story 2 – Toggle from the lanes, not a card (P1)

**Goal**: lanes and overlays toggled from labels, eyes and Layers; remembered.
**Independent test**: every overlay and lane can be shown and hidden from the lanes or Layers; no controls card.

- [ ] T023 [US2] In `LaneHeader`: clicking the label collapses/expands (collapsed = label row only); an eye toggles the overlay for kinds with one (none for others); Faces gets an overlay menu with the OpenFace 3 toggles from `OpenFace3Controls` (landmarks, gaze, head pose, action units); all saved through `lanePrefs`
- [ ] T024 [US2] Create `viewer/src/components/lanes/LayersMenu.tsx`: every lane with both toggles; "all" and "none" for lanes and for overlays
- [ ] T025 [US2] Timeline height: a drag handle on the timeline's top edge (min 120 px, max 70% of the viewport), saved in `lanePrefs`
- [ ] T026 [US2] Layout budget at 1440×900 (SC-003): header and player controls in one 48 px row, no footer on the Video page, default timeline height 200 px with lanes scrolling inside it, side panel 360 px; measure the video area on the old player and the new page and record both in `specs/028-video-page/checks.md`
- [ ] T027 [P] [US2] Tests `viewer/src/test/components/laneToggles.test.tsx`: label collapses; eye hides that overlay only; no eye on kinds without an overlay; Layers none/all; preferences restored after remount; the page renders no `UnifiedControls`

**Checkpoint**: US2's scenarios pass; the video column fills the space the card used.

---

## Phase 5: User Story 3 – The values beside the video (P1)

**Goal**: the Data tab: rows of the spec 025 table following playback.
**Independent test**: the highlighted row tracks the playhead; clicking a row seeks; filter to Speakers.

- [ ] T028 [US3] Create `viewer/src/components/video/DataTable.tsx`: windowed list (fixed row height; renders only visible rows plus a margin) of the rows from `useRunFiles` for the shown runs, columns time, lane, row, value/label (full row in a hover/expand); highlights the row at or before the playhead (binary search on start) and scrolls it into view until the user scrolls; a "Follow playback" button resumes; lane filter and search (label, value, track; case-insensitive); "n of N rows"
- [ ] T029 [US3] Link table and timeline (SC-005): clicking a row seeks to its start and selects its lane; clicking an event in a lane selects its row (via the event's row index)
- [ ] T030 [P] [US3] Tests `viewer/src/test/components/dataTable.test.tsx`: same rows and order as `rowsFromParsed`; highlight follows `currentTime`; user scroll stops following; filter and search counts; 50,000 rows render fewer than 200 DOM rows; row click seeks; event click selects row

**Checkpoint**: US3's scenarios pass.

---

## Phase 6: User Story 4 – Plays at once, no folder question (P1)

**Goal**: streaming everywhere results are opened; old addresses and standalone mode keep working.
**Independent test**: a 2 GB video plays without downloading the whole file; no folder dialog.

- [ ] T031 [US4] Create `viewer/src/hooks/useVideoStream.ts`: token from `getStreamToken`, stream URL; on a media error with HTTP 401 (probe with a `HEAD`/range `GET` when the element errors) or within 5 minutes of `expires_at`, fetch a new token, swap `src`, restore `currentTime` and play state; returns the missing-video state from `useVideoRuns` (`available: false`) without requesting a token. Wire it into `VideoPage`
- [ ] T032 [US4] Turn `viewer/src/pages/JobResultsViewer.tsx` into the `/view/:jobId` redirect: fetch the job, `navigate('/video?path=…&run=<jobId>[&from=run:<batch_id>]', {replace: true})`; demo jobs (`isDemoJobId`) keep today's local-library behaviour until spec 029; job not found → the existing error card
- [ ] T033 [US4] Point every in-app link that opened `/view/<job id>` (Jobs, Job detail, Batch detail, Results, Compare, Ask AI run links: `grep -rn "/view/" viewer/src`) at `/video?path=…&run=…` directly
- [ ] T034 [US4] Standalone `/viewer` (`viewer/src/pages/Index.tsx`): render `VideoPage` in local mode with the dropped `File`s (video by object URL, results through `rowsFromParsed` with `run = ''`); no runs menu, no context, Notes lane says notes need a server; keep the file uploader and drag-and-drop as today
- [ ] T035 [P] [US4] Tests `viewer/src/test/pages/videoPage.loading.test.tsx`: `src` is a `/api/v1/videos/stream` URL; no `/artifacts` request and no folder prompt; only the chosen pipelines' files fetched; token refresh keeps time; missing video shows lanes and the message; `/view/:jobId` redirects with `run=`; `/viewer` renders with local files

**Checkpoint**: US4's scenarios pass; the zip is no longer downloaded for server results.

---

## Phase 7: User Story 5 – Which run, and comparing runs (P2)

**Goal**: each lane names its run; another run or a comparison from the lane's menu.
**Independent test**: faces run twice with different thresholds; both lanes shown; About lists the threshold.

- [ ] T036 [US5] Create `viewer/src/lib/settingsDiff.ts`: `settingsDiff(a: Provenance, b: Provenance): {key, a, b}[]` over the pipeline's recorded `settings` (nested keys flattened with `.`; equal values omitted; missing provenance → `not recorded`); tests in `viewer/src/test/lib/settingsDiff.test.ts`
- [ ] T037 [US5] In `LaneHeader`, the run menu: lists the runs with this pipeline (date, status, run name; failed ones disabled with their reason); choosing one sets the lane's run (lane, overlay, table rows switch; saved in page state and the `run` URL parameter only when all lanes share one); "Compare with…" adds a lane of the same kind for the other run directly below, labelled with its run, removable
- [ ] T038 [US5] Create `viewer/src/components/video/AboutPanel.tsx`: per lane, pipeline, model, version, settings and run from provenance (`not recorded` without), the raw files with download links (replacing `FileViewer`); for a compared pair, the differing settings from `settingsDiff`
- [ ] T039 [P] [US5] Tests `viewer/src/test/components/runs.test.tsx`: default newest completed run; `run=` override; switching run changes rows; compare adds an adjacent lane; About lists the differing threshold

**Checkpoint**: US5's scenarios pass.

---

## Phase 8: User Story 6 – Notes, light touch (P2)

**Goal**: moment and span notes on the server, their own lane, in exports.
**Independent test**: add two notes, reload: there; export ELAN: a tier of their own.

### Server

- [ ] T040 [US6] In `src/videoannotator/database/models.py`: `VideoNote` (table `video_notes`) per data-model.md, index on `video_path`; created by the existing `create_all`; test in the existing models test that the table is created
- [ ] T041 [US6] Create `src/videoannotator/video_notes.py`: `create(db, *, path, start, end, text, user) -> VideoNote` (trims text; swaps start/end when reversed; validates per contracts/api.md; fills name and `video_sha256` via `video_runs.sha256_for` when known), `list_for(db, path)`, `update(db, id, fields, user)`, `delete(db, id)`, `for_media_path(db, path) -> list[VideoNote]` (resolved), `to_dict`
- [ ] T042 [US6] Routes in `src/videoannotator/api/v1/videos.py` (or `video_notes` router included under `/videos/notes`): `GET`, `POST`, `PATCH /{id}`, `DELETE /{id}` per contracts/api.md; path rule as `/videos/runs`; author from the signed-in user (`validate_api_key`: id and name/email)
- [ ] T043 [P] [US6] Tests `tests/unit/test_video_notes.py` (validation, swap, trim, order, sha filled when known) and `tests/api/test_video_notes_api.py` (CRUD round trip; 422 codes; 404; another signed-in user may edit and is recorded as `updated_by`; needs the API key)

### Exports

- [ ] T044 [US6] Table format v2 in `src/videoannotator/exporters/readers.py` and `table.py`: `TABLE_FORMAT_VERSION = 2`; `author` appended to `COLUMNS` and `Row` (default empty); `table.rows()` takes an optional `notes_for: Callable[[VideoResults], list[VideoNote-like]]` and yields each video's notes once per export as `note` rows (`run` empty, `video` = file stem, pipeline `notes`) per contracts/table-v2.md; `ExportSummary` gains `notes` count and the "notes not included: no database" line; update `docs/` table contract (spec 025's doc page) and the Parquet schema
- [ ] T045 [US6] ELAN in `src/videoannotator/exporters/elan.py`: `tiers()` puts `note` rows in a `Notes (researcher)` tier (spans as given; moments as `_in_view` does for single times), `ANNOTATOR` = author when one author, else one tier per author `Notes (researcher: <name>)`
- [ ] T046 [US6] In `src/videoannotator/api/v1/exports.py` (and the CLI export command, `grep -n "write_csv\|elan.build" src/videoannotator/cli.py`): pass `notes_for` looking up `video_notes.for_media_path` by `VideoResults.media_path` (falling back to the job's `video_path`); in the CLI, when the database can't be opened, pass none and let the summary say so
- [ ] T047 [P] [US6] Tests `tests/unit/test_table_v2.py` (columns 1–19 unchanged; `author` last; note rows once per video across two runs; version 2 in summary and Parquet metadata), `tests/unit/test_elan_notes.py` (tier, annotator, two authors), `tests/api/test_exports_notes.py` (job and batch table/ELAN include notes; no notes → no tier, no rows). Then rerun T001 with `UPDATE_FIXTURES=1` (the fixture gains the empty `author` column) and drop the T007 exception

### Viewer

- [ ] T048 [US6] Create `viewer/src/hooks/useNotes.ts` (list/create/update/delete with optimistic update and rollback on error) and client methods in `viewer/src/api/client.ts`
- [ ] T049 [US6] Notes on the page: `N` creates a moment note at the current time and opens its editor focused (Enter saves, Esc on empty cancels); `Shift+N` starts a span (a marker shows) and the second `Shift+N` ends it and opens the editor; a "Note" button does the same; the Notes lane is always last, outlined and labelled "Notes (yours)", never inside a pipeline lane; clicking a note opens a popover to edit or delete
- [ ] T050 [US6] Create `viewer/src/components/video/NotesList.tsx` (the Notes tab: time, text, author; edit, delete, click to seek) and add note rows to the Data tab (`rowsFromParsed` output plus notes as `note` rows, styled as notes)
- [ ] T051 [P] [US6] Tests `viewer/src/test/components/notes.test.tsx`: N creates and focuses; Esc cancels empty; Shift+N twice makes a span in either order; typing "n" in search makes no note; notes lane last and styled distinctly; note rows in the Data tab; standalone shows "needs a server"

**Checkpoint**: US6's scenarios pass; table and ELAN exports carry notes.

---

## Phase 9: User Story 7 – Where this video belongs (P2)

**Goal**: the list it was opened from, with previous/next; otherwise its projects.
**Independent test**: from a project, ‹ › move through it; from a folder, through the folder.

- [ ] T052 [US7] Replace `viewer/src/components/BatchNavigation.tsx` with `viewer/src/components/ListNavigation.tsx`: for `from=run:<batch id>` (batch jobs, existing endpoint, in their order), `from=project:<dataset id>` (dataset videos via `GET /datasets/{id}` and its stored videos), `from=folder:<path>` (`GET /ingest/browse`, videos only, name order): the list's name, "3 of 48", ‹ › opening `/video?path=<neighbour>&from=…` (lane preferences carry over, being per user); update its importers
- [ ] T053 [US7] Create `viewer/src/components/video/Header.tsx`: back link to the list (or Videos/Jobs until 029), file name, `ListNavigation` or, without `from`, the video's projects from `/videos/runs` as links; actions Ask AI (`/ask?video=<path>`), Run… (navigate to `/jobs/new` with a `WizardStart` for this video, see `lib/wizardStart.ts`), Export (`ExportMenu` for the shown run's job; one entry per run when lanes show several), ⋯ (debug panel, load different files)
- [ ] T054 [P] [US7] Tests `viewer/src/test/components/videoHeader.test.tsx`: "3 of 48" and neighbours for each `from` kind; projects listed without `from`; Ask AI link carries the path; Export lists each shown run

**Checkpoint**: US7's scenarios pass.

---

## Phase 10: Polish & cross-cutting

- [ ] T055 Ask AI lane overlay (FR-022): on the video, the latest `vlm_label` at or before the playhead, held no longer than the sampling interval, marked "sampled at m:ss.s" (`lib/sampledAtTime.ts`); remove `VlmAnnotationPanel` from the page
- [ ] T056 Remove from the page and then delete once unimported: `UnifiedControls`, `OverlayControls`, `TimelineControls`, `FileViewer`, `VlmAnnotationPanel`, `OpenFace3Controls` (its settings now in the Faces menu), the old `Timeline` if unused (`grep -rn` each); move `DebugPanel` and "Load different files" into ⋯; update `VideoAnnotationViewer` users or delete it if `VideoPage` replaced all
- [ ] T057 [P] Viewer contract test (`viewer/src/test/contract/videoannotator-outputs.test.ts`): lanes build from every fixture without an `other:` lane for a known measure
- [ ] T058 [P] Empty and error states: a video with no runs shows the video, an empty timeline and "Run…"; a failed pipeline's lane shows its reason; `/videos/runs` errors show the API's message and hint
- [ ] T059 Docs: a Video page how-to in `docs/` replacing the results-player page(s) (find them with `grep -rln "View data\|results player\|UnifiedControls" docs/`), the table contract v2 (spec 025's doc page), the API reference for `/videos/runs` and `/videos/notes`, `CHANGELOG.md` (Unreleased: Video page; table format version 2 adds `author` and `note` rows)
- [ ] T060 Bundle and build: `bash scripts/build_viewer.sh`; confirm the initial bundle stays within 300 KB gzipped (Video page in its own chunk); commit `src/videoannotator/viewer_static/`
- [ ] T061 Full gates: `pytest tests/`, `ruff check .`, `mypy src/videoannotator`, `pre-commit run --all-files`, viewer lint/tsc/tests
- [ ] T062 Walk through `quickstart.md` (setup, US1–US7, performance) on the demo clip and a large video; record results, SC-001 time to first frame, SC-003 areas before and after at 1440×900, SC-004 panning frames, and anything that didn't match, in a new `specs/028-video-page/checks.md`
- [ ] T063 Update `docs/development/roadmap_v1.7.0.md` Phase 1 (028 done; link checks.md) and `CLAUDE.md` Current Plan (next: spec 029); mark spec 028 Status "Built"

---

## Dependencies and execution order

- **Phase 1 → Phase 2**: T001 before T007; T002 before T022.
- **Phase 2** blocks all stories. Inside it: T003 → T004 → T005; T006 → T007; T006 → T008 → T009 → T010; T012 → T013; T009, T011, T013, T014 → T015.
- **US1** (T016–T022) needs Phase 2. It is the MVP with Phase 2.
- **US2** needs US1's `LaneHeader` (T017); T026 needs US1's timeline and T020's control row.
- **US3** needs Phase 2 only; T029 also needs US1's timeline (T017).
- **US4** needs Phase 2; independent of US1–US3 (T034 reuses whatever lanes exist).
- **US5** needs US1 (T017) for the lane menu.
- **US6** server tasks (T040–T047) need only Phase 2 and can start at any time; its viewer tasks (T048–T051) need US1 (lanes) and US3 (Data tab, for T050).
- **US7** needs Phase 2 (T004 for projects).
- **Polish** last; T056 only after every story (each removes a piece of the old player).

## Parallel opportunities

- Phase 2: the server half (T003–T005) and the viewer half (T006–T015) run in parallel; T005, T007, T010, T011 touch separate files.
- The notes server and exports (T040–T047) are independent of all viewer work.
- US3 (Data tab) and US4 (loading) can run alongside US1.

Example (Phase 2):

```
T003 src/videoannotator/video_runs.py    ║ T006 viewer/src/lib/table/rows.ts
T004 api/v1/videos.py                    ║ T008 viewer/src/lib/laneKinds.ts
T005 tests (unit, api)                   ║ T011 viewer/src/lib/lanePrefs.ts
```

## Implementation strategy

1. **MVP**: Phases 1–3 (Phase 2 + US1). A researcher opens a video by address, it streams, and the
   timeline reads like ELAN. Stop and walk through quickstart US1.
2. **The P1 set**: US2 (toggles), US3 (Data tab), US4 (redirects, standalone, refresh). After these
   the old player's controls card and JSON modal can go.
3. **P2**: US5 (runs), US6 (notes; server half can land early), US7 (context).
4. Each story ends at its checkpoint with the full suite green and a commit.
