# Research: Video Page

**Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md) | **Date**: 2026-10-19

What the code does today, and the decisions this plan takes. Each decision says what was chosen,
why, and what else was considered.

## What exists (read 2026-10-19, at `42eda64`)

- **Route and loading** (`pages/JobResultsViewer.tsx`, `/view/:jobId`): one page per *job*, not
  per video. `useZipDownloader` downloads `GET /jobs/{id}/artifacts` (a zip with the video
  inside), asks for a local folder the first time (`needs_folder`), unzips, and parses every
  file with `mergeAnnotationData` (`lib/parsers/merger.ts`). The parsed result is cached per job
  in the browser's local library (`lib/localLibrary`). Standalone mode is `/viewer`
  (`pages/Index.tsx`) with dropped files and the same parsers.
- **Player** (`components/VideoAnnotationViewer.tsx`, 419 lines): `VideoPlayer` (946 lines,
  canvas overlays, plays a `File` through `createObjectURL`), `UnifiedControls` (the 384 px
  card; six hard-coded pipelines in `OverlaySettings`/`TimelineSettings`), `Timeline` (332
  lines; DOM rows plus a canvas for the "motion" row, which plots summed person-box area per
  0.1 s, labelled honestly since spec 023's fix), `FileViewer` (the JSON modal),
  `VlmAnnotationPanel`, `OpenFace3Controls`, `DebugPanel` (Ctrl+Shift+D). Settings live in React
  state only: nothing is remembered across videos.
- **Keys today** (`VideoControls.tsx`): Space play/pause, ←/→ one frame, Shift+←/→ one second,
  ↑/↓ a tenth of a second; Ctrl+Shift+D the debug panel. They work only while the controls card
  has focus. Principle V keeps all of them.
- **Lanes from 027** (`components/lanes/`, `lib/lanes.ts`): `TimeAxis`, `SampleTicks`,
  `AnswerLane`, `Playhead`, `LaneLabel`; whole-video only, percentage positions, DOM markers.
- **Server, per job**: `GET /jobs/{id}/results` lists each pipeline's status, error and files
  with a `download_url`; `GET /jobs/{id}/results/files/{pipeline}?name=` downloads one file;
  `GET /jobs/{id}/video` streams the video but needs the `Authorization` header.
- **Streaming from 027** (`api/v1/videos.py`): `POST /videos/stream-token {path}` returns an HMAC
  token valid for an hour; `GET /videos/stream?path=&token=` serves it with range requests
  (`FileResponse`). `readable_video` allows paths under the shared folders, uploads, and any
  path a job read.
- **No per-video view on the server.** Jobs record `video_path` (uploads are copied per job, so
  one file uploaded twice has two paths), `batch_id` and `dataset_id`; `storage.has_video_path`
  scans jobs to find one. Each result's provenance (spec 017) records the input's
  `sha256` (`batch/job_execution.py`).
- **Table (spec 025)**: `exporters/readers.py` turns each native file into rows with the
  `COLUMNS` of format version 1; `table.video_rows` per video; `results_set.VideoResults` knows
  the video's `media_path` from the run record. ELAN (`exporters/elan.py`) builds tiers from the
  same rows. Exports exist per job, batch and dataset (`api/v1/exports.py`).
- **Helpers to reuse**: `lib/categoryColors.ts` (stable colours by key), `lib/provenance.ts`
  (reads provenance), `lib/sampledAtTime.ts` (the held VLM sample), `lib/vlmCompare.ts`,
  `components/BatchNavigation.tsx` (previous/next in a batch), `components/ExportMenu.tsx`,
  `persistence/idbKv.ts`.
- **Dependencies**: recharts (already in the bundle, lazy), TanStack Query, zip.js. No timeline
  or virtual-list library.

## Decisions

### R1. A video is its server path; the page is `/video`

**Decision**: The Video page is `/video?path=<server path>`, optionally with `&from=` (R12) and
`&run=<job id>` (which run to start from). The server resolves the path (symlinks included) and
that is the video's identity in 028. The content `sha256`, when known from provenance or an
upload, is recorded next to it (notes, R9) so spec 029 can join copies of one file without a
migration. `/view/:jobId` keeps working: it looks the job up and replaces itself with
`/video?path=…&run=<jobId>` (Principle V). Standalone mode stays at `/viewer` and renders the
same page with local files.

**Rationale**: Every run already records the path; nothing has to be hashed before a page can
open (hashing a 2 GB file takes tens of seconds). 029's edge case says copies are joined "when
the server can tell, never silently merged", which a path-first key with the hash alongside
allows.

**Alternatives**: a `videos` table with an opaque id (029's to decide, with its list; the `path`
parameter keeps working when it lands); the hash as identity (slow first open; two uploads of
one file would merge before 029 decides that they should).

### R2. One request for what the page needs: `GET /videos/runs`

**Decision**: A new endpoint returns, for a path: whether the video is on the server, its name
and duration, every job that read it (newest first) with status, times, batch, dataset, and per
pipeline its status, error and result files (with download URLs and provenance file names), and
the projects (saved datasets) that hold it, either through a job's `dataset_id` or because the
dataset's folder or manifest contains it. Contract: [contracts/api.md](contracts/api.md).

**Rationale**: The page needs all of it before drawing a lane label; one request instead of one
per job. It's also the server side of US7 ("which projects").

**Alternatives**: the viewer lists all jobs and filters (pages through every job on every open);
a `video_path` filter on `GET /jobs` (still one results request per job).

**Cost**: a scan of the jobs, as `has_video_path` does: fine for a lab's thousands of jobs; 029's
video list will index it.

### R3. Stream with 027's token; refresh it without a jump

**Decision**: The page asks `/videos/stream-token` for the video's path and sets the `<video>`
`src` to the stream URL. On a 401 (expired after an hour, or the server restarted with a new
secret), it fetches a new token, swaps the URL and restores the time and play state. A video
the browser can't decode (H.265 in Firefox) gets the existing codec message; a video no longer
on the server shows the existing missing-video message with the lanes (US4.3).

**Rationale**: Spec 027 decided this for the same reason (a `<video>` element can't send the
header); reusing it keeps one mechanism.

**Alternatives**: a cookie (CSRF surface, and cookies are shared across tabs signed in with
different keys); `GET /jobs/{id}/video` with a token (ties playback to one job, when a video has
several).

### R4. Fetch per file, parse with today's parsers

**Decision**: For each lane's chosen run, fetch only that pipeline's result files through
`/jobs/{id}/results/files/{pipeline}?name=` and parse them with the existing parsers
(`mergeAnnotationData`), which already accept `File`s. Files are cached in TanStack Query for the
session; nothing is written to the local library or a folder. The zip download, its folder
question and `useZipDownloader` leave the Video page (US4, FR-013, FR-014); the local library
stays for the Results page until spec 029 removes it.

**Rationale**: The parsers and overlays work and are contract-tested; the change is where the
bytes come from.

**Alternatives**: a server endpoint returning parsed JSON (a second parser to keep in step, and
standalone mode would still need the viewer's).

### R5. Table rows in the viewer, twinned with the server, and lanes built from them

**Decision**: A TypeScript twin of the spec 025 row builder (`lib/table/rows.ts`) turns parsed
results into the same rows as `exporters/readers.py`. A fixture of expected rows for the
contract fixtures (`tests/fixtures/viewer_contract/expected_table.csv`) is written by a Python
test from the real exporter and checked by a Vitest test against the twin, as 027's sampling
twin does (`sampling_cases.json`). The Data tab shows these rows, and **lanes are built from the
same rows** (R6), so the table and the timeline can't disagree (spec assumption). Overlays keep
drawing from the parsed data (keypoints and landmarks aren't in the table).

**Rationale**: Standalone mode has no server, so the rows must be made in the browser; the
shared fixture is the guard that they match the export.

**Alternatives**: rows from a server endpoint (no standalone mode; a second request per run);
lanes from the parsed structures directly, as today (the table and timeline could then differ).

### R6. What a lane is, and how it's drawn

**Decision**: A lane is one *kind* of result from one pipeline in one run (by measure group: speech, speakers, people, faces,
emotion, scenes, Ask AI, action units, head pose, gaze, …) from one run. Rows within a lane come
from `track` (person, speaker, face track) or, for numeric measures, from `label` (each emotion,
AU). Each measure has one form: **span** (`start` and `end`: speech, speaker turns, scenes),
**moment** (`start` only: VLM labels, single detections), or **numeric** (`value`: emotion
scores, AU intensity, head pose, face count per sampled frame, drawn as a small chart with its
scale labelled). Per-frame detections (people, faces) are drawn as presence (a bar where the
track was seen, gaps where it wasn't, never bridged beyond the sampling interval, Principle VI).
The mapping is a table in [contracts/viewer.md](contracts/viewer.md); a measure it doesn't know
gets a generic moment lane labelled with its pipeline and measure, never dropped (FR-001).

**Drawing**: lane labels are DOM (accessible, clickable); each lane's body is one `<canvas>`
redrawn on view or data change. Events per lane are kept sorted by start, so the visible slice
is two binary searches; when a lane has more events than pixel columns in view, it draws
**density** (count per column) and says "density" in its label (edge case). Hover and click use
the same search. Target: 50,000 events, 60 fps panning (SC-004).

**Rationale**: DOM markers (027's lanes) don't scale to 50,000; canvas does, and the existing
Timeline already uses one.

**Alternatives**: recharts per lane (SVG, slow at this count); a timeline library (none in the
bundle; the budget is 300 KB).

### R7. Zoom, pan, and the visible-range bar

**Decision**: The timeline holds a view `[start, end]`; Ctrl/⌘+wheel and `+`/`−` zoom around the
pointer or playhead, drag or wheel pans, `0` fits the whole video; the visible-range bar above
the lanes shows and drags the view. While playing, the view pages forward when the playhead
leaves it (US1.3). 027's `lanePercent`/`timeAtClick` gain a view range; `AnswerLane` and
`SampleTicks` keep working with the whole video as the view.

### R8. Keys

**Decision**: Keep every key that exists (Space, ←/→, Shift+←/→, ↑/↓, Ctrl+Shift+D), now active
anywhere on the page rather than only when the controls card has focus. Add: `[` / `]` previous /
next event in the selected lane, `+` / `−` / `0` zoom, `N` note at this moment, `Shift+N` start
or end a span note, `Esc` cancel. No key fires while typing in an input or textarea (edge case).
A `?` button lists them.

### R9. Notes on the server, keyed to the video

**Decision**: A new table `video_notes` (data model) keyed by the video's resolved path, with
`video_sha256` when known, text, start, optional end, author and times. API under
`/api/v1/videos/notes` ([contracts/api.md](contracts/api.md)). Every signed-in user sees and may
edit every note, as with Ask AI history; each records who wrote and who last edited it. In
standalone mode (no server) the Notes lane says notes need a server and the N key does nothing.

**Rationale**: FR-018 and the spec's assumption: notes live with the video, not a run, so they
survive re-runs. The no-admin-friction rule: anyone signed in may use everything.

**Alternatives**: a notes file in each run folder (a note would belong to one run; spec 023's
original idea, superseded); browser storage (lost with the browser, not shared).

### R10. Notes in the table and ELAN exports

**Decision**: Table format version **2**: adds the measure `note` (pipeline `notes`, `label` =
the text, `start`/`end`) and one column at the end, `author` (empty on every pipeline row).
ELAN: a tier `Notes (researcher)` with `ANNOTATOR` set to the author when every note has one
author, else one tier per author. The exporters find a video's notes by its `media_path` (run
record) or the job's `video_path`. Exports run without the database (CLI on a copied folder)
say "notes not included: no database" in the export summary. The Data tab shows note rows too,
styled as notes.

**Rationale**: The spec 025 contract allows later versions to add columns at the end and
measures; analysis code that selects columns by name is unaffected.

**Alternatives**: author in `track` (track means who the pipeline followed); notes as a separate
file in the export zip (easy to lose, and the table is "one table").

### R11. Which run, and comparing two

**Decision**: Each lane defaults to the newest job on this video whose pipeline result is
completed (FR-016); its label's menu lists the others (date, status, the setting that differs
most visibly) and "Compare with…", which adds a second lane for the other run directly below,
labelled with its run. About lists the settings that differ between the two, from their
provenance files, through a `lib/settingsDiff.ts` that 029's "What changed?" (FR-013) reuses. A
`run=` URL parameter starts every lane on that job where it has results.

### R12. Where the video was opened from

**Decision**: `from=project:<dataset id>`, `from=run:<batch id>` or `from=folder:<server path>`
names the list; the page fetches its order (dataset videos, batch jobs, folder listing through
the existing endpoints), shows "3 of 48" and ‹ › (US7.1), and keeps lane settings when moving
(they're per user, R13). Without `from`, the header lists the projects from R2 as links (US7.2).
`BatchNavigation` becomes `ListNavigation`.

### R13. Lane preferences per user, in the browser

**Decision**: Which lane kinds are collapsed, which overlays are on, timeline height and the Data
tab's lane filter are saved in `idbKv` under the server URL and user id, by lane *kind* (not
run), so they carry across videos and reloads (FR-009, US2.4). Unknown kinds default to shown.

**Rationale**: A display preference, not research data; no server state for it.

**Alternatives**: a server-side preference store (none exists; not worth one for this).

### R14. The Ask AI lane and overlay

**Decision**: VLM results are a lane like any other (FR-022): moments, labelled with the question
name and model. The overlay shows the latest sample at or before the playhead, held no longer
than the sampling interval and marked "sampled at 0:12.5" (`sampledAtTime.ts`, Principle VI).
`VlmAnnotationPanel` goes; the answer and reasoning are in the hover and the Data tab.

### R15. What leaves, what moves

- `UnifiedControls`, `OverlayControls`, `TimelineControls`: go (FR-008).
- `FileViewer` (JSON modal): goes; About lists each lane's raw files with a download (FR-011).
- `DebugPanel`, "Load different files": into ⋯ (FR-010).
- `OpenFace3Controls`: its toggles (landmarks, gaze, head pose, AUs) become the Faces lane's
  overlay menu.
- `useZipDownloader`: stays for the local library until 029; not used by the Video page.

### R16. Failed pipelines and empty states

**Decision**: R2 gives each pipeline's status and error; a failed pipeline's lane shows the
reason (edge case). A video with no runs shows the video, the empty timeline and "Run…" (029's
New run; until then, the existing wizard with this video). A result file without provenance
shows "not recorded" in About.

### R17. Area of the video (SC-003)

**Decision**: At 1440×900 a 16:9 video is limited by height, not width: today's column (window
minus the 384 px card) stacks a header, a controls row, a 160 px timeline and a footer under it.
Removing the card therefore gains no area, and a taller timeline would lose some. SC-003 was
reworded (analysis, 2026-10-19) to "no smaller than today's with the Data tab open". The budget:
header and controls in one 48 px row, no footer on this page, a 200 px default timeline whose
lanes scroll inside it, a 360 px side panel (tasks T026). jsdom has no layout,
so the check is manual: measure both pages at 1440×900 in the browser and record the areas in
`checks.md` ([quickstart](quickstart.md), step 3).
