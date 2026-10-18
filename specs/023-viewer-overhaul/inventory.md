# Inventory: the viewer today (Process step 1)

**Taken**: 2026-10-16, at `a4edb1a` (after spec 025), from `viewer/src/App.tsx`, the pages and
`components/AppLayout.tsx`. First pass; the "terms" column is quoted from the UI.

## Navigation

The top bar has six entries: **New job**, **Jobs**, **Results**, **Datasets**, **Prompts**,
**Pipelines**, plus Home (logo) and Settings. Workbench, Compare and Getting started are routed but
not in the bar.

## Pages

| Route | Page | What it's for | Terms it uses | Reached from |
|---|---|---|---|---|
| `/` | Home | Start points, setup checklist, recent work | "Run pipelines on videos", "Start a new job", "Open latest results", "Recent runs", "No jobs yet", datasets, prompts | Logo |
| `/jobs/new` | New job | 4-step wizard: videos, pipelines, settings, review; presets bar | "Create New Annotation Jobs", preset, pipeline configuration, "Save as dataset" | Nav, Home, Pipelines, Datasets, Run again |
| `/jobs` | Jobs | Runs (batches) and single jobs on the server | "Annotation Runs", "No individual jobs" | Nav |
| `/jobs/:id` | Job detail | One video's job: log, settings, outputs, run again, export | "Job", "Run it again", "Pipeline Configuration", "Results" | Jobs, Batch detail (double-click) |
| `/batches/:id` | Batch detail | A run's videos and progress, export | "Videos in this run", "This batch no longer has any jobs" | Jobs |
| `/view/:jobId` | Results player | Video with overlays and timeline; prev/next within a batch | overlays, timeline, "View Data" | Job detail, Batch detail, Home, Results |
| `/viewer` | Open files | Standalone player: drop in a video and annotation files | "View files from this computer" | Home |
| `/results` | Results | The browser's own downloaded copies of results, plus demos | "The annotations your jobs produced… kept in a folder on this computer" | Nav, Home |
| `/datasets` | Datasets | Named lists of videos, shared on the server | "Named lists of videos to run jobs on" | Nav, Home |
| `/prompts` | Prompts | Every VLM prompt ever run, named and starred | "Every VLM prompt run on this server, in a job or a preview" | Nav, Home |
| `/workbench` | Prompt workbench | Try prompts × models on moments of a video, side by side | rounds, prompts, models | Home (one sentence), Prompts (button) |
| `/compare` | Compare | Two VLM runs on the same videos: where they disagree | "Compare two VLM runs" | Job detail (`CompareWith`) only |
| `/pipelines` | Pipelines | Which pipelines exist, install extras | pipelines, extras | Nav, Home checklist |
| `/settings` | Settings | Server URL, token, diagnostics | "Server Integration Status" | Bar |
| `/getting-started` | Getting started | Intro, demos, help | "VAV", DeepWiki link | Home, footer |

## One thing, two names

- **Run / batch / jobs.** The API's batch is a "run" on Jobs, Home and Batch detail, but the
  route is `/batches/` and the empty state says "This batch". The nav says Jobs; the Jobs page
  heading says "Annotation Runs".
- **Job / run of one video.** Home lists "Recent runs" and then says "No jobs yet". The wizard
  creates "Annotation Jobs" (plural, one per video) when the user thinks of starting one run.
- **Settings / configuration / preset.** "Pipeline Configuration" on Job detail, the
  "configuration" step in the wizard, "preset" for a saved one, and "settings" in Run again's
  tooltip ("Run again with the same settings").

## One name, two things

- **Results.** The nav's Results page is the browser's local library; the "Results" card on Job
  detail is the server's output files; spec 022's results are the run folder on disk.
- **Settings.** The app's Settings page (connection) vs pipeline settings.
- **Preview.** A VLM preview (a prompt tried in the wizard or workbench, recorded in Prompts) vs
  previewing results.

## Same activity spread over pages

- **Trying VLM prompts**: wizard's test panel (`VlmPromptTestPanel`), Workbench, Prompts, Compare.
- **Saving something to reuse**: presets (wizard), datasets (wizard's "Save as dataset", Datasets
  page), prompts (automatic, Prompts page), Run again (Home, Job detail).
- **Exporting**: Export menu on Job detail and Batch detail only; preset export on the wizard;
  downloads on Results.

## Second pass: the results player (2026-10-16)

From `pages/JobResultsViewer.tsx`, `components/VideoAnnotationViewer.tsx`, `Timeline.tsx`,
`UnifiedControls.tsx`, `VideoControls.tsx`, `VlmAnnotationPanel.tsx`, `OpenFace3Controls.tsx`.

### Getting in

- Opening a job's results downloads its whole artifacts zip. The first time, a full-page card asks
  the user to **choose a folder** on their computer ("Choose folder" / "View without saving")
  before anything plays. This is the browser's local library ("Results"), not spec 022's run
  folder, so looking at a result starts with a storage decision.
- "Back to Batch / Jobs / Results" depends on how the job was reached; the player itself doesn't
  say which run or dataset the video belongs to.

### Layout

| Region | Size | Contents |
|---|---|---|
| Header | one row | Back, file name, prev/next (batch only), **Load Different Files**, **View Data**, **Debug** |
| Video | remaining height | Video with overlays |
| Transport | one row | Play/pause, frame step, rate. Keys: Space, ←/→, ↑/↓ (only shortcuts in the player) |
| Timeline | fixed 160 px | Lanes, below |
| VLM panel | when VLM ran | Label and reasoning at the current time, ELAN ground truth if loaded |
| Footer | one row | App footer, inside the player column (DeepWiki link) |
| Right column | fixed 384 px | `UnifiedControls` card, then `OpenFace3Controls` |

Debug and "Load Different Files" sit next to View Data as equals; both are rarely what a
researcher wants.

### Controls (`UnifiedControls`)

- Six rows fixed in code (pose, faces, emotions, transcript, speakers, scenes), each with an
  overlay switch and a timeline switch, plus all-on/all-off for each column and "lock timeline to
  overlays". Rows for pipelines that didn't run stay, greyed, with "(No data)", so the card is
  the same size whatever ran.
- Failed pipelines say why on their row (good, and worth keeping).
- VLM and OpenFace 3 are outside this card: VLM as its own panel under the timeline, OpenFace 3
  as its own card with many settings. So there are three different patterns for "show this
  pipeline's output".
- `OverlayControls`, `TimelineControls` and `WelcomeScreen` are not imported anywhere (dead code
  from before `UnifiedControls`).

### Timeline (`Timeline.tsx`)

- Lanes, top to bottom: transcript cues, speakers, scenes, VLM moments (dots), ELAN tiers, motion
  canvas. **No lane labels**: the user has to know which row is which. Hover shows a native
  `title` tooltip on each bar.
- Always the whole video: no zoom, pan or scroll. A long video makes events slivers.
- **Speakers all get the same colour.** Hue is `speaker_id.charCodeAt(0) * 137.508`, and every
  diarization ID starts with "S" (`SPEAKER_00`, `SPEAKER_01` in the contract fixtures), so two
  speakers are indistinguishable. VLM labels and ELAN tiers use the same first-letter hash, so
  "happy"/"hungry" or two tiers starting with the same letter collide too. This conflicts with
  Principle VI (display that misleads) and is worth fixing before the overhaul.
- **The faces and emotions timeline switches control nothing**: `Timeline.tsx` has no faces or
  emotions lane, only `showMotion`, `showSubtitles`, `showSpeakers`, `showScenes`. A switch that
  does nothing breaks the same principle.
- **"Motion" is box area**: the largest person box's area per 0.1 s, scaled to the maximum. It
  moves when someone walks toward the camera and not when they wave.
- *Fixed 2026-10-16* (before the overhaul): colours now come from each value's rank among the
  distinct values (`lib/categoryColors.ts`, shared by timeline and speaker overlay); faces and
  emotions show "—" (no timeline lane yet) instead of a dead switch; the box-area lane says what
  it is in the controls and on hover. A real movement measure is still theme 2.
- A waveform switch exists (`showWaveform`) but drawing it is a placeholder.
- Time readout shows current time and percent through; no frame number.

## Second pass: no server (standalone mode)

- A banner (`ConnectionErrorBanner`) shows on every page; the nav still offers every server page.
- Home is the only page that says what to do without a server ("Connect to a server", demos,
  Getting started).
- Jobs, Datasets, Prompts and the rest show their load error ("Couldn't load datasets: …")
  rather than "this needs a server; here's what you can do without one".
- What does work without a server: `/viewer` (drop in files), `/results` (local library and
  demos), the player on anything already in the library, Getting started.

## Empty states and errors (sampled)

Good, says what to do next: Home's setup checklist; Datasets' empty state (how to save one from the
wizard); the player's failed-pipeline reasons; the missing-video message (where it was expected
and why it's gone). Weak: raw API error text on list pages; Batch detail's "may have been
deleted"; the player's "Failed to load results" with only Retry.
