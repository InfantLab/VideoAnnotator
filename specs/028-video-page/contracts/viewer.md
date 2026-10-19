# Contract: the Video page (viewer)

What the tests check. Layout as in [spec.md](../spec.md#what-the-researcher-sees).

## Routes

| Route | Behaviour |
|---|---|
| `/video?path=…[&from=…][&run=…][&t=…]` | The Video page (data model, Page context) |
| `/view/:jobId` | Kept (Principle V): fetches the job, then `replace`s to `/video?path=<job.video_path>&run=<jobId>` (plus `from=run:<batch id>` when the job has one). A missing job: the existing not-found message |
| `/viewer` | Standalone: the same page with files from the computer; no notes, no runs menu, no context |

## Lane kinds

Built from table rows (research R5/R6). Order on the page is this table's order; `notes` is
always last.

| Kind | Label | From (`pipeline` / `measure`) | Form | Rows by | Overlay |
|---|---|---|---|---|---|
| `speech` | Speech | `speech` | span | — | subtitles |
| `speakers` | Speakers | `speaker_turn` | span | `track` | speaker label |
| `people` | People | `person` | presence | `track` (`label` names the row: parent, infant) | boxes and pose |
| `faces` | Faces | `face` | presence, plus a "count" numeric row (faces per sampled frame) | `track` when present | boxes; OpenFace 3 menu: landmarks, gaze, head pose, AUs |
| `emotion` | Emotion | `emotion`, `emotion_score`, `valence`, `arousal` | moment (dominant) + numeric per `label` | `label` | label on the face box |
| `scenes` | Scenes | `scene` | span | — | scene label |
| `ask_ai` | Ask AI · *question name* | `vlm_label` | moment | — | latest answer, held ≤ interval, "sampled at …" |
| `action_units` | Action units | `action_unit` | numeric | `label` | — |
| `head_pose` | Head pose | `head_pose` | numeric (degrees) | `label` | (OpenFace 3 menu) |
| `gaze` | Gaze | `gaze` | numeric | `label` | (OpenFace 3 menu) |
| `person_area` | People · box area (not movement) | from person rows, as today's "motion" | numeric | — | — |
| `other:<p>/<m>` | *pipeline · measure* | anything else | moment, or numeric when `value` is set | `track` | — |
| `notes` | Notes (yours) | notes API / `note` rows | moment or span | — | — |

Rows not drawn as lanes (`emotion_score` details, `age`, `gender`, `vlm_reasoning`,
`action_unit_presence`, `scene_score`) are in the Data tab and the hover.

## Behaviour (one test each, at least)

**Lanes (US1, US2)**
- Only kinds with rows, or with a failed pipeline, get a lane; a failed one shows its reason.
- Each lane's label names the kind and its run ("run 12 Oct"); rows are labelled; a row's colour
  equals its overlay colour (`categoryColors` by key).
- Clicking a label collapses / expands; the eye toggles the overlay; kinds without an overlay
  have no eye. Layers lists every lane with both toggles and all / none.
- Preferences (data model) survive reload and another video.
- A lane with more events than visible pixel columns draws density and says "density".
- A kind produced by two pipelines in one run is two lanes, each naming its pipeline; the eye on
  each toggles only that pipeline's overlay.
- Every lane and row label has a tooltip saying what it shows and which pipeline made it.

**Timeline (US1)**
- Zoom (`+`, `−`, Ctrl/⌘+wheel), pan (drag, wheel), fit (`0`); the range bar matches the view and
  drags it; while playing, the view follows the playhead.
- `[` / `]` and the ⏮ ⏭ buttons seek to the previous / next event of the selected lane.
- Hover shows label, text, times, confidence; click seeks to the event's start.
- Numeric lanes label their scale (min, max, unit).
- The readout shows time and frame number (frame = round(t × fps) from the video's metadata).
- The timeline's height is adjustable by dragging its top edge.

**Data tab (US3)**
- Same rows as the twin builder for these runs (and notes), same order.
- The row at or before the playhead is highlighted and scrolled into view, until the user scrolls;
  "Follow playback" resumes.
- Clicking a row seeks to its start and selects its lane; clicking an event highlights its row
  (SC-005).
- Lane filter and search; "12 of 3,408 rows".
- Renders only visible rows (windowed), so 50,000 rows don't stall.

**Loading (US4)**
- The `<video>` `src` is the stream URL; no zip is fetched, no folder dialog appears.
- Only files for the shown runs' pipelines are fetched.
- On a 401 from the stream, a new token is fetched and the time and play state restored.
- Missing video: lanes and Data tab still shown, with the missing-video message.

**Runs (US5)**
- Default run per lane: newest job with that pipeline completed; `run=` overrides where it has
  results.
- Choosing another run in a lane's menu changes that lane, its overlay and its table rows.
- "Compare with…" adds the other run's lane directly below; About lists differing settings.

**Notes (US6)**
- `N`: a moment note at the current time, focused; Esc on an empty note cancels it.
- `Shift+N` twice: a span between the two times (either order).
- Notes lane is last, styled distinctly (outlined, "yours"), never inside a pipeline lane; in the
  Data tab as `note` rows, styled the same.
- Edit and delete from the Notes tab or the note's popover.
- Keys don't fire while typing in an input, textarea or contenteditable.
- Standalone: the Notes lane says notes need a server.

**Context (US7) and header (FR-021)**
- With `from`, the header names the list and shows "3 of 48" and ‹ ›, which open the neighbours
  with `from` kept.
- Without `from`, the header lists the video's projects as links.
- Ask AI (`/ask?video=<path>`, 027's parameter), Run… (the existing wizard, started with this
  video through its `WizardStart` state), Export (table, ELAN, methods: the job exports of the
  run shown; when lanes show more than one run, the menu lists each) and ⋯ (debug panel, load
  different files).

**Gone**: the controls card, "View data" and its JSON modal (raw files are in About).
