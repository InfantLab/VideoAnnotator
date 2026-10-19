# Feature Specification: Video Page

**Feature Branch**: `028-video-page` (work lands on `1.7-dev`)
**Created**: 2026-10-16
**Status**: Draft
**Design**: [spec 023](../023-viewer-overhaul/spec.md) (The Video page; themes 3, 5, 6, 10, 13),
journey 2 in [journeys.md](../023-viewer-overhaul/journeys.md),
[inventory](../023-viewer-overhaul/inventory.md) (second pass: the results player)
**Input**: User description: "The results player as one Video page: everything found in a video,
from any run, on one time axis, with the values behind it. Annotation controls take too much
space: toggle from the timeline. The timeline is minimal: labelled lanes, one row per person or
speaker, zoom, numeric tracks. The data as a proper table beside the video, not a JSON modal.
Aware of the project or folder it was opened from. Researchers' own notes, light touch."

## Context

The results player today (spec 023 inventory) puts a 384-pixel card of fixed switches beside the
video, six rows for six pipelines whether they ran or not; draws the timeline at a fixed height
over the whole video with no lane labels; shows the data as pretty-printed JSON in a modal; and
before anything plays, downloads the whole video inside a zip and asks the researcher to choose a
folder on their computer. A researcher used to ELAN's labelled tiers (journey 2) has to learn
which unlabelled row is which.

This spec makes the player the **Video page**: one video, every result for it, on labelled lanes
that are also its controls, with the table of values beside it, streamed from the server rather
than downloaded first.

### What the researcher sees

```
┌ ← Peekaboo study   p01_t1.mp4   3 of 48  ‹ ›          [Ask AI] [Run…] [Export ▾] [⋯] ┐
│                                                  │ Data │ Notes │ Runs │ About        │
│                                                  │ ─────────────────────────────────│
│              video (with overlays)               │ Lane: [all ▾]   Search [      ]  │
│                                                  │ 0:11.9  Speakers  SPEAKER_00     │
│                                                  │▶0:12.4  Speech    "Peekaboo!"    │
│                                                  │ 0:12.5  Ask AI    touching       │
│ ▶  0:12.4 / 3:20  ⏮ ⏭ event   1×   frame 372     │ 0:13.0  People    person 2 in    │
├──────────────────────────────────────────────────┴──────────────────────────────────┤
│ [Layers ▾]  [− zoom +]  ▕██████▏────────────────────────── (visible range)          │
│ 👁 Speech                     ▬▬▬▬  ▬▬▬▬▬     ▬▬                 run 12 Oct ▾       │
│ 👁 Speakers   SPEAKER_00      ▬▬      ▬▬▬                                           │
│               SPEAKER_01          ▬▬▬     ▬                                         │
│ 👁 People     person 1     ▔▔▔▔▔▔▔▔▔▔▔▔▔                                            │
│               person 2           ▔▔▔▔▔▔▔                                            │
│ 👁 Faces      count        ╱╲__╱╲_                                                  │
│ 👁 Ask AI     touch        ●  ●●   ●     ●                                          │
│    Notes      (yours)      ▲      ▬▬▬                     N: note · ⇧N: span        │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Read the timeline like ELAN (Priority: P1)

A linguist opens a video with speech and speaker results. Each lane is labelled at the left:
"Speech", then "Speakers" with a row for each speaker, each in its own colour. They zoom into ten
seconds to see utterance boundaries, hover a segment to read its text and times, and step to the
next utterance with a key.

**Why this priority**: Journey 2's researchers expect a shared, labelled timeline per video; it is
the core of the page.

**Independent Test**: Open a demo video with every pipeline's results; a new user names what every
lane and row shows without help.

**Acceptance Scenarios**:

1. **Given** results, **When** the page opens, **Then** there is one labelled lane per kind of
   result present, and none for results absent.
2. **Given** results with people or speakers, **When** shown, **Then** each person or speaker has
   its own labelled row with a stable, distinct colour, the same on the overlay.
3. **Given** the timeline, **When** zoomed and panned, **Then** a visible-range bar shows where in
   the video the view is, and the playhead stays in view while playing.
4. **Given** a selected lane, **When** next/previous event is used (button or key), **Then** the
   video seeks to that lane's next/previous event.
5. **Given** an event, **When** hovered, **Then** its values (label, text, times, confidence)
   show; **When** clicked, **Then** the video seeks to it.
6. **Given** numeric results (face count, scores), **When** shown, **Then** they are a small chart
   in their lane with their scale labelled.

---

### User Story 2 - Toggle from the lanes, not a card (Priority: P1)

The researcher hides the scenes lane by clicking its label, and turns off the face boxes on the
video with the eye on the Faces lane. "Layers" lists every lane for showing or hiding many at once.
The video gets the space the card used to take.

**Why this priority**: Spec 023 theme 10a: the controls card takes too much room for what it does.

**Independent Test**: Every overlay and lane can be shown and hidden from the lanes or Layers; the
page has no separate controls card.

**Acceptance Scenarios**:

1. **Given** a lane, **When** its label is clicked, **Then** it collapses to its label row; again,
   it expands.
2. **Given** a lane with an overlay, **When** its eye is toggled, **Then** the overlay on the video
   shows or hides; lanes without an overlay have no eye.
3. **Given** Layers, **When** opened, **Then** every lane is listed with both toggles, and "all"
   and "none" for each.
4. **Given** a reload or another video, **When** the page opens, **Then** the same lanes are shown
   or hidden as last time.

---

### User Story 3 - The values beside the video (Priority: P1)

Beside the video, the Data tab is the video's results table (spec 025): time, lane, row, value.
As the video plays, the current row is highlighted and kept in view; clicking a row seeks;
filtering by lane or searching narrows it. "View data" and its JSON modal are gone.

**Why this priority**: Spec 023 theme 13; journey 2's "videos turned into data in a standard
table", visible where they look at the video.

**Independent Test**: Play a demo video: the highlighted row tracks the playhead; click a row in
the second half: the video seeks there; filter to Speakers: only speaker rows remain.

**Acceptance Scenarios**:

1. **Given** the Data tab, **When** shown, **Then** it lists the same rows as the spec 025 table
   for this video and these runs.
2. **Given** playback, **When** the playhead moves, **Then** the row at or before it is
   highlighted and scrolled into view, unless the user has scrolled the table themselves.
3. **Given** a row, **When** clicked, **Then** the video seeks to its start.
4. **Given** the Data tab, **When** filtered by lane or searched, **Then** only matching rows show,
   and the count says how many of how many.

---

### User Story 4 - Plays at once, no folder question (Priority: P1)

The researcher opens a video and it starts playing within seconds, even when it's a large
high-resolution file. Nothing asks them to choose a folder on their computer first.

**Why this priority**: Spec 023 inventory: looking at a result starts with a storage decision and
a full download.

**Independent Test**: Open a 2 GB video's page on a server: playback starts without downloading
the whole file, and no folder dialog appears.

**Acceptance Scenarios**:

1. **Given** a video on the server, **When** its page opens, **Then** the video streams and can be
   played and sought before it has fully loaded.
2. **Given** results, **When** the page opens, **Then** only the result files needed are fetched.
3. **Given** a video no longer on the server, **When** the page opens, **Then** the results still
   show, with the existing missing-video message where the video would be.
4. **Given** standalone mode (no server), **When** a video and result files are opened from the
   computer, **Then** the page works as it does today.

---

### User Story 5 - Which run, and comparing runs (Priority: P2)

A video has been run twice, with different face settings. Each lane says which run it shows
(newest by default). From a lane's menu the researcher picks the earlier run, or shows both as two
lanes to compare.

**Why this priority**: Spec 023 settings layer 1: runs are the version history; comparing is how
a researcher sees what a change did.

**Independent Test**: Run faces twice on a demo clip with different thresholds; show both lanes;
they differ where expected.

**Acceptance Scenarios**:

1. **Given** a video with several runs of a pipeline, **When** the page opens, **Then** each lane
   shows the newest successful run and names it.
2. **Given** a lane's menu, **When** another run is chosen, **Then** the lane, overlay and table
   switch to it.
3. **Given** "compare", **When** chosen, **Then** the two runs show as adjacent lanes, and the
   About tab lists the settings that differ.

---

### User Story 6 - Notes, light touch (Priority: P2)

Watching, the researcher presses N to mark a moment ("infant looks away") or Shift+N twice to mark
a span, and types a short note. Notes are their own lane, styled as the researcher's, listed in
the Notes tab, and included (marked as notes) in the table and the ELAN export.

**Why this priority**: Spec 023 theme 6; researchers' own notes travel with the results to the
tools they use. Not human review of pipeline output, which is shelved.

**Independent Test**: Add two notes, reload: they're there; export ELAN: they're a tier of their
own.

**Acceptance Scenarios**:

1. **Given** playback, **When** N is pressed, **Then** a note is created at the current time and
   focused for typing; Esc cancels an empty note.
2. **Given** Shift+N pressed twice, **When** at two times, **Then** a span note is created between
   them.
3. **Given** notes, **When** shown, **Then** they are visibly distinct from pipeline output
   (their own lane, style and label), and never mixed into a pipeline's lane (Principle VI).
4. **Given** notes, **When** the table or ELAN file is exported, **Then** they are included,
   marked as researcher notes with their author.

---

### User Story 7 - Where this video belongs (Priority: P2)

Opened from a project or folder, the header names it and shows "3 of 48" with previous/next.
Opened directly, it shows which projects the video is in.

**Why this priority**: Spec 023 theme 10c; moving through a project's videos is the normal way to
check results.

**Independent Test**: Open a video from a project; ‹ › move through that project in its order;
open it from a folder; ‹ › move through the folder.

**Acceptance Scenarios**:

1. **Given** the page was opened from a list, **When** ‹ or › is used, **Then** the previous or
   next video in that list opens with the same lane settings.
2. **Given** the page was opened directly, **When** shown, **Then** the header lists the video's
   projects as links.

---

### Edge Cases

- A pipeline that failed shows its lane with the failure reason, not "no data".
- A lane with thousands of events (person tracking at full rate) stays responsive when zoomed out
  (events may be drawn as density, labelled as such).
- Results from an older run without provenance show "not recorded" in About.
- The person-box-area lane (spec 023 fix) keeps its honest label until a movement measure exists.
- A video with no results yet shows the video, an empty timeline, and "Run…".
- Two notes at the same moment are both shown.
- Keyboard shortcuts don't fire while typing in a note or search box.

## Requirements *(mandatory)*

### Functional Requirements

**Lanes and timeline**
- **FR-001**: The page MUST show one lane per kind of result present, derived from the results,
  not from a fixed list; absent results take no space.
- **FR-002**: Each lane MUST be labelled with what it shows (spec 023 Decision 1) and the run it
  comes from.
- **FR-003**: People, speakers and faces MUST be shown one labelled row per person, speaker or
  face track, with colours that are distinct within a lane and the same on the overlay.
- **FR-004**: Spans, moments and numeric values MUST each have their own form (bars, markers,
  small charts with a labelled scale).
- **FR-005**: The timeline MUST support zoom, pan, a visible-range bar, hover details, click to
  seek, and next/previous event in the selected lane by button and key.
- **FR-006**: The readout MUST show time and frame number.
- **FR-007**: The timeline's height MUST be adjustable, and lanes MUST be collapsible.

**Controls**
- **FR-008**: Overlays and lanes MUST be toggled from the lanes (label: collapse; eye: overlay)
  and from a Layers menu; the separate controls card MUST go.
- **FR-009**: Lane and overlay visibility MUST be remembered per user across videos.
- **FR-010**: Rarely used actions (debug panel, load different files) MUST move to a ⋯ menu.

**Data**
- **FR-011**: A Data tab beside the video MUST list the video's results as the spec 025 table
  rows, following playback, seeking on click, filterable by lane and searchable; the JSON modal
  MUST go (raw files stay reachable from About).
- **FR-012**: An About tab MUST show each lane's provenance (spec 017): pipeline, model, version,
  settings, run.

**Loading**
- **FR-013**: The page MUST stream the video from the server so playback starts before the file
  has loaded, and fetch only the result files it shows.
- **FR-014**: The page MUST NOT ask the researcher to choose a local folder before showing a
  result.
- **FR-015**: Standalone mode (files from the computer, no server) MUST keep working.

**Runs**
- **FR-016**: Each lane MUST default to the newest successful run of its pipeline on this video and
  offer the others; two runs MUST be showable as adjacent lanes, with differing settings listed.

**Notes**
- **FR-017**: Researchers MUST be able to add moment and span notes by key or button, edit and
  delete them, and see them as their own lane and in a Notes tab.
- **FR-018**: Notes MUST be stored on the server with the video, with author and time, and be
  included, marked as notes, in the table and ELAN exports.
- **FR-019**: Notes MUST be visibly distinct from pipeline output everywhere they appear.

**Context**
- **FR-020**: Opened from a project, folder or other list, the page MUST show it and offer
  previous/next within it; otherwise it MUST list the video's projects.
- **FR-021**: The header MUST offer Ask AI (spec 027) on this video, Run…, and Export (table, ELAN,
  methods: spec 025).
- **FR-022**: Ask AI answers (spec 027) MUST be a lane like any other, with an overlay that shows
  the answer for the current moment on the video (the latest sample at or before the playhead,
  marked as sampled, as the VLM panel does today).

### Key Entities

- **Lane**: one kind of result from one pipeline in one run, with rows; visible or collapsed;
  overlay on or off.
- **Row**: one person, speaker, face track or label within a lane.
- **Note**: a moment or span on a video, text, author, created and edited times.
- **Lane preferences**: per user, which lanes and overlays are shown.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A video's page starts playback within 3 seconds on a local network, whatever the
  file size.
- **SC-002**: Every lane and row label has a tooltip saying what it shows and which pipeline made
  it; in the pilot after release, researchers name every lane on the demo video without help
  (spec 023: checked through pilot feedback).
- **SC-003**: At 1440×900, with the Data tab open and the timeline at its default height, the
  video area is no smaller than today's, while the timeline shows every lane of the demo video
  labelled.
- **SC-004**: With an hour-long video and 50,000 events, zooming, panning and playback stay smooth
  (no visible stutter).
- **SC-005**: Every value in the table can be reached from the timeline in one click and vice
  versa.
- **SC-006**: Notes survive reload and appear in exports, every time.

## Assumptions

- The server's existing video streaming (spec 021) and per-file result downloads are the basis;
  a `<video>` element needs auth it can send (short-lived token or cookie), decided in the plan.
- The browser's local copies of results (today's "Results" page) become an invisible cache or go;
  spec 029 removes the page.
- Lanes are built from the same parsing as the spec 025 table, so the table and the timeline
  can't disagree.
- Notes live on the server, keyed to the video (not to a run), so they survive re-runs; this
  supersedes spec 023's "in the run folders unless the spec decides otherwise".
- A real movement measure is not in this spec: simple per-person movement metrics, computed on
  the server, come in their own v1.7.0 spec after 029 (spec 023 theme 2, clarified 2026-10-19)
  and will appear here as numeric lanes.

## Out of Scope

- Correcting or reviewing pipeline output (shelved).
- Coding schemes, controlled vocabularies, or ELAN-style tier types for notes.
- Several videos side by side.
- The Ask AI window (spec 027) and the Videos page (spec 029).
