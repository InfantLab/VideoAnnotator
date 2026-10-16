# Feature Specification: Videos and Projects

**Feature Branch**: `029-videos-and-projects` (work lands on `1.6-dev`)
**Created**: 2026-10-16
**Status**: Draft
**Design**: [spec 023](../023-viewer-overhaul/spec.md) (Decisions 1–8; Draft proposal: settings,
pages), both journeys in [journeys.md](../023-viewer-overhaul/journeys.md),
[inventory](../023-viewer-overhaul/inventory.md)
**Input**: User description: "Restructure the viewer around videos, not the API's objects. Videos
and projects are aspects of one page. A project is the optional, only grouping: a set of videos
(today's saved datasets); a video can be in several. Projects remember their current settings;
workflows can be saved for reuse on other projects, never required. Call things what they are.
Exports and methods in plain sight. Help where the question arises, not on an external wiki."

## Context

To code some videos and look at the results, a researcher today meets jobs, batches called runs,
datasets, results (the browser's own copies), presets and prompts, on separate pages named after
the API's objects (spec 023, The problem; inventory). The same videos appear on three pages under
different words. The nav says Jobs; the page says "Annotation Runs"; Home says "Recent runs" then
"No jobs yet". Help is a DeepWiki of the archived viewer repo.

Spec 023 settled the model: **videos**, **folders**, **runs**, and an optional **project** as the
only grouping, which is what a saved dataset already is. This spec builds the page that holds them,
the settings that projects remember, workflows for reuse, the navigation around them, and help in
context. The API keeps its names and endpoints (Principle V); the viewer stops showing them.

### What the researcher sees

```
┌ Videos ───────────────────────────────────────────────────────────────┐
│ All videos      (212) │ Peekaboo study · 48 videos     [Run…] [Export▾]│
│ Not in a project (37) │ Settings: faces, speech, Ask AI "touch" [Edit] │
│ ───────────────────── │ Runs: ● running 31/48 · ✓ 12 Oct · ✓ 3 Oct     │
│ Projects              │ ───────────────────────────────────────────────│
│  Peekaboo study  (48) │ ☐ Video         Folder      Faces Speech Ask AI│
│  Pilot lab B     (90) │ ☐ p01_t1.mp4    /data/pk    ✓     ✓      ●     │
│  + New project        │ ☐ p01_t2.mp4    /data/pk    ✓     ✓      –     │
│ ───────────────────── │ ...                                            │
│ Folders               │                                                │
│  /data/pk        (48) │ 2 selected: [Run…] [Export▾] [Add to project▾] │
└───────────────────────┴────────────────────────────────────────────────┘
```

Nav: **Home · Videos · Ask AI · Pipelines**, an activity indicator for what's running, Help,
Settings. Once set up, the viewer opens on Videos.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run pipelines on a folder of videos and get a table (Priority: P1)

A researcher with 48 videos in a folder opens Videos, picks the folder on the left, presses Run…,
keeps the default settings for faces and speech, and starts. Progress shows in the header and the
activity indicator. When it's done, the header says "48 videos → results table" with a download,
and each row shows which results the video has.

**Why this priority**: Journey 2's core: several videos at once, turned into a standard table.

**Independent Test**: From a fresh install, run two pipelines on a folder of three videos and
download one table, without meeting the words job, batch or dataset.

**Acceptance Scenarios**:

1. **Given** a server folder, **When** chosen on the left, **Then** its videos are listed with the
   results each has.
2. **Given** a folder, selection or project, **When** Run… is pressed, **Then** run setup opens on
   exactly those videos, with settings prefilled (FR-012).
3. **Given** a run in progress, **When** watching, **Then** progress shows in the header of the
   folder or project it ran on and in the activity indicator, and rows fill in as videos finish.
4. **Given** a finished run, **When** it completes, **Then** the results table is offered as the
   run's headline result, with ELAN and methods beside it (spec 025).

---

### User Story 2 - Projects after the fact (Priority: P1)

The researcher selects 12 of the videos and chooses "Add to project → New project…", names it, and
it appears on the left. Later they add videos from another folder. One video is also in a second
project. Nothing ever required a project to run or view anything.

**Why this priority**: Spec 023 Decisions 2, 5, 6: projects optional, made after the fact, videos
in several.

**Independent Test**: Make a project from a selection, add a video to it from another folder,
put that video in a second project; each project lists its videos correctly.

**Acceptance Scenarios**:

1. **Given** a selection, **When** "Add to project" is used, **Then** the videos join an existing
   or new project.
2. **Given** a video in two projects, **When** either project is shown, **Then** the video is
   listed, with project chips on its row.
3. **Given** an existing saved dataset, **When** the viewer is upgraded, **Then** it appears as a
   project with the same name, videos and runs.
4. **Given** a project made from a folder, **When** new videos appear in that folder, **Then** the
   project offers to add them (replacing the drift dialog).
5. **Given** a project, **When** removed, **Then** its videos and results are untouched, and the
   confirmation says so.

---

### User Story 3 - Projects remember their settings (Priority: P1)

The researcher changes the face threshold for a project's second run. The next time they run
anything on that project, setup starts from those settings. Nothing asked them to save. A video
outside any project starts from the settings last used, or the defaults.

**Why this priority**: Spec 023 settings layer 2: light touch, nothing to name.

**Independent Test**: Run on a project with a changed setting; open Run… again on it: the change
is there; open Run… on a video in no project: last-used settings.

**Acceptance Scenarios**:

1. **Given** a run on a project, **When** it starts, **Then** its settings become the project's
   current settings.
2. **Given** a project's settings, **When** "Edit" is used in its header, **Then** they can be
   changed without starting a run.
3. **Given** run setup, **When** prefilled, **Then** it says where the settings came from ("This
   project's settings, last changed 12 Oct").
4. **Given** two runs, **When** "What changed?" is chosen, **Then** the settings that differ are
   listed, from the runs' provenance.

---

### User Story 4 - Workflows, only when wanted (Priority: P2)

A lab that has found good settings saves them as a workflow ("Infant faces + speech") from a run or
a project, and applies it to a new project; the workflow's settings are copied into the project.
They share it with another lab as a file.

**Why this priority**: Spec 023 settings layer 3; reuse across projects and labs, never forced.

**Independent Test**: Save a workflow from a project, apply it to another project, edit the
workflow: the second project's settings are unchanged.

**Acceptance Scenarios**:

1. **Given** a run or project, **When** "Save as workflow" is used, **Then** its pipelines,
   settings, Ask AI questions and sampling, and export choices are saved under a name.
2. **Given** a workflow, **When** applied to a project or selection, **Then** its settings are
   copied in; later edits to the workflow do not change the project.
3. **Given** a workflow, **When** exported and imported elsewhere, **Then** it is the same.
4. **Given** existing presets, **When** the viewer is upgraded, **Then** they appear as workflows.

---

### User Story 5 - One video, one place (Priority: P1)

Clicking a video opens its Video page (spec 028), with previous/next through the list it came
from. Everything about a video, every run on it, its results and notes, is reachable from there or
from its row.

**Why this priority**: Replaces Jobs, Job detail and Results; ends "the same videos on three pages".

**Independent Test**: Every link to a job, batch, dataset or results page in the old viewer leads
to the matching video, project, folder or run filter.

**Acceptance Scenarios**:

1. **Given** a row, **When** clicked, **Then** the Video page opens in the context of that list.
2. **Given** an old address (`/jobs/...`, `/batches/...`, `/datasets`, `/results`), **When**
   opened, **Then** it leads to the matching place.
3. **Given** a run on videos outside any project, **When** shown, **Then** it is a filter of
   Videos ("videos in this run") with its progress and log.

---

### User Story 6 - Help where the question arises (Priority: P2)

Terms and settings have tooltips in plain words. After an action, a toast says what happened and
what to do next ("Run started on 48 videos. You can close this tab; results will be here."). Each
page has a line of help and a link into VideoAnnotator's own documentation at the right section.
Without a server, pages say what needs one and what works anyway.

**Why this priority**: Spec 023 theme 8; the DeepWiki links point at an archived repo.

**Independent Test**: No link to DeepWiki remains; every page has inline help with a working docs
link; every action that starts or finishes something gives a toast with a next step.

**Acceptance Scenarios**:

1. **Given** any page, **When** shown, **Then** it has a one-line explanation and a link into this
   project's docs.
2. **Given** a term from spec 023's vocabulary (run, project, workflow) or a setting, **When**
   hovered or focused, **Then** a tooltip explains it.
3. **Given** a run started, finished or failed, **When** it happens, **Then** a toast says so and
   offers the next step.
4. **Given** no server, **When** a server page is opened, **Then** it says it needs a server, how
   to connect, and what works without one (opening files from this computer, demos).

---

### User Story 7 - First run (Priority: P2)

A new user's Home shows three start points: "Ask AI about a video", "Run pipelines on videos",
"Open files from this computer", and the setup checklist. Once they have results, the viewer opens
on Videos.

**Why this priority**: Spec 023: Home for first run and no-server use only.

**Independent Test**: Fresh install: Home shows the start points; after one run: opening the
viewer lands on Videos.

**Acceptance Scenarios**:

1. **Given** no results yet, **When** the viewer opens, **Then** Home shows the start points and
   checklist.
2. **Given** results exist, **When** the viewer opens at its root, **Then** it shows Videos; Home
   stays reachable.

---

### Edge Cases

- A video known only from a job whose file has since moved (spec 022) is listed, marked
  unavailable, with its results.
- The same file reached by two paths (a symlink, a re-mounted share) is one video when the server
  can tell, otherwise two, never silently merged.
- Two projects with the same name from different owners: names are unique per owner as today;
  the list shows the owner when names clash.
- A run that covered videos in two projects shows under both.
- A workflow that names a pipeline not installed here applies the rest and says what's missing
  (as presets do today).
- Hundreds of videos: the list pages and filters stay fast.

## Requirements *(mandatory)*

### Functional Requirements

**Videos page**
- **FR-001**: The viewer MUST have one Videos page listing videos, with a left side of: All
  videos, Not in a project, each project, each server folder with videos.
- **FR-002**: Each video row MUST show its name, folder, projects, and which results it has
  (done, running, failed, none) per pipeline.
- **FR-003**: Rows MUST be selectable; for any selection, folder or project the page MUST offer
  Run…, Export (table, ELAN, methods) and Add to project.
- **FR-004**: The server MUST provide a list of the videos it knows, built from jobs, projects and
  folders, with each video's results and projects; existing endpoints MUST NOT change.

**Projects**
- **FR-005**: A project MUST be a named set of videos, optionally linked to a folder; the viewer
  MUST present every saved dataset as a project.
- **FR-006**: A video MUST be able to be in any number of projects; removing a project MUST NOT
  touch videos or results.
- **FR-007**: A project linked to a folder MUST offer to add new videos found in it.
- **FR-008**: A project selected MUST show its current settings, its runs with progress, Run…,
  Export, "What changed?" between runs, and Save as workflow.

**Settings**
- **FR-009**: A project MUST remember its current settings: those of its last run, or as edited.
- **FR-010**: The server MUST store a project's current settings with the project (additive to
  saved datasets).
- **FR-011**: Settings for videos in no project MUST start from the user's last-used settings, or
  the defaults.
- **FR-012**: Run setup MUST be prefilled from, in order: a workflow the user chose, the project's
  current settings, the last-used settings, the defaults; and say which.
- **FR-013**: "What changed?" MUST list settings that differ between two runs, from provenance.

**Workflows**
- **FR-014**: Users MUST be able to save a workflow (pipelines, settings, Ask AI questions and
  sampling, export choices) from a run or project, and apply it to a project or selection by
  copying.
- **FR-015**: Workflows MUST be exportable and importable as files; existing presets MUST appear as
  workflows; the presets API MUST keep working.
- **FR-016**: Workflows MUST be listed on the Pipelines page and offered in run setup.

**Run setup**
- **FR-017**: Run setup MUST open from a video, selection, folder, project or Home on exactly
  those videos, show pipelines with their guide (spec 026), and start with at most one
  confirmation when defaults are kept.

**Navigation and words**
- **FR-018**: The nav MUST be Home, Videos, Ask AI, Pipelines, an activity indicator, Help and
  Settings.
- **FR-019**: The viewer MUST NOT show the words job, batch, dataset or preset; it uses video,
  folder, run, project and workflow (spec 023 Decisions 1, 5, 7, 8).
- **FR-020**: The Jobs, Job detail, Batch detail, Results and Datasets pages MUST go, with their
  addresses leading to the matching place (User Story 5).
- **FR-021**: The browser's local copies of results MUST NOT be presented as a separate place.
- **FR-022**: With results present, the viewer's root MUST open Videos; Home keeps first-run and
  no-server start points.

**Help**
- **FR-023**: Every page MUST have a one-line explanation and a link to the matching section of
  this project's documentation; DeepWiki links MUST go.
- **FR-024**: The vocabulary's terms and every setting MUST have tooltips.
- **FR-025**: Starting, finishing, failing or stopping a run, and creating or changing a project
  or workflow, MUST give a toast saying what happened and the next step.
- **FR-026**: Without a server, server pages MUST say they need one, how to connect, and what
  works without one.
- **FR-027**: Getting started MUST be rewritten for this structure and named Help.

### Key Entities

- **Video**: a file the server knows, by location; its results (from runs), its projects, its
  notes (spec 028).
- **Folder**: a server folder with videos (spec 022).
- **Project**: a named, optional set of videos (a saved dataset), possibly linked to a folder;
  current settings; its runs.
- **Run**: settings applied to some videos once (today's job or batch), with progress and
  provenance.
- **Workflow**: saved, named settings for reuse (today's preset, extended with export choices).

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A first-time user runs pipelines on a folder and downloads one results table in
  under 5 minutes of their own time, without documentation.
- **SC-002**: No page in the viewer shows the words job, batch, dataset or preset (checked
  automatically).
- **SC-003**: Every pre-existing saved dataset and preset appears as a project or workflow after
  upgrade, with nothing lost.
- **SC-004**: The nav has at most four destinations besides Help and Settings.
- **SC-005**: Shown the Videos page, users can say where their set of videos is and what has been
  found in each (target: all test users).
- **SC-006**: Every old viewer address leads somewhere meaningful (no "not found").
- **SC-007**: No link to an external wiki remains; every page's docs link resolves.

## Assumptions

- Projects are shared with everyone on the server, as saved datasets are today (spec 018).
- The API keeps datasets, presets, jobs and batches as they are; new fields (a project's current
  settings, a workflow's export choices) and the video list are additive (Principle V).
- "Last-used settings" are per user.
- Documentation sections for the help links are written as part of this spec, in `docs/`.
- Spec 028's Video page exists, or this spec links to the current player until it does.

## Out of Scope

- Projects within projects, or any level above projects (Decision 5).
- Per-project permissions beyond today's.
- The Video page (spec 028) and Ask AI (spec 027) themselves.
- Corpus overview charts (roadmap v1.6.0 Phase 6).
