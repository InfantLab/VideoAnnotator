# Feature Specification: Saved Datasets in the Viewer

**Feature Branch**: `018-saved-datasets-viewer` (work lands on `1.6-dev`)
**Created**: 2026-10-04
**Status**: Draft
**Input**: Spec 007's viewer handoff (`specs/007-datasets-and-presets/viewer-handoff.md`): the
server has stored saved datasets since v1.5.0, but the viewer has no way to create, list, use,
share or delete them. The v1.6.0 roadmap (Phase 2) puts this before release candidate 1.

## Context

A researcher's study is a set of videos they process again and again: with a new pipeline, new
settings, a corrected prompt. Today they choose those videos from scratch each time, either by
picking files in the browser or by naming a folder the server can see. Spec 007 gave the server
"saved datasets" (a named list of videos: file names and sizes, not the videos themselves), and
spec 008 lets a job record which dataset it came from. Nothing in the viewer uses them. Presets
(saved pipeline settings) already have load and save in the job wizard; their import/export is
still missing.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Save the videos I just chose, and reuse them next time (Priority: P1)

A researcher picks 40 videos in the job wizard and runs them. Next week they want to run the same
40 with another pipeline. In the wizard they save that selection as a named dataset in the same
step; next time they choose "Use a saved dataset", pick it, and the wizard has the same videos
ready, without re-selecting them one by one.

**Why this priority**: reuse is the point of a dataset, and the wizard is where the researcher
already is. It also feeds the Phase 2 "Run on more videos" and "Run it again" work.

**Independent Test**: pick a folder of videos in the wizard, save it as a dataset, start a fresh
wizard, choose the dataset; the same videos are selected and a job runs on them with the dataset
recorded on the job.

**Acceptance Scenarios**:

1. **Given** videos chosen in the wizard (uploaded files or a server folder), **When** the
   researcher chooses "Save as dataset" and names it, **Then** the dataset is saved with those
   videos, and the run they start is recorded as coming from it.
2. **Given** a saved dataset of uploaded files, **When** the researcher chooses it in the wizard,
   **Then** the viewer asks for access to the folder only if it no longer has it, then selects the
   files that match the dataset by name and size.
3. **Given** a saved dataset of a server folder, **When** the researcher chooses it, **Then** the
   wizard uses that folder directly, as the existing server-folder option does.
4. **Given** that some files no longer match (missing, added, or a different size), **When** the
   dataset is chosen, **Then** the wizard lists the differences before anything runs, and the
   researcher chooses to continue with the matching files, update the dataset, or cancel. Nothing
   proceeds silently.

---

### User Story 2 - See and manage my datasets (Priority: P1)

A researcher opens a Datasets page and sees every saved dataset on this server: name,
description, number of videos, who saved it, when it was created and last used. They can open one
to see its videos, rename it, edit its description, remove videos from it, delete it, or start a
job from it.

**Why this priority**: without a list, saved datasets are invisible and can't be cleaned up.
Datasets are shared on a server, so people must see whose dataset they are looking at.

**Independent Test**: with a few datasets saved (by two users), the page lists them all with
owners; the owner can edit and delete theirs; another user can view and use them but not change
them.

**Acceptance Scenarios**:

1. **Given** saved datasets, **When** the Datasets page opens, **Then** each shows its name,
   video count, owner, created and last-used times, newest-used first.
2. **Given** a dataset the researcher owns, **When** they rename it, edit its description,
   remove videos or delete it, **Then** the change is saved, and deleting asks for confirmation
   and says that jobs already run from it are not affected.
3. **Given** a dataset someone else owns, **When** it is viewed, **Then** it can be used and
   exported but its edit and delete actions are not offered.
4. **Given** a dataset, **When** the researcher chooses "Start a job", **Then** the wizard opens
   with that dataset chosen.
5. **Given** no saved datasets, **When** the page opens, **Then** it explains what a dataset is
   (a remembered list of videos; the videos stay where they are) and how to save one.

---

### User Story 3 - Share a dataset or preset with a colleague (Priority: P2)

A researcher exports a dataset, or a preset, to a file and sends it to a colleague, who imports
it into their own server (or the same one) and uses it.

**Why this priority**: labs share protocols. The server already accepts an exported definition as
an import; only the viewer actions are missing.

**Independent Test**: export a dataset and a preset from one server, import both into another,
and use each to start a job.

**Acceptance Scenarios**:

1. **Given** a dataset or preset, **When** the researcher chooses Export, **Then** a file
   containing its full definition downloads, named after it.
2. **Given** an exported file, **When** another user imports it, **Then** it appears in their
   list as theirs, under a name that doesn't clash with their own existing ones.
3. **Given** a file that isn't a valid export, **When** it is imported, **Then** the viewer says
   what is wrong and nothing is created.
4. **Given** an imported preset that names a pipeline not installed on this server, **When** it is
   applied, **Then** it applies what it can and shows that pipeline as unavailable, as presets do
   today.

---

### Edge Cases

- A browser without folder-access support, or a lapsed permission: the researcher re-picks the
  folder, and files are matched by name and size (User Story 1, scenario 4).
- Two files with the same name in different subfolders: matched by relative path where recorded,
  otherwise reported as ambiguous rather than guessed.
- A server folder that no longer exists or isn't readable: the wizard says so and doesn't start.
- A dataset name that already exists for this owner: saving asks for another name (the server
  enforces unique names per owner).
- A dataset whose videos are all gone: offered for deletion or update, never run.
- Authentication off (`AUTH_REQUIRED=false`): one implicit user owns everything; owner labels are
  hidden.
- Very large datasets (thousands of videos): the list and detail views stay responsive, and
  matching a re-picked folder gives progress.

## Requirements *(mandatory)*

### Functional Requirements

**Datasets page**

- **FR-001**: The viewer MUST have a Datasets page, reachable from the main navigation, listing
  all saved datasets on the connected server with name, description, video count, owner, created
  and last-used times.
- **FR-002**: Owners MUST be able to rename a dataset, edit its description, remove videos from
  it, and delete it (with confirmation). Non-owners MUST NOT be offered these actions.
- **FR-003**: The page MUST let the researcher start a job from a dataset.
- **FR-004**: When there are no datasets, the page MUST explain what a dataset is and how to save
  one from the wizard.

**Wizard**

- **FR-005**: The wizard's "Choose videos" step MUST offer "Use a saved dataset" beside uploading
  files and using a server folder.
- **FR-006**: The wizard MUST offer "Save as dataset" for the videos currently chosen, whichever
  way they were chosen.
- **FR-007**: A dataset MUST be able to remember a server folder as well as a list of uploaded
  files. Backend additions, both optional fields so existing datasets and exports stay valid: a
  dataset's server folder, and each manifest entry's relative path.
- **FR-008**: Choosing an uploaded-files dataset MUST reuse the browser's folder access when it
  still has it, and otherwise ask the researcher to re-pick the folder; files MUST be matched by
  relative path, name and size.
- **FR-009**: Any difference between a dataset and the files found (missing, added, size changed)
  MUST be shown before a job starts, with the choice to continue with the matching files, update
  the dataset, or cancel.
- **FR-010**: Jobs started from a dataset MUST record it (the existing `dataset_id`), and the
  dataset's last-used time MUST update.

**Sharing**

- **FR-011**: Datasets and presets MUST each offer Export (download the definition as a file) and
  Import (upload one), using the server's existing export/import contract.
- **FR-012**: Import MUST reject invalid files with a clear reason and MUST resolve name clashes
  without overwriting existing datasets or presets.

**Constraints**

- **FR-013**: Videos MUST NOT be uploaded or copied to save a dataset; only names, sizes, relative
  paths and (for server folders) the folder are stored.
- **FR-014**: Everything a person can do here MUST also be possible from the CLI (`videoannotator
  dataset list|show|export|import|delete`), per the roadmap's "agents are users too".

### Key Entities

- **Saved dataset** (exists, spec 007): name, description, owner, video manifest (name, size, and
  now relative path), timestamps; gains an optional server folder.
- **Video manifest entry**: one remembered video: relative path, name, size.
- **Dataset match**: the result of comparing a dataset to the files currently available: matched,
  missing, added, changed.
- **Preset** (exists, spec 007): gains viewer export/import only.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Rerunning a saved 40-video dataset with a new pipeline takes under 1 minute from
  opening the wizard to starting the run, versus re-selecting every video today.
- **SC-002**: 100% of differences between a dataset and the files found are shown before a run
  starts (tested with missing, added and resized files).
- **SC-003**: A dataset and a preset exported from one server import into another and start a job
  successfully, with no manual editing of the files.
- **SC-004**: No placeholder or disabled "coming soon" control appears on the Datasets page.
- **SC-005**: In the Playwright first-time-user run, a new user saves and reuses a dataset without
  help.

## Assumptions

- Visibility stays as spec 007 defined it: every authenticated user can see and use every
  dataset; only the owner edits or deletes.
- Re-granting folder access depends on the browser; where it isn't possible, re-picking the
  folder is the supported path.
- The existing Library page (datasets downloaded to the browser's local library: job results and
  demos) stays separate. The Datasets page is about input videos, the Library about results; each
  page says which it is to avoid confusion.
- Matching by relative path, name and size is sufficient; content hashing of input videos is left
  to spec 017's provenance.

## Dependencies

- Spec 007 (datasets and presets API), spec 008 (`dataset_id` on jobs, server-folder ingest).
- Phase 2's "Run it again" and "Reuse settings on new videos" items build on this.
