# Feature Specification: Datasets & Saved Pipeline Presets

**Feature Branch**: `007-datasets-and-presets`
**Created**: 2026-08-26
**Status**: Draft
**Input**: User description: "Let a researcher save a named set of videos (a dataset) and a named set of pipeline selections + config (a preset) for reuse, instead of re-picking files and re-typing configuration on every job submission. Server-side, DB-backed, so it survives cleared browser data and is usable by anyone with access to the same server — not a client-only/single-browser feature."

## Relationship to Existing Specs

This is the "related, downstream feature" that
[`specs/005-pipeline-extras-install/spec.md`](../005-pipeline-extras-install/spec.md) explicitly
named and deferred (Assumptions: *"Saved pipeline configurations / dataset presets ... depends on
this one landing first, but is explicitly out of scope... will be proposed as its own future
spec"*). It also depends on
[`specs/006-job-execution-consolidation/spec.md`](../006-job-execution-consolidation/spec.md)
having landed first, only in the sense that
[`specs/008-batch-group-workflow/`](../008-batch-group-workflow/) (which depends on *this* spec)
needs 006's reliable execution path — this spec itself has no functional dependency on 006.

This spec covers the **backend capability only**. The consumer — a Datasets page (today a disabled
mockup, `src/pages/Datasets.tsx`), "save as preset"/"load preset" affordances in the job-creation
wizard — is a UI concern for `video-annotation-viewer`'s own spec. See
[`viewer-handoff.md`](viewer-handoff.md).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Save a dataset once, reuse it every run (Priority: P1)

A researcher has a folder of 40 videos they'll run the same analysis against repeatedly over weeks
as they refine their prompt. Today they re-pick all 40 files from a browser file dialog every single
time. They save the folder once as a named dataset and select it by name for every future
submission.

**Why this priority**: This is the single biggest piece of repeated manual effort in the current
workflow — the entire motivation for this spec.

**Independent Test**: Save a dataset from a set of video files with a name and description; retrieve
it later by id; confirm it lists the same set of video filenames that were originally saved.

**Acceptance Scenarios**:

1. **Given** a set of video files and a chosen name, **When** a user saves them as a dataset,
   **Then** the system persists a record identifying that set of videos by filename and size, under
   that name, associated with the saving user.
2. **Given** a previously saved dataset, **When** a user lists their datasets, **Then** they see it
   by name along with when it was created and (once used) when it was last used.
3. **Given** a previously saved dataset whose files have since changed locally (renamed, moved, one
   removed), **When** a client attempts to reuse it, **Then** the system's manifest allows the client
   to detect and report the mismatch rather than silently proceeding as if nothing changed.

---

### User Story 2 - Save a pipeline+config preset once, reuse it every run (Priority: P1)

The same researcher has settled on a specific set of pipelines and a carefully-worded VLM prompt.
Today they re-select pipelines and re-type (or re-paste) the exact prompt text on every submission,
risking small unintentional variations between runs. They save the current selection and
configuration as a named preset and apply it in one action on future submissions.

**Why this priority**: Equal to User Story 1 — together these two are the whole point of the spec.
For prompt-based pipelines specifically, exact reproducibility of wording matters for research
validity, not just convenience.

**Independent Test**: Save a preset consisting of a specific pipeline selection and configuration
(including a multi-line prompt value); retrieve it later; confirm the selected pipelines and every
configuration value, including the prompt text verbatim, match what was saved.

**Acceptance Scenarios**:

1. **Given** a chosen set of pipelines and their configuration values, **When** a user saves them as
   a named preset, **Then** the system persists the exact selection and configuration under that
   name, associated with the saving user.
2. **Given** a previously saved preset, **When** a user applies it to a new job submission, **Then**
   the resulting selected pipelines and configuration values are identical to what was saved —
   including any long free-text field (e.g. a VLM prompt) reproduced verbatim.
3. **Given** a previously saved preset that references a pipeline no longer available on this server
   (e.g. its extras group was never installed, or was removed), **When** a user retrieves it,
   **Then** the system still returns the preset but indicates which referenced pipeline(s) are
   currently unavailable, rather than failing to return it at all.

---

### User Story 3 - Share configurations across a research team (Priority: P2)

A second researcher on the same server (e.g. a supervisor reviewing the first researcher's setup)
opens the datasets/presets list and can see and reuse what's already been saved, without needing the
original files or configuration relayed to them by hand.

**Why this priority**: Lower than US1/US2 because it's a consequence of the server-side design
choice rather than a distinct capability to build, but it's the reason this spec is server-side (DB-
backed) rather than client-local (browser storage) — worth its own scenario to make sure the design
actually delivers it.

**Independent Test**: Save a dataset/preset as one authenticated user; confirm a second
authenticated user on the same server can list and retrieve it.

**Acceptance Scenarios**:

1. **Given** a dataset or preset saved by one user, **When** a different authenticated user on the
   same server lists datasets/presets, **Then** they see it.
2. **Given** a dataset or preset saved by one user, **When** a different user attempts to modify or
   delete it, **Then** the request is rejected — visibility is shared, but modification is not.

---

### Edge Cases

- Two datasets/presets owned by different users happen to share the same name: MUST NOT collide —
  names are unique per owner, not globally.
- A dataset or preset is deleted while a job record exists that was originally submitted using it:
  deleting MUST NOT corrupt, orphan, or cascade-delete that job's own historical record.
- Importing an exported dataset/preset definition whose name collides with one the importing user
  already owns: MUST be handled deterministically (reject with a clear error, or auto-disambiguate)
  — never a silent overwrite.
- A preset's saved configuration includes a value for a config field that no longer exists on the
  referenced pipeline (the pipeline's schema changed since saving): retrieval MUST still succeed;
  deep field-level validation against the current schema is explicitly out of scope (see
  Assumptions).
- A dataset's video manifest is empty (all videos removed from an otherwise-valid saved dataset):
  MUST remain a valid, listable dataset — not an error state — since removing all entries is a
  legitimate (if unusual) editing action.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST allow an authenticated user to save a named dataset representing a set
  of video files, with an optional description.
- **FR-002**: A saved dataset MUST record, per video, at minimum its filename and size, sufficient
  for a client to detect whether a locally re-selected set of files matches what was originally
  saved.
- **FR-003**: The system MUST allow an authenticated user to save a named pipeline preset consisting
  of a set of selected pipeline identifiers and their per-pipeline configuration values.
- **FR-004**: The system MUST allow listing, retrieving, updating, and deleting datasets and presets.
- **FR-005**: Saved datasets and presets MUST record an owning user and MUST be readable by any
  authenticated user of the server (matching existing job-listing visibility); modification and
  deletion MUST be restricted to the owner or an administrator.
- **FR-006**: The system MUST support exporting a saved dataset or preset as a self-contained
  definition that can be re-imported, including on a different server instance.
- **FR-007**: Importing a dataset/preset definition MUST be rejected or auto-disambiguated, never
  silently overwritten, when its name collides with one already owned by the importing user.
- **FR-008**: Deleting a saved dataset or preset MUST NOT affect the historical record of any job
  previously submitted using it.
- **FR-009**: Retrieving a saved preset that references a pipeline currently unavailable on the
  server MUST succeed and indicate which referenced pipeline(s) are unavailable, rather than failing.
- **FR-010**: The system MUST record, for each saved dataset/preset, its creation time and (once
  applied to a job submission at least once) the time it was last used.

### Key Entities

- **Saved dataset**: a named, owned collection of remembered video identities (filename + size per
  video), with a description and usage metadata — not the video files themselves.
- **Saved pipeline preset**: a named, owned combination of selected pipeline identifiers and their
  configuration values, with a description, tags, and usage metadata.

## API Contract for Downstream Consumers

- **Datasets**: `GET/POST /api/v1/datasets`, `GET/PUT/DELETE /api/v1/datasets/{id}` — standard CRUD.
  Each record includes its video manifest, `created_at`, `updated_at`, `last_used_at`.
- **Presets**: same CRUD shape at `/api/v1/presets`. Each record's `config`/`selected_pipelines`
  mirror exactly what a job submission's own `config`/`selected_pipelines` fields already look like
  today, so applying a preset is a direct copy into a new job submission with no translation needed.
  A retrieved preset includes an `unavailable_pipelines: string[]` field (empty when everything
  referenced is currently available).
- **Export/import**: `GET` on either resource already returns a self-contained definition; `POST`
  accepts either a fresh definition or a previously-exported one — no separate export/import
  endpoints needed.
- **Stability expectation**: once published, these shapes are a stable contract for the viewer,
  matching spec 004/005's precedent.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Submitting a job against a previously-saved dataset requires selecting it once, not
  re-picking each individual video file.
- **SC-002**: A preset applied to a new job submission reproduces its exact saved
  `selected_pipelines` and `config` — including a multi-line prompt field verbatim — verified by
  test.
- **SC-003**: Deleting a dataset or preset never breaks retrieval of a job that already used it,
  verified by test.
- **SC-004**: Exporting then re-importing a dataset or preset produces an equivalent, immediately
  usable saved item, verified by a round-trip test.
- **SC-005**: A dataset/preset saved by one user is visible to a second authenticated user on the
  same server but not modifiable by them, verified by test.

## Assumptions

- **Visibility is shared-read by default**, matching how job listings already behave on this
  server — not private-per-user. Per-team or private visibility scoping is a possible future
  refinement, not in scope here.
- **Deep config validation against a pipeline's current schema is out of scope** beyond FR-009's
  availability flag — a saved preset is stored data, not re-validated field-by-field against schema
  drift at save or retrieval time.
- **No video file content is ever stored or uploaded as part of saving a dataset** — only the
  manifest (filename, size). The actual files remain wherever the client can access them locally;
  this spec is metadata-only, consistent with keeping the core install and this feature's storage
  footprint small.
- This spec has no functional dependency on `006-job-execution-consolidation`. It is, however, a
  prerequisite for `008-batch-group-workflow` (dataset-tagged job submission) and
  `010-corpus-analysis-foundations` (dataset-wide aggregation).
