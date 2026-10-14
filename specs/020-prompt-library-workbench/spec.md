# Feature Specification: Prompt Library and Prompt Workbench

**Feature Branch**: `020-prompt-library-workbench` (work lands on `1.6-dev`)
**Created**: 2026-10-04
**Status**: Draft
**Input**: v1.6.0 roadmap, Phase 2 "Run it again": **Prompt library**: every VLM prompt used, in
a job or a preview, saved once (deduplicated by SHA-256) with its model, first/last used and the
jobs that used it; a browser to search, view, diff, name, star and reuse them. **Prompt
workbench**: the "test prompt" panel as a standalone page: pick a video and frame (or burst), a
model and a prompt; run; compare responses side by side across prompt versions or models; send
the winner to a job or preset. Also through the CLI/MCP.

## Context

The `vlm_annotation` pipeline labels sampled frames by asking a vision-language model a question
(the prompt). The prompt is the main thing a researcher changes, and small wording changes move
results a lot. Today:

- A prompt lives only inside a job's settings. To find "the prompt I used in March" a researcher
  has to remember which job it was.
- The only place to try a prompt before running a job is a small panel inside the wizard's
  settings step. It tests one frame of an uploaded video, with one prompt and one model at a time,
  and forgets the result when the panel closes.

Prompt design is iterative and comparative, so it needs a record of what was tried and room to
compare.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Every prompt I've used is kept and findable (Priority: P1)

Every prompt that runs (in a job or a preview) is saved once in a prompt library with the models
it ran with, when it was first and last used, and the jobs that used it. The researcher browses
and searches the library, opens a prompt to see its full text and history, gives it a name,
stars it, and compares two versions word by word.

**Why this priority**: the library is the record (provenance for methods sections, spec 017) and
the input for everything else: reuse, the workbench, comparisons.

**Independent Test**: run two VLM jobs with different prompts and one preview with a third; the
library holds three prompts, each with its models, jobs and dates; naming, starring, search and a
two-prompt diff work.

**Acceptance Scenarios**:

1. **Given** a VLM job or preview runs, **When** its prompt was never used before, **Then** a
   library entry is created with the full text, a SHA-256 identity, the model, and first-use time;
   **When** the same text was used before, **Then** the existing entry's last-use time, models and
   jobs are updated instead.
2. **Given** the library, **When** the researcher searches by text, name, model or tag, **Then**
   matching prompts are listed, starred first, then by last use.
3. **Given** two prompts, **When** the researcher compares them, **Then** the differences are
   shown word by word.
4. **Given** a prompt, **When** the researcher names, stars or tags it, **Then** that is saved and
   shown to every user of the server, with who did it.
5. **Given** a prompt, **When** the researcher chooses Use, **Then** the wizard's VLM prompt field
   (or the workbench) is filled with its exact text.

---

### User Story 2 - Try prompts and models side by side on real frames (Priority: P1)

The researcher opens the Prompt Workbench page, picks a video (already on the server, or from a
past job), a moment in it, single frame or a short burst, then one or more prompts and one or more
models. They run, and see each combination's label, reasoning and timing side by side. They
change a word and run again; earlier results stay visible to compare. When one works, they send it
to a new job or save it as a preset.

**Why this priority**: this is the iterative design loop that the wizard panel is too small for,
and the place most VLM prompts will be written.

**Independent Test**: on the demo clip, run two prompts × two models on one frame; four results
appear side by side; edit one prompt and rerun; the new results appear next to the old; "Send to
job" opens the wizard with that prompt and model.

**Acceptance Scenarios**:

1. **Given** a video the server can read and a moment in it, **When** the researcher runs a
   prompt with a model, **Then** the frame (or burst) used, the label, the reasoning, the raw
   response and the timing are shown.
2. **Given** several prompts and/or models selected, **When** run, **Then** each combination's
   result is shown side by side for the same frame(s).
3. **Given** earlier runs in this session, **When** a prompt is edited and run again, **Then** the
   new result is added next to the earlier ones rather than replacing them.
4. **Given** a result, **When** the researcher chooses Send to job or Save as preset, **Then** the
   wizard (or preset) gets that prompt, model and sampling settings exactly.
5. **Given** a moment where frames are unreadable or the model server is down, **When** run,
   **Then** that cell shows the error, and the others still run.
6. **Given** several moments chosen (e.g. five frames across the video), **When** run, **Then**
   results are shown per moment, so a prompt is judged on more than one frame.

---

### User Story 3 - The same from the command line (Priority: P2)

A researcher or an agent lists, searches and shows library prompts, and previews a prompt on a
video frame, from the CLI (and later MCP, Phase 4), with the same results as the page.

**Why this priority**: "agents are users too": an agent iterating on a prompt needs the same loop.

**Independent Test**: `videoannotator prompts list|show|diff` and `videoannotator vlm preview
<video> --at 12.5 --prompt-file p.txt --model m` give the same entries and results as the page.

**Acceptance Scenarios**:

1. **Given** the library, **When** listed or searched from the CLI, **Then** the results match the
   page.
2. **Given** a video and time, **When** previewed from the CLI, **Then** the label and reasoning
   match the workbench for the same inputs, and the prompt is recorded in the library.

---

### Edge Cases

- Prompts differing only in whitespace at the ends: stored as distinct texts (exact identity is
  what reproducibility needs), but the diff view says the difference is whitespace only.
- Very long prompts (several KB): stored in full; lists show the first lines.
- A model removed from the model server since a prompt used it: still shown in its history; using
  it again reports the model as unavailable.
- Deleting a prompt that jobs used: not allowed (it's part of their provenance); it can be hidden
  from the default list instead.
- Concurrent users editing the same prompt's name or tags: last write wins, and the history
  records who changed it.
- Workbench on a remote/hosted model backend: the page says plainly that frames leave the
  machine (Local-First principle), as the wizard does.
- Workbench session results are not jobs: they are kept until the page is closed or cleared,
  except that every prompt run is recorded in the library.

## Requirements *(mandatory)*

### Functional Requirements

**Library**

- **FR-001**: Every prompt run by a VLM job or a preview (wizard panel, workbench, CLI) MUST be
  recorded in a prompt library, once per distinct text, identified by the SHA-256 of its exact
  text.
- **FR-002**: Each library entry MUST keep the full text, first-use and last-use times, the models
  it ran with, the jobs that used it, and who first used it.
- **FR-003**: Users MUST be able to name, star, tag and hide entries; changes are shared on the
  server and attributed.
- **FR-004**: The library MUST be searchable by text, name, tag and model, and sortable by last
  use, with starred entries first by default.
- **FR-005**: Users MUST be able to compare two entries word by word.
- **FR-006**: Users MUST be able to send an entry's exact text to the job wizard or the workbench.
- **FR-007**: Entries used by jobs MUST NOT be deletable.

**Workbench**

- **FR-008**: A standalone Prompt Workbench page MUST let a user choose a video the server can
  read (a server folder video or a past job's video), one or more moments, single frame or burst
  sampling, one or more prompts and one or more models.
- **FR-009**: Running MUST show, for each prompt × model × moment, the frame(s) used, label,
  reasoning, raw response and timing, side by side.
- **FR-010**: Results from earlier runs in a session MUST stay visible beside new ones until
  cleared.
- **FR-011**: A failure in one combination MUST be shown in its place without stopping the others.
- **FR-012**: A result MUST be sendable to a new job or a preset with its prompt, model and
  sampling settings.
- **FR-013**: The wizard's existing test panel MUST link to the workbench with its current video,
  prompt and model.

**CLI**

- **FR-014**: The CLI MUST offer listing, searching, showing and diffing library prompts, and
  previewing a prompt on a video at a given time, recording it in the library.

### Key Entities

- **Prompt** (library entry): exact text, SHA-256, name, tags, starred, hidden, first/last use,
  first user.
- **Prompt use**: one prompt run with one model in one job or preview, at a time, by a user.
- **Workbench run**: one prompt × model × moment, with the frame(s) used and the model's response;
  session-scoped.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of prompts run in jobs and previews appear in the library, and no text appears
  twice.
- **SC-002**: A researcher finds a prompt used in a past job in under 30 seconds by searching any
  word in it.
- **SC-003**: Comparing 3 prompts × 2 models on 5 moments takes one action to start, and every
  result is visible on one screen.
- **SC-004**: A prompt chosen in the workbench reaches a running job with zero retyping.
- **SC-005**: For identical inputs, the workbench, the wizard panel and the CLI preview give the
  same label.

## Assumptions

- Previews already exist on the server (`POST /api/v1/vlm/preview`, `GET /api/v1/vlm/models`);
  this feature adds the library, the page and multi-run orchestration around them.
- The library is shared across users of a server, as presets and datasets are (spec 007).
- The prompt SHA-256 is the same identity spec 017's provenance records, so jobs, library and
  methods paragraphs agree.
- Running many combinations is sequential or lightly parallel, limited by the local model server;
  the page shows progress.

## Dependencies

- Spec 017 (provenance records the prompt SHA-256 and model digest).
- Spec 019 (Send to job reuses the wizard pre-fill), spec 007 (presets).
- Phase 4 adds MCP access to the same operations.
