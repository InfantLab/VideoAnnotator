# Feature Specification: Run It Again

**Feature Branch**: `019-run-it-again` (work lands on `1.6-dev`)
**Created**: 2026-10-04
**Status**: Draft
**Input**: v1.6.0 roadmap, Phase 2 "Run it again": rerun a job; reuse settings on new videos; UX
cues that lead there. "A researcher who likes a result wants the same settings on more videos;
one who doesn't wants to tweak and rerun. Today both mean rebuilding the job in the wizard from
memory, although every job already stores its `selected_pipelines` and `config`."

## Context

The path a researcher takes doesn't end at reviewing a result. Two things usually come next:

- **It worked**: run the same settings on more videos (the rest of the study, a new batch of
  recordings).
- **It didn't quite work**: change a setting (a threshold, a prompt, a model) and run the same
  videos again, then compare.

Today neither has a direct path. "Retry" exists, but only for failed or cancelled jobs, and it
resets the same job in place, discarding its results. The wizard can be pre-filled from a failed
job, but that path is reached only from a failure. Presets (spec 007) save settings, but nothing
offers to save a job's settings as one.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Rerun a job, as is or changed (Priority: P1)

From a finished job (completed, completed with errors, failed or cancelled), the researcher
chooses "Run again". The same videos and settings become a new job, which links back to the
original; the original and its results stay untouched. "Edit and run again" opens the wizard with
the videos and settings filled in, to change something first.

**Why this priority**: tweak-and-rerun is the core iteration loop of annotation work, and the
comparison it enables (old vs new) needs both jobs kept.

**Independent Test**: complete a job, choose Run again; a second job runs with identical videos,
pipelines and settings, shows "Rerun of <original>", and the original's results are unchanged.
Then choose Edit and run again, change one setting, run; the third job differs only in that
setting.

**Acceptance Scenarios**:

1. **Given** a finished job, **When** the researcher chooses Run again, **Then** a new job is
   created with the same videos, pipelines and settings, records which job it reruns, and the
   original keeps its status and results.
2. **Given** a finished job, **When** the researcher chooses Edit and run again, **Then** the
   wizard opens with the videos chosen and the pipelines and settings filled in, and says which
   job it started from.
3. **Given** a rerun job, **When** its page is viewed, **Then** it links to the job it reruns, and
   the original links to its reruns.
4. **Given** a job whose original video is no longer stored, **When** Run again is chosen,
   **Then** the viewer says the video is gone and offers Edit and run again to choose it anew.
5. **Given** a batch (run of many videos), **When** the researcher chooses Run again on the batch,
   **Then** a new batch reruns every video in it with each job's settings, linked to the original
   batch.

---

### User Story 2 - Use these settings on new videos (Priority: P1)

From a job or a batch whose results they like, the researcher chooses "Use these settings". The
wizard opens at "Choose videos" with the pipelines and settings filled in, so they only pick the
new videos. "Save as preset" stores the settings under a name for later.

**Why this priority**: scaling a validated setup to the rest of a study is the most common next
step after a good result.

**Independent Test**: from a completed job, choose Use these settings, pick two other videos,
start; the new jobs have the original's pipelines and settings. Choose Save as preset; the preset
appears in the wizard's preset list with those settings.

**Acceptance Scenarios**:

1. **Given** a job or batch, **When** Use these settings is chosen, **Then** the wizard opens at
   Choose videos with pipelines and settings filled in and the source named.
2. **Given** a job or batch, **When** Save as preset is chosen and a name given, **Then** a preset
   with exactly its pipelines and settings is saved (the existing preset store).
3. **Given** settings naming a pipeline no longer installed, **When** they are reused, **Then**
   the rest are applied and the missing pipeline is shown as unavailable, as presets do today.

---

### User Story 3 - The next step is offered where the decision is made (Priority: P2)

The actions appear where the researcher is looking when they decide, not hidden in a menu: on the
job and batch result pages, in the wizard's first step, and in the messages for failed or partial
jobs.

**Why this priority**: the roadmap's structural pass judges features by whether a first-time user
finds them; buttons nobody sees don't change the workflow.

**Independent Test**: in the Playwright first-time-user run, a new user who has finished one job
reruns it with a changed setting, and runs its settings on another video, without instructions.

**Acceptance Scenarios**:

1. **Given** a completed job's or batch's page, **Then** Run again, Edit and run again, Use these
   settings and Save as preset are visible near the results without opening a menu.
2. **Given** the wizard's first step, **Then** it offers "Start from a previous job" and the most
   recent presets before the blank form.
3. **Given** a failed or partly failed job, **Then** its error message offers "Fix settings and
   run again" (Edit and run again).
4. **Given** a completed batch, **Then** it suggests "Run on more videos" (Use these settings).

---

### Edge Cases

- Rerunning a job that is itself a rerun: links to its direct original; the chain is navigable.
- Rerunning while the original is still running: not offered until it has finished.
- Settings that include secrets (tokens): reused as stored on the server; never shown in clear in
  the wizard.
- A job from before this feature: Run again and Use these settings work from its stored
  pipelines and settings; it simply has no rerun links.
- A rerun of a server-folder run: uses the same folder; if videos were added or removed there
  since, the new run says how many videos it found compared with the original.
- Deleting an original job: its reruns remain and say the original was deleted.
- Retry (reset in place) stays for failed/cancelled jobs, and is labelled so users can tell it
  from Run again.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST create a new job from any finished job, with the same videos,
  pipelines and settings, recording the original as `rerun_of`. The original MUST NOT change.
- **FR-002**: The system MUST create a new batch from a finished batch the same way, each new job
  recording its original.
- **FR-003**: The viewer MUST offer Edit and run again, opening the wizard pre-filled with the
  job's videos, pipelines and settings, and naming the source job.
- **FR-004**: The viewer MUST offer Use these settings on a job or batch, opening the wizard at
  Choose videos with the pipelines and settings filled in.
- **FR-005**: The viewer MUST offer Save as preset on a job or batch, saving its pipelines and
  settings to the existing preset store.
- **FR-006**: Job and batch pages MUST link a rerun to its original and an original to its reruns.
- **FR-007**: When the original video is no longer stored, Run again MUST say so and offer Edit
  and run again instead.
- **FR-008**: The actions MUST be visible on job and batch result pages without opening a menu;
  the wizard's first step MUST offer Start from a previous job and recent presets; failed and
  partial jobs MUST offer Fix settings and run again; completed batches MUST suggest Run on more
  videos.
- **FR-009**: Run again and Use these settings MUST also be available from the CLI and API (agents
  are users too).
- **FR-010**: Pipelines unavailable on the server MUST be reported, not silently dropped, when
  settings are reused.
- **FR-011**: The existing Retry (reset in place) MUST keep working and be labelled distinctly.

### Key Entities

- **Job** (exists): gains `rerun_of` (the job it reruns, optional).
- **Batch** (exists): a rerun batch records the batch it reruns.
- **Wizard start state**: where a wizard session began (blank, a job, a batch, a preset), shown to
  the user and used to pre-fill.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Rerunning a finished job with one changed setting takes under 30 seconds of the
  researcher's time, from its page to the new job starting.
- **SC-002**: Running a validated job's settings on new videos requires no re-entry of any setting
  (zero fields typed beyond choosing videos).
- **SC-003**: 100% of rerun jobs link to their original, and the original's results are
  byte-for-byte unchanged after a rerun.
- **SC-004**: In the Playwright first-time-user run, the new user completes both "rerun with a
  change" and "same settings on another video" without help.

## Assumptions

- A rerun reuses the stored copy of each video; it doesn't re-upload.
- Comparing an original with its rerun is covered for VLM jobs by spec 021; general side-by-side
  comparison of other pipelines is out of scope here.
- Settings reuse copies settings at the moment of use; later edits to the source job (none are
  possible today) or preset don't affect jobs already created.

## Dependencies

- Spec 007 (presets), spec 008 (batches, batch retry), spec 018 (saved datasets: Use these
  settings can also start from a dataset).
