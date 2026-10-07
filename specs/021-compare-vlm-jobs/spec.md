# Feature Specification: Compare Two VLM Jobs

**Feature Branch**: `021-compare-vlm-jobs` (work lands on `1.6-dev`)
**Created**: 2026-10-04
**Status**: Draft
**Input**: v1.6.0 roadmap, Phase 2: "Compare two VLM jobs on the same video: their labels on one
timeline, with the frames where they disagree listed, and ELAN ground truth as a third row when
there is one. The workbench compares prompts on single frames; this compares whole runs. Asked for
in spec 009's viewer handoff; comparing across a whole dataset stays in Phase 6."

## Context

A researcher changes a VLM prompt or model (spec 020's workbench, spec 019's Edit and run again)
and runs it again on the same video. The question is then: what changed, and is it better? The
viewer can already compare one VLM job against ELAN ground truth at one moment. It can't put two
VLM runs side by side, so today the answer means opening two tabs and scrubbing both.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - See where two runs disagree (Priority: P1)

The researcher picks two finished jobs that ran `vlm_annotation` on the same video (for example a
job and its rerun). The viewer shows both runs' labels as two rows on one timeline under the
video, highlights the moments where they disagree, and lists those moments; clicking one jumps the
video there and shows both runs' labels and reasoning for that frame.

**Why this priority**: it's the whole feature: finding the frames where a prompt change mattered,
without scrubbing.

**Independent Test**: run the demo clip twice with two prompts; open the comparison; the two rows,
the disagreement list and the per-moment detail match the two jobs' output files.

**Acceptance Scenarios**:

1. **Given** two jobs with VLM output on the same video, **When** compared, **Then** both runs'
   labels are shown as rows on one timeline aligned to the video, each row named with its job,
   prompt name (or start of text), model and version.
2. **Given** both runs sampled the same moments, **When** compared, **Then** each moment is marked
   agree or disagree, and disagreements are listed with time and both labels.
3. **Given** the list, **When** the researcher selects a moment, **Then** the video seeks there and
   both runs' label, reasoning and raw response for that moment are shown side by side.
4. **Given** the comparison, **Then** a summary shows how many moments were compared, how many
   agree, and the agreement rate, plus a table of label pairs (how often run A said X when run B
   said Y).

---

### User Story 2 - Measure both against ground truth (Priority: P2)

When the video has an ELAN ground-truth file (in the job, or loaded alongside), it appears as a
third row, and each run's agreement with ground truth is shown, so the researcher can see which
run is closer to the human coding and where.

**Why this priority**: disagreement alone says what changed; ground truth says which is right.

**Independent Test**: with an `.eaf` for the demo clip, the comparison shows a ground-truth row
and, per run, its agreement with ground truth over the compared moments.

**Acceptance Scenarios**:

1. **Given** ELAN ground truth for the video, **When** compared, **Then** it appears as a third
   row, labelled as ground truth from that file.
2. **Given** ground truth, **Then** each run's agreement with it is shown, and the disagreement
   list can be filtered to moments where exactly one run matches ground truth.
3. **Given** a ground-truth tier with labels that don't match the VLM's label set, **Then** the
   researcher maps labels (or picks the tier) before agreement is computed, as the existing
   VLM-vs-ELAN view does.

---

### User Story 3 - Find the comparison from where I am (Priority: P2)

From a VLM job's page, the researcher chooses "Compare with…" and gets the other jobs on the same
video that ran `vlm_annotation`, its reruns first. From a rerun's page, "Compare with original" is
one click.

**Why this priority**: the comparison is only used if it's offered at the moment a rerun finishes.

**Independent Test**: after a VLM rerun completes, its page offers Compare with original, which
opens the comparison of the two.

**Acceptance Scenarios**:

1. **Given** a completed VLM job, **When** Compare with… is chosen, **Then** other completed VLM
   jobs on the same video are offered, reruns and originals first, newest next.
2. **Given** a VLM job that is a rerun (spec 019), **Then** Compare with original is offered on
   its page.
3. **Given** a comparison, **Then** it has its own link that can be bookmarked or shared with
   another user of the server.

---

### Edge Cases

- Different sampling (one run every 5 s, the other every 2 s, or burst vs single frame): moments
  are paired by nearest time within half the larger interval; unpaired moments are shown as such
  and excluded from agreement, and the summary says how many.
- Same video name but different content (re-encoded file): compared only when the videos are the
  same file (same stored video, or matching content hash from spec 017); otherwise the viewer says
  they are different videos.
- A run with error entries (a sample point the model failed on): shown as errors, excluded from
  agreement, counted in the summary.
- Labels differing only in case or whitespace: compared as recorded (Principle VI: no silent
  normalisation); a visible option normalises case for the agreement figures, labelled as such.
- One job still running: not offered until complete.
- Very long videos (thousands of moments): timeline and list stay responsive.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The viewer MUST compare two completed jobs' `vlm_annotation` outputs on the same
  video, as two timeline rows aligned to the video.
- **FR-002**: Each row MUST be labelled with its job, prompt (name or opening text), model and
  version (spec 017's attribution).
- **FR-003**: Moments MUST be paired by time (exactly, or nearest within half the larger sampling
  interval), and unpaired moments shown as unpaired.
- **FR-004**: The viewer MUST list disagreeing moments with time and both labels; selecting one
  seeks the video and shows both runs' label, reasoning and raw response.
- **FR-005**: The viewer MUST summarise moments compared, unpaired, errors, agreement count and
  rate, and a label-pair table.
- **FR-006**: When ELAN ground truth exists for the video, it MUST appear as a third row, with each
  run's agreement with it, using the existing tier/label mapping.
- **FR-007**: Labels MUST be compared exactly as recorded; any normalisation MUST be an explicit,
  visible option.
- **FR-008**: Jobs on different videos MUST NOT be compared.
- **FR-009**: Job pages MUST offer Compare with… (other VLM jobs on the same video) and, for
  reruns, Compare with original.
- **FR-010**: A comparison MUST have a stable link identifying its two jobs.
- **FR-011**: The disagreement list and summary MUST be exportable as CSV.

### Key Entities

- **Comparison**: two job IDs (and optionally a ground-truth file), identified by its link; not
  stored on the server.
- **Paired moment**: a time, each run's annotation (or none), agree/disagree/unpaired/error, and
  the ground-truth label when present.
- **Agreement summary**: counts, rate and label-pair table, for runs against each other and each
  against ground truth.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: From a finished rerun, a researcher sees every frame where it differs from the
  original in under 30 seconds and two clicks.
- **SC-002**: The agreement figures match a hand count from the two output files on the demo clip
  exactly.
- **SC-003**: With ground truth loaded, the researcher can say which run agrees more with the human
  coding without leaving the comparison.
- **SC-004**: No comparison silently drops a moment: compared + unpaired + error moments add up to
  all moments in both runs.

## Assumptions

- No new server storage: comparisons are computed in the viewer from the two jobs' existing output
  files (spec 009's handoff), so a comparison is reproducible from its link.
- Two jobs only; more than two, and whole datasets, are Phase 6 (corpus view).
- The tidy export (Phase 3) will offer the same paired table for analysis in R/Python; this
  feature's CSV covers one comparison.

## Dependencies

- Spec 017 (provenance for row labels and same-video checks), spec 019 (reruns and
  Compare with original), spec 020 (prompt names).
- Existing VLM-vs-ELAN comparison in the viewer (`VlmAnnotationPanel`), which this generalises.
