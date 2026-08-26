# Feature Specification: Corpus-Wide Analysis Foundations

**Feature Branch**: `010-corpus-analysis-foundations`
**Created**: 2026-08-26
**Status**: Draft
**Input**: User description: "Give a researcher an at-a-glance view of an entire dataset's results — completion coverage, label distribution, and ground-truth agreement where available — without opening every video individually. Deliberately scoped as foundations: basic counts and rates, not a full statistics suite. Deeper analysis (kappa, significance testing, plots) stays in the existing Python research repo."

## Relationship to Existing Specs

Depends on both [`specs/007-datasets-and-presets/spec.md`](../007-datasets-and-presets/spec.md)
(the dataset concept this aggregates over) and
[`specs/008-batch-group-workflow/spec.md`](../008-batch-group-workflow/spec.md) (the `dataset_id`
job-tagging that links jobs to a dataset) — cannot be built before both land.

Deliberately narrow in scope, matching the division of labor already established in
[`docs/development/vlm_annotation_pipeline.md`](../../docs/development/vlm_annotation_pipeline.md):
this repo's job is making a corpus's results *visible*, not computing inter-rater reliability or
running significance tests — that stays in the `mother-infant-touch-detection` research repo, which
already does it well.

Backend-only. The consumer — a Dataset Overview page (thumbnail grid of the dataset's videos with
per-video status, doubling as the entry point for both batch review and corpus analysis) — is
`video-annotation-viewer`'s own spec. See [`viewer-handoff.md`](viewer-handoff.md).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - See coverage and label distribution across a whole dataset (Priority: P1)

A researcher has run `vlm_annotation` against most of a 40-video dataset over several sessions. They
want to know, at a glance: how many videos have results, and what the overall label split looks like
(e.g. what fraction of sampled frames across the whole corpus were classified as `TOUCH`) — without
opening each video's results individually and mentally tallying.

**Why this priority**: This is the direct payoff of everything datasets/batches/VLM work in this
effort has been building toward — a corpus-level answer, not a video-by-video slog.

**Independent Test**: For a dataset where some videos have a completed `vlm_annotation` job and
others don't, request its summary; confirm it correctly reports per-video coverage and a label
distribution matching manual aggregation of the covered videos' actual stored annotations.

**Acceptance Scenarios**:

1. **Given** a dataset with a mix of videos that do and don't yet have a completed `vlm_annotation`
   job, **When** its summary is requested, **Then** the response correctly identifies, per video,
   whether results exist and (if so) which job produced them.
2. **Given** a dataset where several videos have completed `vlm_annotation` jobs, **When** its
   summary is requested, **Then** the reported label distribution matches the true counts across
   those videos' stored annotations.
3. **Given** a video in the dataset that has been run more than once (e.g. re-run with a revised
   prompt), **When** the summary is computed, **Then** only that video's most recently completed job
   contributes to the aggregate — not every historical run for it.

---

### User Story 2 - See ground-truth agreement across the dataset, where available (Priority: P2)

Of the dataset's videos, some have ELAN ground truth available and some don't (Irene codes ground
truth incrementally). The researcher wants an aggregate agreement rate across whichever videos
actually have it, clearly scoped to that subset.

**Why this priority**: Lower than User Story 1 because it depends on ground truth existing for at
least some videos (not guaranteed), but it's the natural corpus-scale extension of the single-video
VLM-vs-ELAN comparison already shipped.

**Independent Test**: For a dataset where ELAN ground truth exists for a known subset of videos,
request its summary; confirm the reported agreement rate is computed only over that subset and the
response states how many videos it covers.

**Acceptance Scenarios**:

1. **Given** a dataset where ELAN ground truth exists for some but not all videos, **When** its
   summary is requested, **Then** the agreement rate is computed only over the videos that have
   ground truth, and the response states that coverage count explicitly.
2. **Given** a dataset with no ELAN ground truth available for any video, **When** its summary is
   requested, **Then** the response indicates no agreement data is available, rather than a
   misleading zero or absent field.

---

### Edge Cases

- A dataset contains jobs from more than one distinct `vlm_annotation` configuration (different
  prompts run at different times across different videos): the summary MUST clearly aggregate only
  each video's most recent completed run, and MUST NOT silently blend incompatible prompt runs
  together into one label distribution without that being the documented behavior (see Acceptance
  Scenario 3 above).
- A dataset is large (dozens of videos, each with potentially hundreds of frame annotations): summary
  computation MUST remain a reasonably fast, synchronous API response — not require the caller to
  separately fetch and parse every job's full annotation file client-side.
- A dataset has zero videos with any completed job yet: summary MUST return a valid, well-formed
  response reflecting "nothing yet," not an error.
- A video appears in the dataset's manifest but its underlying job was deleted or never actually ran:
  MUST be reported as "no results" for that video, not cause the whole summary request to fail.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide a way to retrieve an aggregate summary for a saved dataset,
  covering the videos it contains and the jobs run against them.
- **FR-002**: The summary MUST report, per video in the dataset, whether a `vlm_annotation` job has
  completed for it and, if so, that job's identifier.
- **FR-003**: The summary MUST report a label distribution aggregated across each video's most
  recently completed `vlm_annotation` job — never blending more than one historical run per video
  into the same aggregate.
- **FR-004**: Where ELAN ground truth is available for a subset of the dataset's videos, the summary
  MUST report an agreement rate computed only over that subset, along with an explicit count of how
  many videos it covers.
- **FR-005**: Summary computation MUST remain practical as a synchronous API response for a dataset
  of realistic research-corpus size (dozens of videos) without requiring the caller to separately
  fetch every individual job's full annotation file.
- **FR-006**: This feature MUST NOT compute or expose statistical measures beyond basic counts and
  rates — no inter-rater reliability coefficients, no significance testing. Deeper statistical
  analysis remains out of scope, consistent with the existing division of labor with the Python
  research repo.

### Key Entities

- **Dataset summary**: an aggregate view, computed over a dataset's videos and their most-recently-
  completed `vlm_annotation` job results — per-video result coverage, corpus-wide label distribution,
  and (where applicable) an ELAN-ground-truth agreement rollup with its own coverage count.

## API Contract for Downstream Consumers

- `GET /api/v1/datasets/{id}/summary` →
  ```
  {
    dataset_id, video_count,
    videos: [{ filename, job_id: string | null, status }],
    label_distribution: { [label: string]: number },
    elan_agreement: { covered_videos: number, agreement_rate: number } | null
  }
  ```
  `elan_agreement` is `null` when no video in the dataset has ELAN ground truth available.
- **Stability expectation**: matches prior specs' precedent once published.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Summary correctly reflects per-video job coverage for a dataset containing a mix of
  completed, pending, and never-run videos, verified by test.
- **SC-002**: Label distribution counts match manual aggregation of the underlying jobs' actual
  stored annotations for a test dataset, verified by test.
- **SC-003**: ELAN agreement rollup is computed only over videos that actually have ground truth
  and correctly states its coverage count, verified by test.
- **SC-004**: Summary computation for a dataset of at least 20 videos completes in under 5 seconds,
  verified by benchmark/test.

## Assumptions

- **Cannot be built before both `007` and `008` land** — needs the dataset concept and the
  `dataset_id` job-tagging mechanism to exist first.
- **"Most recent job per video"** is determined by job creation time among jobs tagged with both this
  dataset's id and the `vlm_annotation` pipeline. Comparing *multiple* historical runs per video
  across a whole dataset (rather than just using the latest) is out of scope here — single-video,
  two-job comparison is spec 009's job; this spec is single-run-per-video, corpus-wide.
- **ELAN ground-truth association** (which video's ground-truth file corresponds to which job) reuses
  whatever mechanism the viewer already establishes for single-video VLM-vs-ELAN comparison — this
  spec assumes that association is resolvable, not inventing a new one.
- **No statistical measures beyond basic counts/rates** — kappa, significance testing, and plots stay
  in the Python research repo, per the existing division of labor.
- **Viewer-side work (Dataset Overview page, summary card) is out of scope here** — see
  [`viewer-handoff.md`](viewer-handoff.md).
