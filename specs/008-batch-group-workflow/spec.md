# Feature Specification: Batch/Group Workflow — Submission Tagging, Real Progress & Group Controls

**Feature Branch**: `008-batch-group-workflow`
**Created**: 2026-08-26
**Status**: Draft
**Input**: User description: "Let jobs submitted together (e.g. 10+ videos with one config in one wizard pass) be tracked, viewed, cancelled, and retried as a group — with real aggregate progress and a time estimate — instead of N independent job rows with no shared identity. Real-time push notifications over the existing (currently stub) SSE stream, replacing polling-only status checks."

## Relationship to Existing Specs

Depends on [`specs/006-job-execution-consolidation/spec.md`](../006-job-execution-consolidation/spec.md)
having landed: group-level cancel/retry only means something if single-job cancel/retry actually
works, and group progress needs the real (non-stub) per-job progress field 006 introduces. Optionally
composes with [`specs/007-datasets-and-presets/spec.md`](../007-datasets-and-presets/spec.md) — a
batch may reference the dataset it was submitted from, but does not require one.

Backend-only. The consumer — a group progress card replacing N independent job rows, bulk
cancel/retry buttons, and prev/next navigation between a batch's videos while reviewing results — is
`video-annotation-viewer`'s own spec. See [`viewer-handoff.md`](viewer-handoff.md).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Submit a group of videos, see one aggregate progress view (Priority: P1)

A researcher submits 10 videos with the same `vlm_annotation` config in one wizard pass. Today this
produces 10 fully independent job rows in the jobs list, each polled and displayed separately, with
no way to tell "how's the whole run going" without mentally aggregating 10 numbers. They instead see
one group with an overall completion percentage and a real time estimate.

**Why this priority**: This is the direct capability gap driving the whole workflow-upgrades effort —
Irene's actual use case is a ~40-video corpus run as one conceptual unit.

**Independent Test**: Submit 3+ jobs tagged with the same batch identifier; query the batch's
aggregate status; confirm it reports correct counts by state and a completion percentage matching the
underlying jobs' true states.

**Acceptance Scenarios**:

1. **Given** several jobs submitted with the same batch identifier, **When** the batch's status is
   queried, **Then** the response reports counts of jobs by state (pending/running/completed/failed/
   cancelled) that match the true state of each underlying job.
2. **Given** a batch with at least one completed job and others still running, **When** the batch's
   status is queried, **Then** it includes a time-remaining estimate derived from real observed
   per-job processing time, not a fixed guess independent of actual performance.
3. **Given** a batch where every job has reached a terminal state, **When** its status is queried,
   **Then** it reports as fully complete with no time estimate implied.

---

### User Story 2 - Retry every failed job in a batch in one action (Priority: P1)

Of 10 submitted videos, 2 fail (e.g. a transient error). Today the researcher would have to find and
retry each one individually. They instead retry the whole batch's failures in one action.

**Why this priority**: Direct extension of spec 006's single-job retry to the group scale this
feature exists to serve — without it, spec 006's retry endpoint still leaves group-scale cleanup
tedious.

**Independent Test**: Submit a batch where some jobs are forced to fail; call batch-retry; confirm
only the failed jobs are retried (using spec 006's retry semantics) and already-succeeded jobs are
untouched.

**Acceptance Scenarios**:

1. **Given** a batch containing both failed and successfully-completed jobs, **When** batch-retry is
   requested, **Then** only the failed jobs are retried; completed jobs are unaffected.
2. **Given** a batch containing a job that is still running (not yet terminal), **When** batch-retry
   is requested, **Then** that job is left alone and the request does not error because of it — only
   jobs actually in a retryable state are retried.
3. **Given** a batch where no job is currently in a retryable state, **When** batch-retry is
   requested, **Then** the system reports that nothing needed retrying, not an error.

---

### User Story 3 - Learn about job status changes without polling (Priority: P2)

A researcher has the jobs list open and watches a batch progress. Today this only updates via polling
(every few seconds at best); the existing server-sent-events stream sends nothing but heartbeats.
Real status-change events let the client learn about progress as it happens.

**Why this priority**: Lower than US1/US2 because polling already provides a (less efficient,
slightly laggier) path to the same information — this is a quality improvement, not a new capability,
so it's appropriately P2.

**Independent Test**: Subscribe to the event stream while a job's status changes (e.g. via
cancellation or completion); confirm a corresponding event is received without a separate poll.

**Acceptance Scenarios**:

1. **Given** an active subscription to the event stream, **When** a job's status changes, **Then** an
   event carrying that job's id, its new status, and its batch identifier (if any) is delivered on
   the stream.
2. **Given** no active event-stream subscription, **When** a client instead polls job/batch status
   endpoints, **Then** it observes the same state as a subscribed client would — the event stream is
   additive, not a required path to correct information.

---

### Edge Cases

- A batch-retry request arrives while some of its jobs are still running: MUST only touch the
  actually-failed/cancelled ones (see US2, Acceptance Scenario 2) — never block or error on the whole
  batch because of in-progress jobs.
- The dataset a batch was submitted from (if any) is later deleted: batch summary MUST continue to
  work from the jobs' own recorded data, independent of the dataset record's continued existence.
- A very large batch (dozens of videos) runs over an extended period: the ETA calculation MUST remain
  meaningful as more jobs complete over time, not just extrapolate from the first one or two results.
- The event-stream connection drops mid-batch (network interruption): a client MUST be able to
  recover the true current state via polling without any state loss — the stream is a convenience,
  not the source of truth.
- Two jobs are submitted with the same batch identifier by two different, unrelated submissions
  (e.g. a reused/collided client-generated id): system MUST NOT need to disambiguate this at the API
  level — a batch is simply "whatever jobs currently carry this identifier," so this is a client-side
  correctness concern (use a real UUID), not something the server needs to detect or reject.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST allow a job submission to carry a client-supplied batch identifier
  grouping it with other jobs submitted under the same identifier.
- **FR-002**: The system MUST provide a way to retrieve aggregate status — counts of jobs by state —
  for all jobs sharing a given batch identifier.
- **FR-003**: The aggregate batch status MUST include a time-remaining estimate, computed from real
  observed per-job processing time, once at least one job in the batch has completed.
- **FR-004**: The system MUST provide a way to retry every job within a batch that is currently in a
  retryable terminal state, in one request, applying spec 006's single-job retry semantics to each.
- **FR-005**: A batch-retry request MUST only affect jobs in the batch that are actually retryable; it
  MUST NOT fail the entire request because other jobs in the batch are non-terminal or already
  succeeded.
- **FR-006**: The system MUST emit real job status/progress change events over the existing
  server-sent-events stream, in addition to (not instead of) its current heartbeat behavior.
- **FR-007**: A job submission MAY optionally carry a dataset identifier (spec 007) alongside its
  batch identifier, recording which saved dataset (if any) it came from.
- **FR-008**: Deleting a dataset (per spec 007's own non-corruption guarantee) MUST NOT invalidate or
  corrupt batch status for jobs previously tagged with it.
- **FR-009**: Single-job cancel/retry/status behavior defined by spec 006 MUST continue to work
  unchanged — batch-level operations are additive.

### Key Entities

- **Batch identifier**: an opaque string, generated by the submitting client (not the server),
  grouping the jobs submitted together under it. Not a separately-created resource — a batch exists
  simply as "the set of jobs currently carrying this identifier."
- **Batch summary**: an aggregate, computed-on-read view over all jobs sharing a batch identifier —
  status counts, overall completion percentage, and (once available) a time-remaining estimate.

## API Contract for Downstream Consumers

- **Submission**: `POST /api/v1/jobs` accepts optional `batch_id` and `dataset_id` form fields
  alongside its existing fields. Omitting `batch_id` behaves exactly as today (a fully standalone
  job).
- **Batch summary**: `GET /api/v1/batches/{batch_id}` → `{ batch_id, total, by_status: { pending,
  running, completed, failed, cancelled }, completion_percentage, estimated_seconds_remaining: number
  | null }`. `estimated_seconds_remaining` is `null` until at least one job in the batch has
  completed.
- **Batch retry**: `POST /api/v1/batches/{batch_id}/retry` → reports how many jobs were retried and
  how many were skipped (with why — e.g. "still running," "already succeeded").
- **Events**: the existing `/api/v1/events/stream` now also emits `job_status_changed` events shaped
  `{ job_id, batch_id: string | null, status, progress_percentage }`, in addition to its current
  heartbeat events (unchanged, still sent).
- **Stability expectation**: matches spec 004/005/007's precedent — once published, these shapes are
  a stable contract for the viewer to build against.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Submitting N videos with one batch identifier produces one queryable summary covering
  all N jobs, verified by test.
- **SC-002**: The batch ETA reflects real observed per-job processing time as jobs complete, verified
  by test or a documented manual run comparing estimate drift against actual completion time.
- **SC-003**: Batch-retry retries only a batch's actually-retryable jobs, leaving others untouched,
  verified by test.
- **SC-004**: A client subscribed to the event stream observes a job status change without polling,
  verified by an integration test.
- **SC-005**: Deleting a dataset referenced by a batch does not affect that batch's summary or job
  records, verified by test.

## Assumptions

- **No new multi-file-upload endpoint.** Video uploads remain one HTTP request per video; only the
  shared batch identifier is new. A true batch-upload-in-one-request endpoint is deferred as
  unnecessary complexity given realistic video file sizes.
- **Batch identifiers are client-generated**, not server-assigned — avoids a create-batch round-trip
  before the first video can be uploaded. A batch becomes queryable the moment its first job exists.
- **Real-time push is additive, not authoritative.** Any client that never connects to the event
  stream must still be able to determine correct current state entirely through polling, exactly as
  today.
- **Viewer-side work is out of scope here** — group progress display, bulk-action buttons, and batch
  review navigation between videos are `video-annotation-viewer`'s own spec; see
  [`viewer-handoff.md`](viewer-handoff.md).
