# Feature Specification: Job-Execution Path Consolidation & Real Cancel/Retry

**Feature Branch**: `006-job-execution-consolidation`
**Created**: 2026-08-26
**Status**: Draft
**Input**: User description: "Consolidate the three divergent job-execution code paths into one, and make cancel and retry actually work. Prerequisite for the workflow-upgrades effort (datasets, saved presets, batch/group tracking, VLM tooling) — group-level progress/cancel/retry need one reliable execution path to build on, not three."

## Relationship to Existing Specs & Roadmap

This is the pulled-forward execution of the consolidation already committed to in
[`docs/development/roadmap_v1.6.0.md`](../../docs/development/roadmap_v1.6.0.md) Phase 1: *"Consolidate
the duplicate job-execution paths (`api/job_processor.py` and
`batch/batch_orchestrator._process_single_job`) into one function that both the CLI and API call."*
It's being done now, ahead of the rest of v1.6.0, because it's a hard prerequisite for the
workflow-upgrades effort's group progress/cancel/retry features — those need one execution path with
working cancellation and status tracking to attach to, not three divergent ones.

Today there are three separate implementations of "run a job's selected pipelines":

1. `api/background_tasks.py`'s `BackgroundJobManager` → `api/job_processor.py`'s `JobProcessor` — the
   path that actually runs by default (auto-started when `videoannotator server` boots). Has **zero**
   cancellation checks anywhere in its path.
2. `worker/job_processor.py`'s `JobProcessor` → `batch/batch_orchestrator.py`'s `BatchOrchestrator` —
   only reachable via the separate `videoannotator worker` CLI process, which the project's own docs
   (`docs/development/vlm_annotation_pipeline.md`) tell users not to run alongside the server. This is
   the *only* path with real cancellation-checkpoint and auto-retry logic today — logic nobody's
   default deployment ever executes.
3. `batch/batch_orchestrator.py` used directly (e.g. by a script or future CLI batch command) —
   already correctly passes per-pipeline config, unlike path 1 (a real divergence bug already found
   and fixed once this session, see `CHANGELOG.md`'s "Fixed" entry for `api/job_processor.py`).

This spec does not touch pipeline internals (`process()` contracts, output formats) — only how a
job's pipelines are dispatched, tracked, cancelled, and retried.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Cancelling a running job actually stops it (Priority: P1)

A researcher submits a job running several pipelines against a video, realizes partway through that
they picked the wrong config, and cancels it. Today, cancelling only flips a database flag — the job
keeps running every pipeline to completion regardless, and the in-flight work silently overwrites the
cancelled status back to completed or failed once it finishes.

**Why this priority**: This is a confirmed, live bug, not a hypothetical — cancel is currently
cosmetic on the path the server actually runs by default. Without a real fix, every downstream
group-cancel feature would inherit the same non-functional behavior.

**Independent Test**: Submit a job with two or more pipelines, request cancellation after the first
pipeline has started but before the second would start, and confirm the second pipeline never runs
and the job settles in a cancelled state — not completed or failed.

**Acceptance Scenarios**:

1. **Given** a running job with multiple selected pipelines, **When** cancellation is requested while
   the first pipeline is still executing, **Then** the system does not start the next pipeline in the
   job's list once the current one finishes.
2. **Given** a job that has been cancelled between pipelines, **When** the currently-executing
   pipeline finishes its own work, **Then** the job's final status is cancelled, not completed or
   failed — the in-flight pipeline's completion does not overwrite the cancellation.
3. **Given** a job that has already reached a terminal state (completed/failed/cancelled), **When** a
   cancellation is requested for it, **Then** the request is a safe no-op and does not change the
   job's terminal state.

---

### User Story 2 - Retrying a failed job doesn't require re-uploading the video (Priority: P1)

A job fails (e.g. a transient model-server error, or a config mistake now corrected). Today there is
no retry endpoint at all — the only way to "retry" is submitting an entirely new job, which means
re-selecting and re-uploading the same video file from the browser, even though the server already
has it in storage from the original submission.

**Why this priority**: Equal priority to User Story 1 — both are the two concrete, user-facing
capability gaps this spec exists to close. Re-uploading a video that's already on the server (often
minutes of upload time for real research footage) is pure friction with no value.

**Independent Test**: Submit a job with a configuration guaranteed to fail (e.g. an unavailable
pipeline), confirm it reaches a failed state, then call the retry action and confirm it reprocesses
using the already-stored video file and original configuration with zero new file upload.

**Acceptance Scenarios**:

1. **Given** a job in a failed or cancelled terminal state, **When** retry is requested, **Then** the
   system reprocesses it using the original stored video file and original configuration, without the
   caller supplying either again.
2. **Given** a job that is still pending or running (not yet terminal), **When** retry is requested,
   **Then** the request is rejected with a clear error explaining the job isn't retryable yet.
3. **Given** a job whose original video file is no longer present in storage (e.g. removed by a
   storage-cleanup pass), **When** retry is requested, **Then** the request fails with a clear error
   rather than silently producing an empty or corrupt result.

---

### User Story 3 - One execution path, not three (Priority: P2)

A developer adds a new capability to how jobs run (a progress update, a new cancellation check, a
config-handling fix). Today they'd have to find and change it in up to three separate
implementations, with no guarantee the changes stay in sync — exactly how the per-pipeline
config-passing bug happened once already this session.

**Why this priority**: Lower priority than US1/US2 because it's not independently user-visible, but
it's the structural fix that makes US1 and US2 sustainable rather than a fourth divergent
implementation. Without it, this spec would just be adding a fourth code path instead of fixing the
underlying problem.

**Independent Test**: Inspect the codebase and confirm exactly one function/module implements "run
this job's selected pipelines," called by both the API's background job processor and the CLI worker
path — not two or three separate implementations of the same responsibility.

**Acceptance Scenarios**:

1. **Given** the consolidated codebase, **When** a job is submitted via the API and picked up by the
   default background processor, **Then** it is executed by the same underlying function used by the
   CLI worker path.
2. **Given** identical per-pipeline configuration submitted with a job, **When** that job is executed,
   **Then** the effective configuration applied to each pipeline is identical regardless of which
   internal component (API background processor vs. CLI worker) picked the job up.

---

### Edge Cases

- Cancellation requested for a job at the exact moment its currently-running pipeline finishes and the
  next one is about to start: the checkpoint MUST resolve unambiguously to "stopped" — the next
  pipeline MUST NOT start once cancellation has been recorded, even if the timing is tight.
- Retry requested for a job that both failed AND has since had its output directory partially cleaned
  up: MUST fail clearly rather than proceed with a partially-missing environment.
- The server process stops (crash, restart, deliberate shutdown) while a job is pending or running:
  on next startup, that job MUST NOT be reported as running forever — it MUST resolve to a discoverable
  terminal state.
- Two cancellation requests for the same job arrive concurrently: MUST be idempotent — the job ends up
  cancelled exactly once, no error from the second request.
- A job is cancelled, then immediately retried: MUST behave identically to retrying a job that failed
  naturally — no special-casing of "cancelled-then-retried" vs. "failed-then-retried."

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST execute a submitted job's selected pipelines through exactly one
  internal code path, regardless of whether the job was picked up by the API's default background
  processor or the CLI worker process.
- **FR-002**: Before starting each pipeline in a job's selected-pipelines list, the system MUST check
  whether the job has been marked for cancellation, and MUST NOT start further pipelines if so.
- **FR-003**: A cancelled job MUST settle in a cancelled terminal state and MUST NOT be overwritten
  back to completed or failed by pipeline work that was already in flight when cancellation was
  requested.
- **FR-004**: The system MUST provide an action, reachable via the API, that retries a job in a
  failed or cancelled terminal state, reusing its originally stored video file and configuration
  without the caller re-supplying either.
- **FR-005**: Retry MUST be rejected, with a clear error, for a job that is not in a terminal
  failed/cancelled state.
- **FR-006**: Retry MUST be rejected, with a clear error, if the job's original video file is no
  longer available in storage.
- **FR-007**: Per-pipeline configuration submitted with a job MUST be applied identically regardless
  of which internal path executes that job.
- **FR-008**: If the server process stops while a job is in a non-terminal state, the system MUST
  resolve that job to a discoverable terminal state on the next startup rather than leaving it
  reported as running indefinitely.
- **FR-009**: This feature MUST NOT change any pipeline's `process()` contract, on-disk output format,
  or storage layout — only how a job's pipelines are dispatched, tracked, cancelled, and retried.
- **FR-010**: The consolidated execution path MUST update the job's progress field as each pipeline in
  its selected-pipelines list completes, replacing today's hardcoded stub value.

### Key Entities

- **Job execution path**: the single function/module responsible for running one job's selected
  pipelines end-to-end (initialize each pipeline, call `process()`, persist annotations, advance
  status/progress) — replaces the three current divergent implementations.
- **Cancellation checkpoint**: the point, between each pipeline in a job's selected-pipelines list,
  where the execution path checks the job's current status before proceeding to the next one.
- **Retry action**: an API-triggered request to re-run a terminal (failed/cancelled) job using its
  already-stored video and configuration. Whether this reuses the same job record or creates a new one
  referencing the same inputs is a planning-phase decision, not fixed by this spec.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a job with two or more pipelines, cancelling after the first pipeline starts prevents
  the second from ever running, verified by test.
- **SC-002**: A failed job can be retried via a single API call with zero additional file uploads,
  verified by test.
- **SC-003**: Exactly one function/module implements pipeline dispatch for a job, used by both the
  API's default background processor and the CLI worker path — verified by code inspection and a test
  asserting both paths invoke the same underlying function.
- **SC-004**: Identical per-pipeline config submitted through either path produces identical effective
  pipeline configuration, verified by test.
- **SC-005**: A job left in a non-terminal state when the server process stops resolves to a terminal
  state within one restart cycle, verified by test or a documented manual restart scenario.
- **SC-006**: The job progress field reflects real completed/total pipeline counts — not a hardcoded
  stub — for at least one multi-pipeline job, verified by test.

## Assumptions

- **True mid-pipeline interruption is out of scope.** Cancellation is checkpointed *between*
  pipelines, not signal-based interruption of an in-flight pipeline call (e.g. aborting a single VLM
  inference request already in progress). This is a documented, honest limit, not a defect — worth
  stating explicitly so downstream group-cancel UX doesn't imply instant interrupt.
- **Batch/group identity (`batch_id`/`dataset_id` on `Job`) is not part of this spec** — that belongs
  to the workflow-upgrades effort's Spec 2 (Batch/Group Workflow), which depends on this spec landing
  first so its group-cancel/retry/progress features have a reliable single execution path to build on.
- **Real-time push notification of status changes** (replacing the current SSE heartbeat-only stream)
  is out of scope here — also Spec 2's job, which depends on this spec's real progress/status writes
  existing to have something meaningful to push.
- **The `worker/job_processor.py` / `videoannotator worker` CLI path** may be simplified, or reduced to
  a thin wrapper over the consolidated execution path, as part of this work — whether to keep it as a
  separate CLI entry point at all is a planning-phase decision, not fixed here.
- **Single-process deployment is assumed.** The DB-status-polled cancellation checkpoint design does
  not address multi-process/multi-worker horizontal scaling — consistent with the single-process
  model documented in `docs/development/vlm_annotation_pipeline.md`'s "Known gaps" section.
