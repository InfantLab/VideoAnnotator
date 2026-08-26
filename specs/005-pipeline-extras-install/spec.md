# Feature Specification: Pipeline Extras Discoverability & Self-Service Install

**Feature Branch**: `005-pipeline-extras-install`
**Created**: 2026-08-26
**Status**: Draft
**Input**: User description: "In-app pipeline extras discoverability and self-service install. Builds on spec 004 (extras-based install, v1.5.0) which already gives GET /api/v1/pipelines an available:bool and install_hint:str per pipeline, but there is no way to act on that hint without dropping to a terminal and running uv sync --extra <name> plus a manual server restart. Add a self-service install action reachable from the API so a user can go from 'core install' to 'core + the extras group I actually need' without leaving the browser, while keeping the core install exactly as small as spec 004 made it."

## Relationship to Existing Specs

This is a direct follow-on to
[`specs/004-extras-based-install/spec.md`](../004-extras-based-install/spec.md) ("spec 004"), which
shipped the read side of this problem: `GET /api/v1/pipelines?include_unavailable=true` already
returns `available: bool` and `install_hint: str` per pipeline (the exact `pip install
videoannotator[<group>]` command), computed from each pipeline's `requires_extras` metadata field.
What spec 004 did not add is a way to *act* on that hint without leaving the application — today the
only path from "pipeline is unavailable" to "pipeline is available" is a user copying a shell
command, running it in a terminal, and manually restarting the server. This spec adds that missing
write-side action, entirely additive to spec 004's schema and registry design — no existing field,
endpoint, or extras-group name changes shape.

This spec covers the **backend capability only** (this repo, `videoannotator-core`). The consumer of
this capability — locked/lockable pipeline cards, an "Install" action, install progress, and a
"restart required" banner — is a UI concern that belongs to the separate `video-annotation-viewer`
project and requires its own spec written in that project's repository. The
[API Contract](#api-contract-for-downstream-consumers) section below exists specifically so that
spec can be written against a stable contract without needing this repo's implementation.

A related, downstream feature — letting a user save a named preset of dataset selection + pipeline
choices + per-pipeline settings and re-run it — is explicitly **out of scope** here (see
Assumptions). It depends on this feature existing (no point saving a preset that references a
pipeline that isn't installed when replayed) but is its own future spec.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Install a needed pipeline without a terminal (Priority: P1)

A researcher running VideoAnnotator's core (no-extras) install discovers, via the pipeline catalog,
that the face-emotion pipeline they want is unavailable. Instead of copying an `install_hint`
command into a separate terminal session, they trigger the install directly through the API and
watch it complete.

**Why this priority**: This is the entire point of the feature — it's the one gap left after spec
004 that still forces a context-switch out of the application. Without this, spec 004's
`install_hint` is informational only.

**Independent Test**: With only the core install present, call the install-trigger endpoint for the
`face` extras group; poll the returned job until it reports success; confirm the underlying
dependencies are now present in the environment.

**Acceptance Scenarios**:

1. **Given** a core-only install and an authenticated admin caller, **When** they request install of
   a valid, currently-uninstalled extras group, **Then** the system accepts the request and returns
   a trackable job identifier immediately (does not block the request on the install completing).
2. **Given** a running install job, **When** the caller polls its status, **Then** they see one of a
   defined set of states (e.g. pending/running/completed/failed) and, on completion, whether it
   succeeded.
3. **Given** an install job that fails (e.g. network failure, disk full, dependency conflict),
   **When** the caller inspects the job, **Then** they get the captured failure output, not a silent
   or ambiguous failure.
4. **Given** an extras group that is already fully installed, **When** a caller requests installing
   it again, **Then** the system reports it as already satisfied without re-running unnecessary work
   or erroring.

---

### User Story 2 - Know an install needs a restart to take effect (Priority: P1)

After a successful install, the researcher checks whether the pipeline is now usable and is told
clearly that a server restart is required to activate it — rather than the pipeline silently
continuing to appear unavailable with no explanation.

**Why this priority**: Without this, a successful install looks indistinguishable from a failed one
from the caller's point of view — the pipeline still reports `available: false` until the process
restarts, and the API contract must say why, or the whole feature is confusing rather than
frictionless. Equal priority to User Story 1 — an install action without this signal is misleading.

**Independent Test**: Complete an install job for a previously-uninstalled extras group without
restarting the server; confirm the pipeline catalog surfaces a "restart required" signal rather than
just `available: false` with no context, and that the signal clears after the server is restarted
and the pipeline becomes genuinely available.

**Acceptance Scenarios**:

1. **Given** a successfully completed install job, **When** the caller checks that job's status,
   **Then** the response indicates a server restart is required before the newly-installed
   pipeline(s) become usable.
2. **Given** at least one completed install awaiting a restart, **When** the caller lists pipelines
   or checks system status, **Then** a restart-required signal is present and distinguishable from
   "no pending installs."
3. **Given** the server has been restarted after a completed install, **When** the caller lists
   pipelines, **Then** the previously-installed extras group's pipeline(s) report `available: true`
   and the restart-required signal is cleared.

---

### User Story 3 - Install requests are safe by construction (Priority: P1)

An administrator triggers pipeline installs through the application. The system only ever installs
one of the project's own declared, documented extras groups — never an arbitrary string — and only
an authenticated administrator can trigger installs at all.

**Why this priority**: This endpoint executes a package-install subprocess on the server on the
caller's request. Without strict validation and authentication, it is a remote arbitrary-command
surface, which is unacceptable regardless of how convenient the feature is. This is a hard
precondition for shipping User Stories 1 and 2 at all, not an enhancement.

**Independent Test**: Attempt to trigger an install (a) unauthenticated, (b) as a non-admin
authenticated user, and (c) naming an extras group that does not exist in the project's declared
set; confirm all three are rejected before any subprocess runs.

**Acceptance Scenarios**:

1. **Given** an unauthenticated caller, **When** they request an install, **Then** the request is
   rejected and no install runs.
2. **Given** an authenticated caller without administrator privileges, **When** they request an
   install, **Then** the request is rejected and no install runs.
3. **Given** an authenticated administrator, **When** they name an extras group that is not one of
   the project's declared, documented groups, **Then** the request is rejected with a clear error
   and no subprocess is invoked with that value.

---

### Edge Cases

- Two install requests for the same extras group arrive concurrently: the system MUST NOT run two
  overlapping install subprocesses for the same target; the second request either joins the
  in-flight job or is rejected with a clear "already in progress" signal.
- An install is requested for the reserved `all` group: MUST behave the same as any other declared
  group (this spec does not special-case `all`).
- The server process is restarted or crashes while an install job is still running: on next startup,
  that job MUST NOT be reported as silently "running" forever — it must resolve to a terminal state
  (e.g. failed/interrupted) discoverable by the caller.
- An install partially succeeds (some packages installed, then a failure) leaving the environment in
  an inconsistent state: the job MUST report failure (not success), and a retry of the same group
  MUST be possible without manual cleanup.
- A caller requests install of an extras group that is a strict superset of one already installed
  (e.g. `all` when `face` is already present): MUST succeed and result in the full superset being
  available after restart, not a conflict.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST provide an action, reachable via the API, that triggers installation
  of a single named extras group without requiring shell/terminal access.
- **FR-002**: The system MUST validate the requested extras group against the project's actual,
  currently-declared set of installable groups before doing anything else; any other value MUST be
  rejected with a clear error and MUST NOT reach the install mechanism.
- **FR-003**: The install action MUST require administrator authentication; unauthenticated or
  non-administrator callers MUST be rejected before an install is attempted.
- **FR-004**: Triggering an install MUST return promptly with a trackable identifier, not block the
  caller until the (potentially multi-minute, large-download) install finishes.
- **FR-005**: The system MUST expose the status of an install job through a small, defined set of
  states covering at minimum: queued/not yet started, in progress, completed successfully, and
  failed.
- **FR-006**: On failure, the job's status MUST include enough captured detail (e.g. the underlying
  error output) for the caller to diagnose the problem, not just a boolean failure flag.
- **FR-007**: A successfully completed install MUST be distinguishable, via the API, from "no
  restart needed" — i.e. the system MUST expose a restart-required signal once at least one install
  has completed successfully and the running server has not yet been restarted since.
- **FR-008**: The restart-required signal MUST be visible both on the completed install job itself
  and on the general pipeline-listing/system-status surface, so a caller does not have to remember
  which job produced it.
- **FR-009**: After the server process is restarted, pipelines belonging to a successfully-installed
  extras group MUST report as available, and the restart-required signal MUST clear.
- **FR-010**: The system MUST NOT run two concurrent install subprocesses for the same extras group;
  a duplicate request while one is in flight MUST be handled deterministically (join or reject), not
  racily.
- **FR-011**: Requesting install of an extras group that is already fully satisfied MUST be a
  cheap, safe no-op that reports success, not an error.
- **FR-012**: This feature MUST NOT change spec 004's registry/metadata schema, extras-group naming,
  or the existing read-side `available`/`install_hint` fields — it is additive only.
- **FR-013**: This feature MUST NOT cause the default (no-extras) install to pull in any additional
  dependency — the install action only runs when explicitly invoked by an administrator.
- **FR-014 (auto-restart out of scope)**: This feature MUST NOT attempt to restart the server
  process automatically after a successful install. Restart remains a separate, manual action by
  whoever operates the server. (Automatic restart-on-install is a deferred future enhancement — see
  Assumptions.)

### Key Entities

- **Extras install job**: A trackable unit of work representing one in-progress or completed attempt
  to install a named extras group. Has an identifier, the target extras-group name, a status
  (queued/running/completed/failed), captured output/error detail, and whether it leaves the server
  in a restart-required state.
- **Restart-required signal**: A piece of system state, readable through the API, indicating that at
  least one successful install has occurred since the server process last started, and that a
  restart is needed for its pipeline(s) to become available. Clears on restart.

## API Contract for Downstream Consumers

This section is the interface the `video-annotation-viewer` project's own spec should build
against. It intentionally describes shapes and behavior, not this repo's internal implementation.
See [`viewer-handoff.md`](viewer-handoff.md) for the full handoff document (functional
requirements, UX considerations, explicit non-goals) written to be pasted directly into that
repo's own `/speckit-specify`.

- **Trigger install**: an authenticated, admin-only action that accepts one target extras-group name
  and returns a job identifier immediately. Naming an invalid/undeclared group is rejected
  synchronously, before any job is created.
- **Check install job status**: given a job identifier, returns its current state (queued / running
  / completed / failed), and on failure, human-readable diagnostic detail; on success, whether a
  restart is now required.
- **Pipeline listing / system status**: continues to expose spec 004's per-pipeline `available` and
  `install_hint`, and additionally exposes a top-level restart-required signal reflecting whether any
  completed install is awaiting a server restart.
- **Stability expectation**: once published, the shapes of the above (state names, field names) are
  treated as a stable contract for the viewer to build against — changes require the same
  forward-compatibility care spec 004 required of the registry schema.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A user can go from "pipeline unavailable" to "install triggered" without leaving the
  application or opening a terminal, in a single action.
- **SC-002**: 100% of install-trigger requests naming a group outside the project's declared extras
  set are rejected before any subprocess runs (verified by test fixtures covering every invalid
  input class: unknown name, empty string, path-like/shell-metacharacter strings).
- **SC-003**: 100% of install-trigger requests from unauthenticated or non-admin callers are
  rejected, verified by test fixtures.
- **SC-004**: A caller can determine, without ambiguity, whether a completed install requires a
  restart, in 100% of completed-job cases (measured by test fixtures).
- **SC-005**: After a successful install and a subsequent server restart, the corresponding
  pipeline(s) report available in the standard pipeline listing with zero additional manual steps
  beyond the restart itself.
- **SC-006**: The default (no-extras) install's dependency footprint is unchanged by this feature —
  measured the same way as spec 004's SC-002 baseline.
- **SC-007**: Concurrent duplicate install requests for the same extras group never result in two
  simultaneous install subprocesses (verified by a concurrency test).

## Assumptions

- This spec covers the backend (`videoannotator-core`) capability only. The `video-annotation-viewer`
  project's UI for this feature (locked pipeline cards, an Install action, progress display, a
  restart banner) is out of scope here and requires its own spec, written in that project's
  repository, built against the [API Contract](#api-contract-for-downstream-consumers) section
  above.
- **Auto-restart-on-install** (the server automatically restarting itself once an install completes,
  removing the manual-restart step entirely) is explicitly deferred, not part of this spec. It is
  tracked as a future item on the longer-term roadmap
  ([`docs/development/roadmap_v1.7_to_v2.0.md`](../../docs/development/roadmap_v1.7_to_v2.0.md))
  once process-supervision/dispatcher seams from that phase of work exist to do it safely (avoiding
  interrupting in-flight annotation jobs).
- **Saved pipeline configurations / dataset presets** (a user saving a named combination of dataset
  selection + pipeline choices + settings for one-click reuse) is a related, downstream feature that
  depends on this one landing first, but is explicitly out of scope for this spec and will be
  proposed as its own future spec.
- The mechanism used to perform the actual install (this spec deliberately does not name a specific
  tool or command — that is an implementation detail for the planning phase) is assumed to be
  idempotent and safe to invoke against an environment where the target group is already partially
  or fully installed.
- Extras-group names available to install are exactly the set spec 004 established
  (`face`, `face-laion`, `face-openface3`, `audio`, `audio-laion`, `scene`, `person`, `all`, plus any
  added by later specs, e.g. v1.6.0's `llm`); this spec does not add, rename, or remove any group.
