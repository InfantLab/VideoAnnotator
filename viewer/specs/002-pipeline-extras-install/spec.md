# Feature Specification: Pipeline Extras Install UI

**Feature Branch**: `feature/vlm-annotation-support` (continuing on current branch per explicit request; not a dedicated `002-*` branch)
**Created**: 2026-08-26
**Status**: Draft
**Input**: User description: "Extras Install UI: Let users install missing VideoAnnotator pipeline extras (face, audio, scene, person, etc.) directly from the viewer instead of needing shell access." Full handoff brief from VideoAnnotator core team (spec 005-pipeline-extras-install, backend v1.5.0) describing two new endpoints — pipeline discoverability with `available`/`install_hint`, and an admin-only self-service extras-install job — and the UX constraints around them (slow installs, restart-required-before-usable, admin-only action).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - See what pipelines exist, even if not installed (Priority: P1)

A researcher opens the Select Pipelines step of the job creation wizard on a VideoAnnotator server that only has the core package installed. Today they see almost nothing (just a placeholder "stub" pipeline) with no indication that face, audio, scene, and person pipelines exist at all. They need to see the full catalog of pipelines the server knows about, with the ones that aren't installed yet clearly marked as unavailable, so they understand what's possible and don't conclude the tool is broken or limited.

**Why this priority**: Without this, nothing else in this feature is reachable — it's the foundation. It also fixes the most confusing part of the current experience (silent absence).

**Independent Test**: Point the viewer at a core-only server, open Select Pipelines, and confirm every known pipeline (not just installed ones) appears, with locked ones visually distinguished from available ones.

**Acceptance Scenarios**:

1. **Given** a server with only the stub pipeline installed, **When** a user opens Select Pipelines, **Then** they see all server-known pipelines (stub, face, audio, scene, person, etc.), with uninstalled ones shown in a locked/disabled visual state rather than omitted.
2. **Given** a locked pipeline is shown, **When** the user inspects it, **Then** they can see why it's unavailable (e.g. "not installed") without needing to consult external documentation.
3. **Given** a server where every pipeline is already available, **When** a user opens Select Pipelines, **Then** the experience is unchanged from today — no locked state appears.

---

### User Story 2 - Install a missing pipeline from the app (Priority: P1)

An admin-authenticated user sees a locked pipeline they want to use (e.g. "face"). Instead of finding a terminal and SSHing into the server, they click an Install action right on the pipeline, and the viewer kicks off and tracks the install for them.

**Why this priority**: This is the core value proposition of the feature — turning a shell-access chore into an in-app click. Without it, User Story 1 is just a more informative dead end.

**Independent Test**: As an admin-authenticated user on a core-only server, click Install on a locked pipeline and confirm a background job starts and its status is observable in the UI without leaving the page.

**Acceptance Scenarios**:

1. **Given** an admin-authenticated session, **When** the user clicks Install on a locked pipeline, **Then** an install job is triggered and the UI immediately reflects that an install is now in progress for that specific pipeline (not implying other pipelines are installing too).
2. **Given** a non-admin or unauthenticated session, **When** the user views a locked pipeline, **Then** no Install action is offered, or if attempted, the user is clearly told administrator privileges are required — never a silent no-op.
3. **Given** an install is already in progress for a pipeline, **When** the user views that pipeline again (including after a page reload), **Then** the in-progress state is still shown, not reset to plain "locked."

---

### User Story 3 - Understand install progress, failure, and the restart step (Priority: P2)

While an install runs (which can take several minutes), the user needs to know it's genuinely still working, not stuck. If it fails, they need enough detail to act on or escalate. If it succeeds, they need to understand that one more step — a server restart — stands between them and actually using the pipeline, framed as a normal part of the process rather than an error.

**Why this priority**: Builds trust in the feature from User Story 2. Without honest progress/failure/restart messaging, a slow or failed install reads as a broken feature, and a completed-but-not-yet-restarted pipeline reads as a bug.

**Independent Test**: Simulate each of the job statuses (`pending`, `running`, `completed` with `restart_required: true`, `failed`) against the UI and confirm each renders distinct, accurate, non-alarming (for the expected cases) messaging.

**Acceptance Scenarios**:

1. **Given** a job in `pending` or `running` status, **When** the user views the pipeline, **Then** they see an indication that installation is underway and that it may take several minutes (not a spinner that implies seconds).
2. **Given** a job reaches `failed` status, **When** the user views the pipeline, **Then** they see a clear failure indication and can view the command output (at least in a truncated/expandable form) rather than being silently reset to "locked."
3. **Given** a job reaches `completed` status and the server reports `restart_required: true`, **When** the user views the app, **Then** they see messaging that a server restart is needed to finish activating the new pipeline(s), worded as an expected next step, not an error.
4. **Given** `restart_required` is true, **When** the server is restarted (outside the app) and the user's next pipeline list refresh reflects `available: true` for that pipeline, **Then** the restart-required messaging clears and the pipeline behaves exactly like any other available pipeline, with no leftover locked/installing state.

---

### Edge Cases

- User closes the job-creation wizard or navigates away mid-install, then returns later (or reloads the page): the in-progress install must still be discoverable and resume showing progress, not silently disappear.
- Two different pipelines share the same extras group (e.g. both need `face`): installing one should be reflected as installing (or already-installed) for the other too, rather than allowing a duplicate install to be triggered.
- User attempts to install a pipeline whose extras group the server doesn't recognize (a stale/mismatched catalog): the failure must be shown clearly, not treated as a generic network error.
- User's session loses admin privileges (e.g. token rotated) between seeing the Install action and clicking it: the resulting `403` must be shown clearly, not silently swallowed.
- Server never restarts (operator forgets or defers): the restart-required messaging persists indefinitely without becoming alarming or nagging — it's a steady informational state, not a repeating error.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The pipeline selection experience MUST show every pipeline the server knows about, including ones not currently installed, instead of silently omitting them.
- **FR-002**: Each uninstalled pipeline MUST be visually distinguished as locked/unavailable, with a human-readable reason (derived from the server's install hint) rather than being indistinguishable from a normal choice or missing entirely.
- **FR-003**: An admin-authenticated user MUST be able to trigger an install for a locked pipeline's extras group directly from where that pipeline is shown, without leaving the app.
- **FR-004**: The install action MUST NOT be offered to, or MUST fail clearly and legibly for, a session that isn't admin-authenticated — never a silent no-op.
- **FR-005**: Once an install is triggered, the system MUST track its progress and reflect current status (queued/in-progress, failed, completed) specifically for that pipeline, without implying unrelated pipelines are also installing.
- **FR-006**: In-progress and failed install state MUST be recoverable after a page reload or navigation away and back, for as long as the underlying job is still trackable on the server.
- **FR-007**: On install failure, the system MUST surface the server-provided failure detail (command output) in at least a truncated/expandable form, not just "failed."
- **FR-008**: On install completion, if the server indicates a restart is required before the pipeline becomes usable, the system MUST communicate that clearly as an expected next step (not an error, not "done, go use it").
- **FR-009**: The system MUST periodically re-check install job status while a job is in progress, at an interval appropriate for a multi-minute operation (not sub-second polling).
- **FR-010**: Once a previously-locked pipeline becomes available (server restart completed), the system MUST treat it identically to any other available pipeline on the next catalog refresh, with no leftover locked/installing visual state.
- **FR-011**: The system MUST NOT offer any control implying the app can restart the server itself, cancel an in-progress install, or retry with alternate install mechanisms — none of these are supported by the server today.

### Key Entities

- **Pipeline (catalog entry)**: A named annotation capability the server knows about. Has a display name/description, an availability flag, and — when unavailable — a human-readable install hint identifying which extras group provides it.
- **Extras Install Job**: A trackable background operation representing "install this extras group." Has an identity that persists across a page reload, a lifecycle status (queued → in progress → finished successfully or failed), and, on failure, diagnostic output. Associated with exactly one extras group, which may unlock more than one pipeline.
- **Restart-Required Signal**: A system-wide flag indicating that at least one successfully-installed extras group is not yet active because the server process hasn't restarted since the install completed.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a core-only server, a user can go from "opens Select Pipelines" to "sees that face/audio/scene/person exist and how to get them" without leaving the app or consulting external documentation.
- **SC-002**: An admin user can go from "locked pipeline" to "install triggered and visibly tracked" in a single action (one click), with no manual server access required.
- **SC-003**: At any point during a multi-minute install, a user checking the page can correctly state whether it's still in progress, failed, or done-pending-restart — measured by the UI never showing a state that contradicts the actual job status within one polling interval.
- **SC-004**: Zero users mistake a completed-but-not-restarted install for either "ready to use" or "broken" — the restart-required state is read correctly on first encounter in usability review.
- **SC-005**: A user who reloads the page mid-install does not lose visibility into that install's progress.

## Assumptions

- **Admin detection has no dedicated check**: the viewer has no existing concept of "is this session admin" (no role/scope field anywhere in current auth state). Rather than requiring a new capability-check endpoint, the Install action is offered whenever a session is authenticated at all, and a `403` from the install endpoint is treated as the definitive "not admin" signal, shown as a clear message. This matches the backend brief's explicit allowance for either approach.
- **Polling interval**: "a few seconds" from the backend brief is treated as a default of 5 seconds while a job is `pending`/`running`, stopping once it reaches a terminal status (`completed`/`failed`).
- **Job persistence mechanism**: `job_id` (and which pipeline/extras group it belongs to) is persisted in browser local storage rather than the URL, since install-triggering can happen from a multi-step wizard step where URL-based state would be awkward. This satisfies the reload-survival requirement without dictating a specific storage key/schema (implementation detail).
- **Scope of "everywhere pipelines are listed"**: today this is exactly one surface — the Select Pipelines step of the job-creation wizard (`src/pages/NewJob.tsx`). The Settings page's pipeline catalog view is a diagnostics display, not a selection surface, and is not required to gain the same locked/install treatment, though it MAY also show the restart-required signal since it already surfaces server diagnostics.
- **Restart-required banner placement**: shown at minimum wherever pipelines are selected; a global/app-wide placement is allowed but not required by this spec.
- **This spec covers only the viewer side.** The backend endpoints described in the brief are assumed already live per VideoAnnotator v1.5.0; this feature does not modify or depend on backend implementation details beyond the documented contract (endpoint shapes, status codes, field names).
- Out of scope (per explicit non-goals in the handoff): auto-restarting the server, cancelling an in-progress install, saved pipeline/dataset presets, and any opinion on the server's install mechanism.
