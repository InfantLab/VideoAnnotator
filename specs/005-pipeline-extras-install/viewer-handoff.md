# Handoff to video-annotation-viewer: Extras Install UI

**From**: VideoAnnotator core, `specs/005-pipeline-extras-install/` (v1.5.0 branch)
**Purpose**: This document is written to be pasted as the input to that repo's own
`/speckit-specify` (or used as a standalone requirements brief if that repo doesn't use spec-kit).
It describes a UI feature, not an implementation — the viewer team/spec should make its own
technical decisions about components, state management, and styling.

## Why this exists

VideoAnnotator ships as a slim "core" install by default — no pipeline actually runs until its
extras group (`face`, `audio`, `scene`, `person`, `all`, ...) is installed. Today, when a user
opens the viewer's **Select Pipelines** step with a core-only server, they see almost nothing —
in practice, just a placeholder "stub" pipeline — with no explanation and no path forward short of
finding a terminal, SSH'ing into wherever the server runs, and running a `pip install` command by
hand. That's the actual screen this is fixing:

> Step 2 of 4 — Select Pipelines. One category, "stub", with one pipeline: "Stub
> Forward-Compatibility Pipeline". Everything else — face, audio, scene, person — is simply
> absent from the list, with no indication that it exists or how to get it.

VideoAnnotator's backend now exposes a way to fix this from inside the app: pipelines the server
knows about but hasn't installed can be listed as *locked*, with a one-click install action and a
way to track it to completion. This document specifies what the viewer needs to build on top of
that.

## What the backend now provides

Two pieces, both live in the VideoAnnotator API today:

1. **Discoverability** (already existed, spec 004): `GET /api/v1/pipelines?include_unavailable=true`
   returns every pipeline the server knows about, not just the ones currently runnable. Each entry
   has `available: bool` and, when `false`, `install_hint: string` (e.g.
   `"pip install videoannotator[face]"`).
2. **Self-service install** (new, spec 005): an admin-authenticated action that triggers installing
   one named extras group as a trackable background job, plus a signal for "installed, but the
   server needs a restart to activate it."

Full contract (endpoint shapes, exact fields, status codes) is in this same directory:
[`contracts/extras-install-endpoints.md`](contracts/extras-install-endpoints.md) and
[`contracts/restart-required-signal.md`](contracts/restart-required-signal.md). Summary below is
enough to scope the UI work; read those two files before writing the actual spec.

### Endpoints

- `GET /api/v1/pipelines?include_unavailable=true` → `{ pipelines: [...], restart_required: bool }`.
  Each pipeline: `{ name, display_name, description, available, install_hint, ... }` (existing
  fields unchanged; `restart_required` is the one new top-level field).
- `POST /api/v1/pipelines/extras/{extra}/install` → `202 { job_id, extra_name, status }`. Admin-only
  (`401` unauthenticated, `403` non-admin, `422` for an extras name the server doesn't recognize —
  the error body lists the valid names). Returns immediately; does not wait for the install.
- `GET /api/v1/pipelines/extras/install-jobs/{job_id}` → `{ job_id, extra_name, status,
  created_at, started_at, finished_at, command_output, restart_required }`. `status` is one of
  `pending`, `running`, `completed`, `failed`. `command_output` is populated (and worth surfacing)
  on `failed`.

### Things worth knowing before designing the UI

- **Installs are slow.** A group with no torch dependency (`face`) is the fastest; most others pull
  torch and can take several minutes on a real network (empirically ~8 minutes for `scene` in this
  project's own dev container, dominated by CUDA wheel downloads). The UI needs to represent
  "installing, this will take a while" honestly — a spinner that implies seconds will read as
  broken.
- **A completed install still isn't usable until the server restarts.** This is the single most
  important UX point: after `status` reaches `completed`, the pipeline will *still* show
  `available: false` in the pipeline list until the server process restarts. `restart_required:
  true` is the signal that tells you why. Don't let the UI imply "done, go use it" the moment the
  job completes — that will read as a bug to the user; restarting the server is currently a manual
  step outside the viewer's control (auto-restart is explicitly deferred backend work, tracked on
  VideoAnnotator's own roadmap, not assumed here).
- **Auth**: this requires an *admin* API key specifically, not just any authenticated user. If the
  viewer's current session isn't admin-privileged, the install action shouldn't be offered at all
  (or should fail clearly with the `403` case handled), rather than silently doing nothing.
- **Job identity persists across a page reload**: `job_id` is a normal server-side record: the
  viewer can poll it after a refresh/reconnect without losing progress, as long as it persisted the
  `job_id` somewhere (e.g. the URL, or its own local state) before the reload.

## What the viewer needs to build

Functional requirements, phrased as user-facing behavior — the actual component/state design is
that repo's own call.

1. **Locked pipelines are visible, not hidden.** The Select Pipelines step (and anywhere else
   pipelines are listed/chosen) should call `GET /api/v1/pipelines?include_unavailable=true` and
   render *every* returned pipeline, not just the available ones. An unavailable pipeline is shown
   in a clearly locked/disabled visual state (greyed out, a lock icon, whatever fits the existing
   design system) rather than omitted — the whole point is that a user with only the stub pipeline
   installed should be able to see that face/audio/scene/person exist and are one click away, not
   discover them by reading external docs.
2. **An Install action on each locked pipeline**, visible only to an admin-privileged session.
   Clicking it calls the trigger endpoint for that pipeline's extras group and begins tracking the
   returned job.
3. **Progress feedback for an in-flight install.** While a job is `pending`/`running`, the locked
   pipeline (or a dedicated install-progress area — designer's call) shows that an install is
   underway, ideally per-pipeline so unrelated pipelines aren't implied to be installing too. Poll
   the job-status endpoint at a reasonable interval (a few seconds; this is a multi-minute
   operation, sub-second polling is wasted traffic).
4. **Failure is visible and actionable.** On `status: failed`, show that it failed and surface
   `command_output` (at least a truncated/expandable view) so the user — or whoever they escalate
   to — has something to go on, rather than a silent reset back to "locked."
5. **A restart-required banner**, app-wide or job-scoped (designer's call), driven by the
   `restart_required` field. Once true, tell the user a server restart is needed before the
   newly-installed pipeline(s) become usable — this is a real, expected step in the current
   workflow, not an error state, so word it accordingly ("Restart the VideoAnnotator server to
   finish activating new pipelines" reads very differently from "Error: pipeline still
   unavailable").
6. **Post-restart, the pipeline behaves normally.** Once the flag clears and `available` flips to
   `true` for that pipeline (the viewer will observe this on its next `GET /api/v1/pipelines` call
   — there's no push notification for "the server just restarted"), it should look and behave
   exactly like any other pipeline that was available from the start. No special-casing needed here
   beyond the existing available-pipeline rendering path.

## Explicit non-goals for this piece of work

- **Auto-restart.** The viewer should not attempt to restart the server itself or imply it can.
  That's out of scope on the backend too right now (see spec 005's Assumptions section) — deferred
  to VideoAnnotator's longer-term roadmap once safer process-supervision exists.
- **Cancelling an in-progress install.** Not supported by the backend; don't build a cancel button
  that has nothing to call.
- **Saved pipeline configurations / dataset presets.** A related, separately-requested piece of
  future work (letting a user save "this dataset + these pipelines + these settings" as a reusable
  preset) is intentionally not part of this handoff. It depends on this install/discoverability
  work existing first, but should be its own spec, not folded in here.
- **Choosing the install mechanism, retry policy, or anything about how the server actually runs
  `pip`/`uv`.** That's all backend-internal; the viewer only ever talks to the two endpoints above.

## Suggested next step

Paste the "What the viewer needs to build" section (and the endpoint summary) into that repo's own
`/speckit-specify` to produce a proper spec scoped to that codebase's actual components and
conventions — this document intentionally stops at *what*, not *how*, so that repo's spec can make
its own calls about component structure, polling implementation, and visual design.
