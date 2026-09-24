# Handoff to video-annotation-viewer: Pipeline Readiness UI

**From**: VideoAnnotator, `specs/011-pipeline-readiness/` (v1.5.0 branch)
**Purpose**: Paste this into the viewer repo's `/speckit-specify` as its input. It says *what* the
user must be able to do; component structure, state management and visual design are the viewer's
call. Exact API shapes live in [contracts/readiness-contract.md](contracts/readiness-contract.md) —
copy that file into the viewer spec's folder and build against it, not against this summary.

**Supersedes** the [005 viewer handoff](../005-pipeline-extras-install/viewer-handoff.md), whose UI
*is* built in the viewer (`specs/002-pipeline-extras-install`): `LockedPipelineCard` +
`ExtrasInstallStatus`, `useExtrasInstall` (polling, reload-safe tracking, and since `bbb5b68` a
catalog refresh on completion), `useCurrentUser` (`/auth/me`), and `RestartRequiredBanner`. Extend
those; don't start over. Its "no auto-restart" non-goal is reversed.

## The problem, as the user sees it

With the current bundle, **Create Job → Step 2 Select Pipelines** lists every pipeline, and an admin
can install one. But then they hit "Server restart needed. Restart the server, then refresh this
page", which needs a terminal. So does a Hugging Face token for diarization, and a failing Ollama
only shows up mid-job. The goal is that **none of that needs a terminal**.

## The one acceptance test that defines "done"

Run against a core-only server with an admin key (the first user created by `generate-token` is
admin by default):

1. Open Create Job → Select Pipelines. Every pipeline is listed; face/audio/etc. are shown as *Not
   installed* with a download size. No stub.
2. Click **Install** on Face Analysis. See honest progress (installs take minutes, not seconds).
3. When it finishes, Face Analysis is selectable, either immediately or after one click on
   **Restart server** and a short wait handled by the viewer.
4. Select it, finish the wizard, and the job completes.
5. Repeat for Speaker Diarization on a server with `HF_AUTH_TOKEN` in its container env. Without
   it, the card says what to set; with it, it's Ready with a reminder to accept the model licence.

No terminal at any point. Both repos' work is done only when this passes against the viewer build
bundled into VideoAnnotator's `viewer_static/`.

## What the viewer needs to build

1. **List everything, with its state.** Wherever pipelines are chosen, call
   `GET /api/v1/pipelines?include_unavailable=true` and render every pipeline. Drive the card from
   `readiness.state`:
   - `ready`: selectable, as today. Show any `notes` quietly (e.g. "First run downloads ~1.2 GB of
     model weights", "Accept the licence on Hugging Face").
   - `not_installed`: not selectable. Show the extras group, approx size from
     `GET /api/v1/pipelines/extras`, and an **Install** action.
   - `installing`: not selectable; progress indicator; survives page reload (look up the job from
     `readiness.install_job_id`).
   - `restart_required`: not selectable; "Installed, restart to activate", plus the **Restart
     server** action.
   - `needs_setup`: not selectable; list each blocker's `message` with a **Set up** action (see 4).
   - Unknown state: not selectable; show the blockers' messages. Don't crash.

   Pipelines that share an extras group (e.g. the three audio pipelines) install together; make
   that obvious so installing one doesn't look like it did something to the others.

2. **Install**: 005's behaviour. `POST /api/v1/pipelines/extras/{extra}/install` then poll
   `GET /api/v1/pipelines/extras/install-jobs/{job_id}` every few seconds. On `failed`, show
   `command_output` (expandable). On `completed`, re-fetch the pipeline list. If
   `activation == "live"`, the pipeline just becomes Ready (or Needs setup) with no extra step.

3. **Restart server**, offered when any pipeline is `restart_required` (a single app-wide banner
   is fine; top-level `restart_required` on the pipeline list tells you). On click:
   `POST /api/v1/system/restart`. On `202`, show a blocking "Restarting server…" state, then poll a
   health endpoint every 2 s (connection errors are expected here) until `boot_id` changes, then
   re-fetch everything. After 120 s, stop and show the manual-restart hint. Handle the `409`s:
   - `JOBS_RUNNING`: "N jobs are running and would be interrupted." Offer *Restart anyway*
     (`force=true`) or *Wait*.
   - `INSTALL_IN_PROGRESS`: "Wait for the install to finish." No override.
   - `RESTART_UNSUPPORTED`: show `hint` verbatim. The server can't restart itself in this
     deployment.

4. **Set up**: act on each blocker by `kind`:
   - `secret`: show the `message` (it says which variable to set in the server's environment) and
     `help_url` as "Get a token". No input: secrets are container configuration, set by whoever
     runs the server (decision 2026-09-24), and the viewer never handles their values.
   - `service` with `name == "ollama"`: link to the existing Ollama status/model UI (spec 009).
   - `import_error`: show the message and "The server administrator needs to look at this". No
     action.

5. **Admin gating** (carried over from 005 requirement 7): call `GET /api/v1/auth/me` once per
   session. Install and Restart are admin-only. For non-admins, show the states
   but replace the actions with "Requires an administrator API key (see Settings)". Show `is_admin`
   in Settings.

6. **Backward compatibility with older servers**: if pipeline entries have no `readiness` field,
   fall back to 005 behaviour (`available`, `install_hint`, top-level `restart_required`, and the
   manual-restart wording). If `/api/v1/pipelines/extras` or `/system/restart` returns `404`,
   hide sizes and the Restart action.

## Non-goals

- Cancelling an install (the backend doesn't support it).
- Editing server settings, environment variables or secrets. Setup blockers are shown, not fixed.
- Pulling Ollama models (already covered by spec 009's UI, if present).
- Weight prefetch (`POST /api/v1/pipelines/{name}/prefetch`) is P3 on the backend. Build it only if
  the endpoint exists (`404` means hide it).

## Suggested order

1. Readiness-driven cards plus the old-server fallback (the existing `available`-based rendering
   *is* the fallback).
2. Install with progress.
3. Restart with the wait.
4. Setup blockers (display only).
5. Rebuild and hand the bundle back for `viewer_static/`.

After step 3, face/scene/person/audio (minus diarization) meet the acceptance test.
