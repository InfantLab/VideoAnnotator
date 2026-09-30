# Quickstart: Pipeline Extras Install UI

## Prerequisites

- A VideoAnnotator server running **v1.5.0 or later** (with spec 005-pipeline-extras-install's endpoints live). Point the viewer at it the usual way — see the repo README's "Connect to VideoAnnotator API" section (`VITE_API_BASE_URL` / `VITE_API_TOKEN`, or the Settings page).
- For the admin-only path: an API token with admin privileges on that server. A non-admin token is enough to exercise the "locked pipeline visible, Install action denied" path.
- Easiest way to get a genuinely core-only server to test against: run VideoAnnotator's core install (no extras group) so most pipelines start out locked.

## Manual verification flow

1. **Locked pipelines are visible** — Open the job-creation wizard, go to Select Pipelines. Confirm pipelines beyond the stub pipeline are listed (not hidden), each showing a locked/disabled visual state with the install hint text.
2. **Install action gating** — With a non-admin (or unauthenticated) session, confirm no Install action is offered, or clicking it produces a clear "administrator privileges required" message — never a silent failure. Switch to an admin token and confirm the Install action appears.
3. **Trigger + track an install** — As admin, click Install on a locked pipeline (pick a smaller extras group like `face` if available, to avoid a multi-minute wait during manual testing). Confirm:
   - The UI immediately shows that pipeline (and only that pipeline / its extras-group siblings) as installing.
   - Status updates roughly every 5 seconds without needing a manual refresh.
4. **Reload survival** — Mid-install, reload the page (or close and reopen the tab) and navigate back to Select Pipelines. Confirm the in-progress state is still shown, not reset to plain "locked."
5. **Completion → restart-required** — Once the job reaches `completed`, confirm the UI does **not** say the pipeline is ready — instead shows a restart-required message framed as an expected next step. Confirm the pipeline itself still shows as locked (matches server's `available: false` until restart).
6. **Failure path** — Trigger an install that will fail (e.g. an extras group name the server rejects, if reachable via a stale catalog, or simulate via a test server) and confirm the failure is shown with the server's `command_output` visible in an expandable section, not a silent reset to "locked."
7. **Post-restart** — Restart the VideoAnnotator server process (outside the app, as today). Reload the viewer / trigger a catalog refresh. Confirm the previously-locked pipeline now shows and behaves exactly like any other available pipeline, with no leftover "installing"/"restart required" residue for that pipeline.

## Automated coverage (see tasks.md once generated)

- `apiClient.getPipelineCatalog({ includeUnavailable: true })`, `installPipelineExtras`, `getExtrasInstallJob` — unit tests against mocked fetch responses (401/403/422/202/200, and a pre-v1.5.0-shaped response missing the new fields).
- `useExtrasInstall` — polling stops on terminal status; `localStorage` persistence/rehydration; 403 surfaced distinctly from other errors.
- `PipelineSelectionStep` / `LockedPipelineCard` — renders locked state, install button visibility by auth state, progress/failure/restart-required rendering, all via Testing Library against a mocked `apiClient`.
