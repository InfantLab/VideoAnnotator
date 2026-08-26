# Contract: Viewer ↔ VideoAnnotator Extras-Install Endpoints

This documents the viewer's consumption contract for the four server endpoints this feature depends on (three from the original handoff, plus `GET /api/v1/auth/me` added by the 2026-08-26 addendum). It is derived from the VideoAnnotator core team's handoff brief (spec 005-pipeline-extras-install, backend v1.5.0/v1.5.1); the authoritative backend contracts (`contracts/extras-install-endpoints.md`, `contracts/restart-required-signal.md`) live in the VideoAnnotator repo and are not duplicated here beyond what the viewer needs to code against. If the two ever disagree, the VideoAnnotator repo's contracts win — this file should be updated to match, not the other way around.

The viewer treats all four endpoints as **optional capabilities**: a pre-v1.5.0 (or, for `/auth/me`, pre-v1.5.1) server that 404s any of them, or returns responses without the new fields, must not break existing pipeline-selection/job-creation behavior (Constitution Principle II).

## 1. `GET /api/v1/pipelines?include_unavailable=true`

**Client method**: `apiClient.getPipelineCatalog({ includeUnavailable: true, forceRefresh? })`

**Request**: No body. Query param `include_unavailable=true` appended to the existing call.

**Response** (200): Superset of today's response shape.
```json
{
  "pipelines": [
    {
      "...existing fields unchanged...": "...",
      "available": true,
      "install_hint": null
    },
    {
      "...existing fields unchanged...": "...",
      "available": false,
      "install_hint": "pip install videoannotator[face]"
    }
  ],
  "restart_required": false
}
```

**Viewer handling**:
- Map `available` → `PipelineDescriptor.available` (missing ⇒ treated as `true`).
- Map `install_hint` → `PipelineDescriptor.installHint`, and derive `extraName` from it client-side (see data-model.md).
- Map top-level `restart_required` → `PipelineCatalogResponse.restartRequired` (missing ⇒ `false`).
- No change to existing field mapping (`mapLegacyPipelineResponse` continues to run as-is; the two new fields are added alongside it, not a replacement).

## 2. `POST /api/v1/pipelines/extras/{extra}/install`

**Client method**: `apiClient.installPipelineExtras(extraName: string)`

**Request**: `{extra}` is the bracketed group name (e.g. `face`), URL-encoded. Requires the current session's Authorization header (same token mechanism `apiClient` already attaches everywhere else) — no separate admin-token concept in the viewer.

**Responses**:
| Status | Body | Viewer behavior |
|---|---|---|
| `202` | `{ job_id, extra_name, status }` | Store `{ jobId: job_id, startedAt: now }` under `extraName` in the `localStorage` job map; begin polling endpoint 3 immediately. |
| `401` | error body | Unauthenticated — surface via the existing `parseApiError`/`ErrorDisplay` pattern used elsewhere in `NewJob.tsx`; do not retry automatically. |
| `403` | error body | Authenticated but not admin — surface a specific, non-alarming message ("Administrator privileges are required to install pipeline extras.") distinct from a generic error, per spec FR-004. |
| `422` | error body listing valid extras names | Should not occur in normal operation (the viewer only ever sends `extraName` values derived from the server's own catalog response) — treated as an unexpected-error fallback via `parseApiError`, not specially handled, since it indicates a catalog/install-endpoint mismatch outside the viewer's control. |

**Idempotency note**: The viewer does not attempt to prevent a double-click from issuing two `POST`s at the network layer (no client-side dedup lock) but disables the Install button once a `localStorage` entry exists for that `extraName`, which covers the realistic case (per spec edge case: two pipelines sharing one extras group show the shared in-flight state and neither offers a fresh Install action while one is pending).

## 3. `GET /api/v1/pipelines/extras/install-jobs/{job_id}`

**Client method**: `apiClient.getExtrasInstallJob(jobId: string)`

**Request**: No body. `job_id` from either the trigger response or a rehydrated `localStorage` entry.

**Response** (200):
```json
{
  "job_id": "...",
  "extra_name": "face",
  "status": "running",
  "created_at": "...",
  "started_at": "...",
  "finished_at": null,
  "command_output": null,
  "restart_required": false
}
```

**Viewer handling**:
- Mapped 1:1 into `ExtrasInstallJob` (see data-model.md), snake_case → camelCase.
- Polled via React Query `refetchInterval: 5000` while `status` is `pending`/`running`; interval stops (`refetchInterval: false`) once `status` is `completed` or `failed`.
- On `failed`: render `commandOutput` in a truncated/expandable block (reusing the existing `<details>`/expandable pattern already used for the raw-JSON config editor in `NewJob.tsx`).
- On `completed` with `restartRequired: true`: render the restart-required banner; do not remove the `localStorage` entry until the user has seen this state (see data-model.md lifecycle) and, separately, don't clear it just because a *later* catalog refresh shows `available: true` for a *different* pipeline in the same group — only clear once the pipeline this job unlocks is itself observed `available: true`, or the user explicitly dismisses it.
- Not found (404, e.g. server restarted and lost in-memory job state, or an invalid stale `localStorage` entry): treated as terminal-failed for local UI purposes (stop polling, drop the `localStorage` entry, allow re-triggering install) rather than retried indefinitely.

## 4. `GET /api/v1/auth/me` *(Addendum)*

**Client method**: `apiClient.getCurrentUser()` via `useCurrentUser()`

**Request**: No body. Requires the current session's Authorization header.

**Responses**:
| Status | Body | Viewer behavior |
|---|---|---|
| `200` | `{ id, username, email, is_admin }` | Mapped to `CurrentUser` (`isAdmin` from `is_admin`), cached 5 min via React Query. Drives the Install button's enabled/disabled state and the Settings "Admin Access" field. |
| `401` | error body | Unauthenticated — `isAdmin` resolves to `'unknown'`, not `false`; the caller shouldn't have offered an admin-gated action to an unauthenticated session in the first place (the pre-existing token-configured gate handles that). |
| `404` | — | Server predates this endpoint. `isAdmin` resolves to `'unknown'` and `endpointUnsupported: true` is set; the UI falls back to the pre-addendum attempt-then-403 behavior for the install action itself, per FR-012's fallback requirement. |

**Key property this contract relies on**: per the backend brief, this endpoint is *never* `403` for any authenticated caller — it's how the viewer finds out whether it *would* get a `403` elsewhere, so it can't itself be admin-gated. If a future server version ever returned `403` here, the viewer would (incorrectly) treat it the same as any other error → `isAdmin: 'unknown'`, falling back to reactive detection rather than crashing — a safe degradation, but worth knowing this contract's guarantee is what makes the proactive UI possible at all.
