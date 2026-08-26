# Phase 1 Data Model: Pipeline Extras Install UI

All entities below are client-side TypeScript types describing data the viewer receives from or sends to the VideoAnnotator API. The viewer owns no persistent server-side state; `localStorage` entries are noted where relevant. Extends `src/types/pipelines.ts` unless noted.

## PipelineDescriptor (extended)

Existing interface (`src/types/pipelines.ts:46`), extended with two new optional fields. Optional because older (pre-v1.5.0) servers won't send them — per Constitution Principle II, their absence must degrade gracefully, not error.

| Field | Type | Notes |
|---|---|---|
| `available` | `boolean \| undefined` | New. `undefined`/missing is treated as `true` (available) for backward compatibility with pre-v1.5.0 servers that predate this field — matches today's implicit behavior where every returned pipeline is usable. |
| `installHint` | `string \| undefined` | New. Mapped from server's `install_hint` (e.g. `"pip install videoannotator[face]"`). Only meaningful when `available === false`. Rendered verbatim as the locked-state explanation; not parsed or templated by the viewer. |

Derived (not a server field, computed client-side): `extraName` — the extras group a locked pipeline belongs to. The brief's `install_hint` is a full shell command string (`pip install videoannotator[face]`), not a bare group name; the viewer extracts the bracketed group name from it for use as the install-trigger endpoint's `{extra}` path segment and as the `localStorage`/job-tracking key. If extraction fails (unexpected hint format), the pipeline is still shown locked with its raw hint text, but no Install button is rendered (nothing sensible to call) — this is a defensive fallback, not an expected path.

## PipelineCatalog / PipelineCatalogResponse (extended)

Existing interfaces (`src/types/pipelines.ts:59,84`). One new field surfaces the top-level signal from the catalog response:

| Field | Type | Notes |
|---|---|---|
| `restartRequired` | `boolean` | New, on `PipelineCatalogResponse` (sibling to `catalog`/`server`, mirroring where the server puts it at the top level of the `GET /api/v1/pipelines` response). Defaults to `false` if absent (pre-v1.5.0 servers, or no install has ever completed this session). |

## ExtrasInstallJob (new)

Represents the state of one install-job as tracked by the server, returned by `GET /api/v1/pipelines/extras/install-jobs/{job_id}`.

| Field | Type | Notes |
|---|---|---|
| `jobId` | `string` | Server-assigned identity; stable across reloads. |
| `extraName` | `string` | The extras group this job installs (e.g. `"face"`). |
| `status` | `'pending' \| 'running' \| 'completed' \| 'failed'` | Drives all UI branching (in-progress / failed / restart-required-once-completed). |
| `createdAt` | `string` (ISO 8601) | |
| `startedAt` | `string \| null` | |
| `finishedAt` | `string \| null` | |
| `commandOutput` | `string \| null` | Populated on `failed` per the brief; may also be present on `completed` (not guaranteed) — rendered as diagnostic detail either way. |
| `restartRequired` | `boolean` | Job-scoped echo of the same signal as `PipelineCatalogResponse.restartRequired`; the two are reconciled by the UI (see `useExtrasInstall` in Phase 1 contracts) rather than trusted independently, since the catalog-level flag is the one that actually gates pipeline availability. |

**State transitions**: `pending → running → (completed | failed)`. No cancellation, no retry, no transition back from a terminal state (matches spec's explicit non-goals). The viewer never writes `status`; it only reads it via polling.

## ExtrasInstallTriggerResponse (new)

Returned synchronously (202) by `POST /api/v1/pipelines/extras/{extra}/install`.

| Field | Type | Notes |
|---|---|---|
| `jobId` | `string` | Same identity subsequently polled via `ExtrasInstallJob`. |
| `extraName` | `string` | Echoes the requested extras group. |
| `status` | `'pending' \| 'running'` | Initial status; the viewer immediately begins polling rather than trusting this as final. |

## RestartRequiredSignal (conceptual, not a distinct type)

Not a separate interface — represented by `PipelineCatalogResponse.restartRequired` (catalog-level, refreshed on every catalog fetch) and echoed per-job on `ExtrasInstallJob.restartRequired`. The catalog-level flag is authoritative for whether the banner should show app-wide; the per-job flag is used only to decide whether a *just-completed* job's local UI state should immediately show the "restart needed" message without waiting for the next catalog poll.

## Local persistence: in-flight install job tracking

Not a server entity — client-only bookkeeping to satisfy the reload-survival requirement (spec FR-006).

**`localStorage` key**: `videoannotator_extras_install_jobs`
**Shape**: `Record<extraName, { jobId: string; startedAt: string }>` — a JSON object map, one entry per extras group with a known in-flight or recently-triggered job.

**Lifecycle**:
- Written when `POST /extras/{extra}/install` succeeds.
- Read on `useExtrasInstall` mount to resume polling for any entries present.
- Removed for a given `extraName` once its job reaches a terminal status (`completed` or `failed`) **and** the user has seen that terminal state at least once (i.e. not removed the instant the poll returns `completed`, so a `completed`+`restartRequired` or a `failed` state persists across a reload until acknowledged, per spec edge cases — "closes the wizard... returns later" must still show the result, not silently forget it).
- Never written to for pipelines that are already `available` — only ever tracks extras the user actually triggered an install for in this browser.

This is a client-side convenience cache only; the server's job record (queryable by `job_id` indefinitely per the brief) remains the source of truth. Losing this `localStorage` entry (private browsing, cleared storage) degrades to "install still runs server-side, but this browser stops showing its progress" — acceptable per spec (no requirement that install progress survive storage clearing, only ordinary reload/navigation).
