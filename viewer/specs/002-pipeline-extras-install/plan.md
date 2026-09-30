# Implementation Plan: Pipeline Extras Install UI

**Branch**: `feature/vlm-annotation-support` (continuing on current branch; see spec.md header note) | **Date**: 2026-08-26 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `specs/002-pipeline-extras-install/spec.md`

**Note**: This template is filled in by the `/speckit-plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Today `PipelineSelectionStep` in the job-creation wizard only ever sees pipelines the server can currently run, so a core-only VideoAnnotator install shows almost nothing. The backend (v1.5.0) now exposes every known pipeline via `?include_unavailable=true` (with `available`/`install_hint` per pipeline and a top-level `restart_required`), plus an admin-only fire-and-forget install-job endpoint and a job-status endpoint. This plan wires the viewer's existing pipeline-catalog fetch, React Query layer, and `PipelineSelectionStep` UI up to those three endpoints: render locked pipelines with an Install action, poll per-pipeline install-job status at a 5s interval while in flight, surface failures with `command_output`, and show a restart-required banner once an install completes but the server hasn't restarted. No new backend, no new top-level page — this is additive UI + client/hook wiring on the existing job-creation flow.

## Technical Context

**Language/Version**: TypeScript 5, React 18 (existing SPA — no new stack)
**Primary Dependencies**: `@tanstack/react-query` (already used for catalog fetch/mutations elsewhere), existing `apiClient` (`src/api/client.ts`), shadcn/radix UI primitives already in `src/components/ui/`. No new runtime dependency required.
**Storage**: Browser `localStorage`, for persisting in-flight install `job_id`s across reloads (mirrors existing `videoannotator_api_token`/`videoannotator_api_url` use of localStorage in `src/api/client.ts`). No server-side storage owned by the viewer.
**Testing**: Vitest + Testing Library (existing `src/test/` conventions — see `src/test/api/client.v1.3.test.ts` for client-method tests, `src/test/integration/*.test.tsx` for flow-level tests).
**Target Platform**: Browser SPA (existing Vite build), same as rest of app.
**Project Type**: Single web application (no frontend/backend split within this repo — backend is the separate VideoAnnotator project).
**Performance Goals**: Polling at ~5s while a job is `pending`/`running`; no polling once terminal. No new bundle-size-relevant dependency (stays within the constitution's 300KB gzipped budget / 50KB-per-dependency code-splitting threshold).
**Constraints**: Must degrade gracefully against a pre-v1.5.0 server that doesn't support `include_unavailable`, the install endpoint, or the job-status endpoint (older servers just won't send `available`/`install_hint`/`restart_required` — treat their absence as "available" per existing optional-field handling, per Constitution Principle II).
**Scale/Scope**: One wizard step (`PipelineSelectionStep` in `src/pages/NewJob.tsx`), a small set of new client methods/types, one new polling hook, one new small UI unit (locked-pipeline card + install button + restart banner). Not a new page, not a new route.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **Principle I (Local-First / No Telemetry)**: PASS. All new calls go to the same user-configured VideoAnnotator API base URL already in use (`apiClient`'s existing `baseURL`/`token`). No new third-party endpoint, no analytics.
- **Principle II (Stable API Consumption Contract)**: PASS with a documented gap — the generated `src/api/schema.d.ts` (from `openapi_spec.json`) does not yet include the v1.5.0 extras-install endpoints or the `available`/`install_hint`/`restart_required` fields, because no updated OpenAPI spec from the live server was available while planning. New request/response shapes are hand-written types (see `data-model.md` and `contracts/`) sourced from the backend team's handoff brief rather than a regenerated schema. This is the same pattern already used for hand-mapped legacy pipeline responses (`mapLegacyPipelineResponse` in `client.ts`). Regenerating `schema.d.ts` against a live v1.5.0 server before/at merge time is called out as a follow-up, not a blocker (see research.md).
- **Principle III (Faithful Annotation Display)**: N/A — this feature doesn't touch annotation overlays/data.
- **Principle IV (Composable View Layers)**: N/A — not an annotation layer; still respected in spirit by keeping the new UI self-contained within `PipelineSelectionStep` and a small set of new hooks/components rather than reaching into unrelated state.
- **Principle V (API & UI Stability)**: PASS. No URL params, keyboard shortcuts, or file-format expectations change. Existing pipeline-selection behavior on servers that don't support the new fields is unchanged (graceful degradation, see Constraints above).
- **Engineering Standards**: New code ships typed (`strict: true`, no new `any`), covered by Vitest unit tests for the new client methods/hook and at least one integration test for the locked/install/restart UI states, and stays under the dependency/bundle budget (no new dependency added).

No violations requiring the Complexity Tracking table.

## Project Structure

### Documentation (this feature)

```text
specs/002-pipeline-extras-install/
├── plan.md              # This file (/speckit-plan command output)
├── research.md          # Phase 0 output (/speckit-plan command)
├── data-model.md         # Phase 1 output (/speckit-plan command)
├── quickstart.md         # Phase 1 output (/speckit-plan command)
├── contracts/             # Phase 1 output (/speckit-plan command)
│   └── extras-install-viewer-contract.md
└── tasks.md              # Phase 2 output (/speckit-tasks command - NOT created by /speckit-plan)
```

### Source Code (repository root)

```text
# Option 1: Single project (this repo — existing structure, no new top-level dirs)
src/
├── api/
│   ├── client.ts                 # + getPipelineCatalog(includeUnavailable), installPipelineExtras(), getExtrasInstallJob()
│   └── schema.d.ts                # unchanged by this feature (see Constitution Check note)
├── types/
│   └── pipelines.ts               # + available/installHint on PipelineDescriptor; + ExtrasInstallJob, RestartRequiredSignal types
├── hooks/
│   ├── usePipelineCatalog.ts      # + surfaces restartRequired from catalog response
│   └── useExtrasInstall.ts        # NEW: trigger-install mutation + per-job polling query + localStorage persistence
├── components/
│   ├── LockedPipelineCard.tsx     # NEW: locked pipeline row — badge, install hint, Install button, progress/failure state
│   └── RestartRequiredBanner.tsx  # NEW: small reusable banner, driven by restartRequired
└── pages/
    ├── NewJob.tsx                 # PipelineSelectionStep renders every pipeline incl. locked ones via LockedPipelineCard
    └── Settings.tsx                # OPTIONAL: also renders RestartRequiredBanner (diagnostics surface already shows catalog)

src/test/
├── api/client.v1.3.test.ts        # extend, or new client.extras-install.test.ts, for the 2 new client methods
├── hooks/useExtrasInstall.test.ts  # NEW: polling/terminal-state/localStorage-persistence behavior
└── integration/extras-install.test.tsx  # NEW: locked pipeline → install → progress → failure / restart-required states
```

**Structure Decision**: No new project or route. This is additive work inside the existing single-page React app, following the existing `src/{api,types,hooks,components,pages}` layout and the existing `apiClient` + React Query hook pattern (mirrors `usePipelineCatalog.ts` / `useJobCancellation.ts`).

## Complexity Tracking

*No constitution violations — table intentionally omitted.*
