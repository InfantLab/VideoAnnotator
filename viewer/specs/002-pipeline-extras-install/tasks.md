---
description: "Task list for Pipeline Extras Install UI"
---

# Tasks: Pipeline Extras Install UI

**Input**: Design documents from `specs/002-pipeline-extras-install/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/extras-install-viewer-contract.md, quickstart.md

**Tests**: Included. The project constitution (`.specify/memory/constitution.md`, Engineering Standards) mandates Vitest coverage for new parsers/hooks/components as a standing project requirement, so test tasks are part of each story rather than optional.

**Organization**: Tasks are grouped by user story (matches spec.md's three prioritized stories) so each can be implemented and demoed independently.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Maps the task to US1/US2/US3 from spec.md
- File paths are exact and relative to the repo root

## Path Conventions

Single project (existing Vite/React SPA). All paths under `src/`, tests under `src/test/` (existing convention — not a separate top-level `tests/`).

---

## Phase 1: Setup

**Purpose**: Confirm a clean baseline before touching shared pipeline-catalog code.

- [X] T001 [P] Run the existing pipeline/catalog test suite as a baseline (`src/test/pipeline-integration.test.tsx`, `src/test/api/client.v1.3.test.ts`, `src/test/AppRender.test.tsx`) and confirm all pass pre-change, so any later failure is attributable to this feature.

**Checkpoint**: Clean baseline confirmed.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Shared type/client/hook plumbing that every user story's pipeline-visibility work depends on.

**⚠️ CRITICAL**: Complete before starting any user story phase.

- [X] T002 [P] Extend `PipelineDescriptor` with optional `available?: boolean` and `installHint?: string`, and `PipelineCatalogResponse` with `restartRequired: boolean`, in `src/types/pipelines.ts` (see data-model.md "PipelineDescriptor (extended)" and "PipelineCatalog / PipelineCatalogResponse (extended)"). Missing-field defaults: `available` → `true`, `restartRequired` → `false`.
- [X] T003 [P] Add an `includeUnavailable` option to `apiClient.getPipelineCatalog()` in `src/api/client.ts`: append `?include_unavailable=true` to the `GET /api/v1/pipelines` call it makes via `getPipelines()`, and extend `mapLegacyPipelineResponse` to map the server's `available`/`install_hint` fields onto the new `PipelineDescriptor` fields (per contracts/extras-install-viewer-contract.md §1) and the response's top-level `restart_required` onto `PipelineCatalogResponse.restartRequired` in `buildCatalogResponse`.
- [X] T004 [US-shared] Update `pipelineCatalogQueryOptions`/`usePipelineCatalog` in `src/hooks/usePipelineCatalog.ts` to always call `getPipelineCatalog({ includeUnavailable: true, ... })`, so every existing consumer (`NewJob.tsx`, `Settings.tsx`) transparently starts receiving the superset response. Depends on: T002, T003.

**Checkpoint**: Catalog fetch now returns every known pipeline (locked included) plus the restart flag. No UI changes yet — existing screens are unaffected until Phase 3 renders the new state.

---

## Phase 3: User Story 1 - See what pipelines exist, even if not installed (Priority: P1) 🎯 MVP

**Goal**: Select Pipelines shows every server-known pipeline, with uninstalled ones clearly locked and explained.

**Independent Test**: Point the viewer at a core-only server fixture/mock, open Select Pipelines, confirm every known pipeline appears and locked ones are visually distinct with a reason shown.

### Tests for User Story 1

- [X] T005 [P] [US1] Unit test `apiClient.getPipelineCatalog({ includeUnavailable: true })` mapping: verifies `available`/`installHint`/`restartRequired` are mapped correctly, and that a response shaped like a pre-v1.5.0 server (fields absent) defaults to `available: true`/`restartRequired: false`, in `src/test/api/client.extras-install.test.ts`.

### Implementation for User Story 1

- [X] T006 [P] [US1] Create `LockedPipelineCard` presentational component — renders a pipeline's name/description, a locked badge, and its `installHint` text — in `src/components/LockedPipelineCard.tsx`. No install action yet (added in Phase 4).
- [X] T007 [US1] Update `PipelineSelectionStep` in `src/pages/NewJob.tsx` to iterate every pipeline from `usePipelineCatalog()` (not just implicitly-available ones), rendering the existing selectable card for `available !== false` pipelines and `LockedPipelineCard` for `available === false` ones, grouped the same way as today (`groupedPipelines`). Depends on: T004, T006.
- [X] T008 [P] [US1] Integration test: a core-only catalog fixture (stub + several `available: false` pipelines with `installHint`) renders all pipelines, locked ones visually distinct with their hint text visible, in `src/test/integration/extras-install.test.tsx`. A fully-available catalog fixture renders identically to today (regression guard for spec Acceptance Scenario 3). Depends on: T007.

**Checkpoint**: User Story 1 fully functional and independently demoable — locked pipelines are visible with reasons, nothing installable yet.

---

## Phase 4: User Story 2 - Install a missing pipeline from the app (Priority: P1)

**Goal**: An admin-authenticated user can trigger and track an extras install for a locked pipeline without leaving the app.

**Independent Test**: As an admin-authenticated session against a mock/fixture server, click Install on a locked pipeline and confirm a job starts and its in-progress state is observable and survives a reload; as a non-admin session, confirm the action is absent or clearly blocked.

### Tests for User Story 2

- [X] T011 [P] [US2] Unit tests for `apiClient.installPipelineExtras` and `apiClient.getExtrasInstallJob` covering 202/401/403/422 (trigger) and 200/404 (status) response handling, in `src/test/api/client.extras-install.test.ts`.
- [X] T013 [P] [US2] Unit tests for `useExtrasInstall`: trigger writes the `localStorage` job-map entry; mount rehydrates a stored entry and resumes polling; polling stops once status is terminal; a stale/expired job (404) is dropped, in `src/test/hooks/useExtrasInstall.test.tsx` (403-on-trigger surfacing is exercised at the UI layer in T015, since the hook itself just propagates the mutation error).

### Implementation for User Story 2

- [X] T009 [P] [US2] Add `ExtrasInstallJob` and `ExtrasInstallTriggerResponse` types to `src/types/pipelines.ts` (per data-model.md), and a small `extraNameFromInstallHint(hint: string): string | null` helper in `src/lib/pipelineExtras.ts` that extracts the bracketed group name from a hint like `"pip install videoannotator[face]"`.
- [X] T010 [P] [US2] Add `apiClient.installPipelineExtras(extraName: string)` (`POST /api/v1/pipelines/extras/{extra}/install`) and `apiClient.getExtrasInstallJob(jobId: string)` (`GET /api/v1/pipelines/extras/install-jobs/{job_id}`) to `src/api/client.ts`, mapping snake_case response fields to the camelCase types from T009 and surfacing 401/403/422/404 per contracts/extras-install-viewer-contract.md §2–3 (reuse the existing `APIError`/`parseApiError` pattern already used elsewhere in the client). Depends on: T009.
- [X] T012 [US2] Create `useExtrasInstall` hook in `src/hooks/useExtrasInstall.ts`: a mutation that calls `installPipelineExtras`, and on success writes `{ jobId, startedAt }` into the `videoannotator_extras_install_jobs` localStorage map keyed by `extraName` (per data-model.md); a query per tracked `extraName` polling `getExtrasInstallJob` with `refetchInterval: 5000` while `status` is `pending`/`running` and `false` once terminal; rehydrates tracked jobs from `localStorage` on mount. Depends on: T010.
- [X] T014 [US2] Wire an Install action into `LockedPipelineCard` / `PipelineSelectionStep`: shown when the session is authenticated (`useTokenStatus`), calls `useExtrasInstall`'s trigger for that pipeline's `extraName` (via T009's helper), and renders the returned in-progress state scoped to that pipeline (and any sibling pipelines sharing the same `extraName`) rather than globally. A `403` result renders "Administrator privileges are required to install pipeline extras." instead of a generic error. Files: `src/components/LockedPipelineCard.tsx`, `src/pages/NewJob.tsx`. Depends on: T007, T012.
- [X] T015 [P] [US2] Integration test: admin session clicking Install shows in-progress state scoped to the right pipeline(s); non-admin session sees no Install action or a clear blocked message on attempt; reloading mid-install (simulated by remounting with a pre-seeded `localStorage` entry) still shows in-progress state, in `src/test/integration/extras-install.test.tsx`. Depends on: T014.

**Checkpoint**: User Stories 1 and 2 both independently functional — admins can see, trigger, and track installs; non-admins are cleanly blocked.

---

## Phase 5: User Story 3 - Understand install progress, failure, and the restart step (Priority: P2)

**Goal**: Slow installs read as "working," failures are actionable, and a completed-but-not-restarted pipeline is explained rather than looking broken or done.

**Independent Test**: Simulate `pending`, `running`, `failed`, and `completed`+`restartRequired: true` job statuses against the UI and confirm each renders distinct, accurate messaging; confirm the pipeline returns to normal once a subsequent catalog fetch reports `available: true`.

### Implementation for User Story 3

- [X] T016 [P] [US3] Add failure-state rendering to `LockedPipelineCard`: when the tracked job's `status` is `failed`, show a failure indicator plus an expandable/truncated view of `commandOutput` (reuse the `<details>` pattern already used for the raw-JSON config editor in `NewJob.tsx`). File: `src/components/LockedPipelineCard.tsx`.
- [X] T017 [P] [US3] Create `RestartRequiredBanner` component in `src/components/RestartRequiredBanner.tsx` — takes `restartRequired: boolean` and renders "Restart the VideoAnnotator server to finish activating new pipelines" styled as an informational/expected-step banner, not an error/alert.
- [X] T018 [US3] Render `RestartRequiredBanner` in `PipelineSelectionStep` driven by the catalog's `restartRequired` (from T004), and optionally in `src/pages/Settings.tsx`'s diagnostics view (it already surfaces server/catalog state). Depends on: T004, T017.
- [X] T019 [US3] Implement the `localStorage` entry lifecycle in `useExtrasInstall` (per data-model.md "Local persistence" section): keep a terminal (`completed`/`failed`) entry visible across reloads until the pipeline it unlocks is observed `available: true` on a later catalog fetch, or the user dismisses it; treat a `404` from `getExtrasInstallJob` (stale/expired job) as terminal-failed locally and drop the entry so Install can be re-triggered. File: `src/hooks/useExtrasInstall.ts`. Depends on: T012.
- [X] T020 [P] [US3] Integration test covering: `pending`/`running` shows "this will take a while" messaging (not a plain spinner); `failed` shows the expandable `commandOutput`; `completed` + `restartRequired: true` shows the expected-step banner (not error wording) and the pipeline itself still renders locked; a subsequent catalog fetch with `available: true` for that pipeline clears all install/restart UI for it and it renders identically to an always-available pipeline. File: `src/test/integration/extras-install.test.tsx`. Depends on: T016, T018, T019.

**Checkpoint**: All three user stories complete. Full `quickstart.md` manual flow should be demoable end-to-end.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Wrap-up that spans all three stories.

- [ ] T021 [P] Walk through `specs/002-pipeline-extras-install/quickstart.md`'s manual verification flow against a real or mocked v1.5.0-capable server and record results.
- [X] T022 [P] Document the new endpoints/UI behavior in `docs/CLIENT_SERVER_COLLABORATION_GUIDE.md`.
- [X] T023 Add a `CHANGELOG.md` entry for the extras-install UI.
- [X] T024 [P] Run `tsc --noEmit`, ESLint, and the full Vitest suite across all touched files; fix any regressions.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Setup. Blocks every user story (all three need the extended catalog shape).
- **User Story 1 (Phase 3)**: Depends on Foundational only. This is the MVP slice.
- **User Story 2 (Phase 4)**: Depends on Foundational; its UI wiring (T014) additionally depends on US1's `LockedPipelineCard`/`PipelineSelectionStep` work (T006, T007), since Install is added to the card US1 introduces. Its types/client/hook tasks (T009, T010, T011, T012, T013) have no US1 dependency and can proceed in parallel with US1.
- **User Story 3 (Phase 5)**: Depends on Foundational and on US2's job-tracking plumbing (`useExtrasInstall`, T012) since it extends that hook's lifecycle and renders states the hook produces.
- **Polish (Phase 6)**: Depends on all three stories being complete.

### User Story Dependencies

- **US1 (P1)**: Independent after Foundational.
- **US2 (P1)**: Independent after Foundational for its data-layer tasks (T009–T013); its UI task (T014) integrates into US1's card component.
- **US3 (P2)**: Builds on US2's job-tracking hook; not independently meaningful without US2 (there'd be no job to show progress/failure/restart for), but its own tasks (banner, failure rendering, lifecycle) are separable work once US2 lands.

### Within Each User Story

- Tests before/alongside implementation (written first per constitution's testing standard; can be developed in parallel with the code they cover per this project's usual pace, but must pass before the story's checkpoint is considered done).
- Types → client methods → hooks → components → integration wiring → integration tests.

### Parallel Opportunities

- T001 (baseline) can run alone before anything else.
- T002 and T003 (Foundational) touch different files and can run in parallel; T004 depends on both.
- T005, T006 (US1) can run in parallel; T007 depends on T006 and the Foundational checkpoint.
- T009, T010, T011, T013 (US2 data layer) can run in parallel with each other and with US1's phase once Foundational is done; only T014 needs US1's card component to exist.
- T016 and T017 (US3) can run in parallel; both feed into T018/T020.

---

## Parallel Example: Foundational + User Story 1

```bash
# After Setup (T001):
Task: "Extend PipelineDescriptor/PipelineCatalogResponse types in src/types/pipelines.ts"          # T002
Task: "Add includeUnavailable support + field mapping in src/api/client.ts"                         # T003
# then:
Task: "Update usePipelineCatalog to request includeUnavailable and expose restartRequired"          # T004

# Once Foundational checkpoint is reached:
Task: "Unit test getPipelineCatalog(includeUnavailable) mapping"                                    # T005
Task: "Create LockedPipelineCard component"                                                          # T006
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Phase 1: Setup.
2. Phase 2: Foundational (blocks everything).
3. Phase 3: User Story 1.
4. **STOP and VALIDATE**: on a core-only server (or fixture), confirm locked pipelines are visible with reasons — this alone already fixes the most confusing part of today's experience.
5. Demo if ready; US2/US3 add the actual install capability on top.

### Incremental Delivery

1. Setup + Foundational → catalog returns the superset shape.
2. US1 → locked pipelines visible (MVP demo).
3. US2 → admin can trigger and track an install (demo: install `face` end-to-end on a test server).
4. US3 → progress/failure/restart messaging polish (demo: full quickstart.md flow, including the restart step).
5. Polish.

### Notes

- [P] tasks touch different files and have no incomplete dependency.
- Commit after each task or logical group, per repo convention (see CLAUDE.md / constitution "Commits").
- Stop at each phase checkpoint to validate that story independently before moving on.
- Avoid: adding install-cancel or auto-restart UI (explicit non-goals in spec.md).
