# Phase 0 Research: Pipeline Extras Install UI

No open `NEEDS CLARIFICATION` markers remained after specification (see spec.md Assumptions). This document records the technical decisions made while translating those assumptions into an implementation approach, plus the codebase investigation that grounds them.

## Decision: Extend the existing catalog fetch rather than add a parallel one

**Decision**: Add an `includeUnavailable` option to `apiClient.getPipelineCatalog()` that appends `?include_unavailable=true` to the existing `GET /api/v1/pipelines` call, and have `PipelineSelectionStep` always request it.

**Rationale**: `usePipelineCatalog()` / `apiClient.getPipelineCatalog()` (`src/hooks/usePipelineCatalog.ts`, `src/api/client.ts:371`) is already the single source of pipeline data for the job-creation wizard and is cached (5 min stale time, plus an internal TTL cache in the client). Adding a second, parallel fetch path for "all pipelines including locked" would create two caches that can disagree. Since `include_unavailable=true` is a strict superset of the default response (confirmed against the brief: existing fields unchanged, `available`/`install_hint` only added, `restart_required` only added at the top level), it's safe to always request the superset and let the UI decide what to show, rather than conditionally switching query params per screen.

**Alternatives considered**:
- A separate `useLockedPipelines()` hook hitting the same endpoint with different params — rejected, doubles cache/staleness surface for no benefit since the superset is always safe to fetch.
- Fetching `include_unavailable=true` only inside `PipelineSelectionStep` and leaving `Settings.tsx`'s diagnostic view on the old query — rejected because it would fragment the query key (React Query would cache two different responses under conceptually "the same" catalog), and the whole point per spec Assumption is to have one behavior. Instead, the option flows through the existing query key/options so both consumers converge on one cached fetch.

## Decision: Hand-write new request/response types instead of regenerating `schema.d.ts`

**Decision**: Define `ExtrasInstallJob`, the install-trigger response, and the extended `PipelineDescriptor` fields (`available`, `installHint`) as hand-written TypeScript in `src/types/pipelines.ts`, independent of the generated `src/api/schema.d.ts`.

**Rationale**: `schema.d.ts` is generated from `openapi_spec.json` (see `scripts/check_openapi.py`), and the existing `PipelineResponse` type (`src/api/client.ts:90`) is derived from it via `paths['/api/v1/pipelines']['get']...`. That generated spec predates VideoAnnotator v1.5.0 and doesn't have `available`/`install_hint`/`restart_required`, nor the two new `extras/*` endpoints — regenerating it would require pointing the generator at a live v1.5.0 server's `/openapi.json`, which isn't available while planning this feature offline. The codebase already has precedent for hand-mapping raw server responses into typed descriptors when the generated schema is insufficient or stale (`mapLegacyPipelineResponse` in `client.ts:277`) — this feature follows that same pattern rather than blocking on regenerating the schema.

**Alternatives considered**:
- Block the feature on first regenerating `schema.d.ts` against a real v1.5.0 server — rejected as a hard blocker; noted instead as a follow-up in the Constitution Check (a stale generated schema doesn't prevent correct behavior, it just means the new fields are hand-typed for now, same as legacy pipeline fields already are).
- Loosely type new responses as `Record<string, unknown>` and cast at call sites — rejected; the rest of the codebase (`PipelineDescriptor`, `PipelineCatalogResponse`, etc.) uses precise interfaces, and the constitution's engineering standards forbid untyped `any` without justification.

## Decision: React Query `refetchInterval` for job-status polling, not a custom SSE/interval hook

**Decision**: Track an in-flight install job with a `useQuery` whose `refetchInterval` is 5000ms while `status` is `pending`/`running`, and `false` (stop) once `completed`/`failed`.

**Rationale**: The codebase already has two polling idioms: hand-rolled `EventSource`-based SSE (`useSSE.ts`, for job *processing* progress, which the server pushes) and React Query for everything else (`usePipelineCatalog`, `useConfigValidation`, etc.). The backend brief is explicit that the extras-install job endpoint is polled, not pushed (no SSE event type is documented for it) — so SSE isn't an option here. React Query's built-in `refetchInterval` (including a function form that can inspect the last-fetched data to decide whether to keep polling) directly expresses "poll every 5s until terminal" without hand-rolling `setInterval`/cleanup logic, and keeps the new hook consistent with `usePipelineCatalog`/`useConfigValidation`'s existing React Query usage.

**Alternatives considered**:
- A bespoke `setInterval`-based hook (like a trimmed-down `useSSE`) — rejected as unnecessary duplication when React Query already solves this exact "poll until terminal" shape declaratively.
- Sub-second polling — explicitly rejected per the spec's FR-009 and the backend brief's "this is a multi-minute operation" framing.

## Decision: Persist in-flight job identity in `localStorage`, keyed by extras group name

**Decision**: On successfully triggering an install, store `{ jobId, extraName, startedAt }` in `localStorage` under a namespaced key (e.g. `videoannotator_extras_install_jobs`, a JSON map keyed by `extraName`). On mount, `useExtrasInstall` rehydrates any stored job(s) and resumes polling.

**Rationale**: Spec FR-006 requires install progress to survive a reload. The existing codebase already uses `localStorage` for exactly this kind of "small piece of client state that should survive a reload" (`videoannotator_api_token`, `videoannotator_api_url` in `client.ts`/`Settings.tsx`). The job-creation wizard (`NewJob.tsx`) holds all its step state in local component `useState`, not the URL, so there's no existing URL-param channel to piggyback on, and introducing one just for this would be a bigger, wizard-wide change out of proportion to this feature. Keying by `extraName` (not by pipeline id) matches the spec's edge case that multiple pipelines can share one extras group — only one in-flight job needs to be tracked per group, and every pipeline in that group can look it up the same way.

**Alternatives considered**:
- URL query param carrying the job id — rejected: the wizard step state isn't URL-driven today, and a locked pipeline can be installed from step 2 while the user later moves to steps 3/4, so a step-scoped URL param would need to survive navigation the wizard doesn't currently model.
- `sessionStorage` — rejected: a multi-minute install plausibly outlives a single tab session if the user closes and reopens the tab; `localStorage`'s longer persistence better matches the "durable server-side job" semantics described in the brief.

## Decision: Admin-gating is attempt-then-handle-403, not a pre-check

**Decision**: Show the Install action whenever the viewer has *any* configured token (i.e. `useTokenStatus`'s existing authenticated state), attempt the install on click, and treat an HTTP 403 response as the "not admin" signal — rendered as a clear inline message, not a silent failure.

**Rationale**: Documented already in spec.md's Assumptions; recorded here because it was the one point in the brief with genuine two-way latitude ("shouldn't be offered at all, **or** should fail clearly with the 403 case handled"). Investigation confirmed the viewer has no existing concept of admin/role/scope anywhere in `src/hooks/useTokenStatus.ts` or `src/types/api.ts` — `validateToken()` only returns `isValid`/`user`/`permissions` (untyped-shape `permissions?: string[]`, never populated by any code path today) — so there is no reliable pre-check available without a new capability-check endpoint, which is out of scope (the brief lists only the three endpoints above). Attempt-then-403 requires no new endpoint and produces a correct, clear outcome either way.

**Alternatives considered**:
- Hide the Install button entirely for non-admins by inferring admin status from `permissions` — rejected: that field is never populated in the current codebase, so building UI logic on it would be speculative and untestable.
- Add a new "am I admin" probe (e.g. calling the install endpoint with a dry-run) — rejected: not part of the documented contract; inventing a probe endpoint violates Constitution Principle II (pin to documented endpoints only).

## Follow-up (non-blocking)

- Regenerate `src/api/schema.d.ts` from a live VideoAnnotator v1.5.0 server's OpenAPI spec once available, and migrate the hand-written types in `src/types/pipelines.ts` back onto generated `paths[...]` types where they now duplicate them. Tracked as a follow-up, not part of this feature's acceptance criteria.
