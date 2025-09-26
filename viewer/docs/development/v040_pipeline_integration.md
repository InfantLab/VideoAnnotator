# v0.4.0 – VideoAnnotator v1.2.x Pipeline Integration Plan

Last updated: 2025-09-18
Scope: Frontend integration for dynamic pipeline discovery, parameter introspection, and capability‑aware UX.

## Goals
- Discover pipelines, parameters, and constraints dynamically from the VideoAnnotator server (v1.2.1/1.2.2).
- Drive the “Create New Job” wizard from server‑provided schemas with validation.
- Make the Viewer capability‑aware and results‑aware (gating overlays, smarter artifact expectations).
- Support version negotiation with graceful fallback for older servers.

## Assumptions
- A v1.2.x server runs at `http://localhost:18011` (configurable via `VITE_API_BASE_URL` or localStorage).
- The server exposes a fixed pipeline catalog and per‑pipeline parameter schemas in v1.2.x.
- Some endpoints may vary by minor version, so we feature‑detect and cache capabilities.

## Deliverables
- Typed client and schema/types to fetch and parse the pipeline catalog and parameter schemas.
- Dynamic pipeline selection and parameter forms in the job creation wizard.
- Viewer overlays and expectations gated by server capabilities and/or job result metadata.
- Tests covering 1.2.1/1.2.2 responses plus fallback behavior for older servers.

## Implementation Plan

### 1) Types and Client
- Add `src/types/pipelines.ts` with:
  - `PipelineCatalog`, `PipelineDescriptor` (id/name/group/version/model, capabilities),
  - `ParameterSchema` (name, type, default, enum/options, min/max/step, required, group, description),
  - `ServerInfo` (version, feature flags, build/meta), `Capabilities`.
- Client surface (extend `src/api/client.ts` or add `src/lib/api/videoAnnotatorClient.ts`):
  - `getPipelineCatalog()` – list all pipelines and high‑level metadata.
  - `getPipelineSchema(pipelineName)` – parameter schema/constraints.
  - `getServerInfo()` – version and feature flags (use debug/system endpoints if available).
- Version negotiation and feature detection:
  - Probe endpoints in priority order; record availability in a local feature map.
  - Cache server version and feature map (memory + localStorage) with expiry.

### 2) Data Fetching and Caching
- Use `@tanstack/react-query` for queries:
  - `pipelines.catalog` (staleTime ~ 5 min),
  - `pipelines.schema[:name]`,
  - `server.info`.
- Invalidate caches when server version changes.
- Persist last successful catalog to localStorage for offline/dev diagnostics.

### 3) Job Wizard – Dynamic Catalog and Parameter UI
- In `src/pages/CreateNewJob.tsx`:
  - Replace hard‑coded pipeline lists with server catalog groups (Face/OpenFace3, Person, Audio, Scenes, etc.).
  - Render per‑pipeline toggle + info (version/model/description).
- Parameter forms:
  - Map `ParameterSchema.type` → UI controls: boolean→switch, number→slider/number, enum→select, string→input.
  - Apply constraints (min/max/step, required), with tooltips from `description`.
  - Validate via Zod prior to submit; show inline errors.
- Request validation:
  - Build config JSON limited to enabled pipelines and validated params.
  - Block submit on invalid inputs; show clear, actionable messages.

### 4) Viewer – Capability‑Aware Overlays and Smarter Results
- Provide server catalog/capabilities via context or a hook.
- Gate overlay toggles by capability and job results metadata:
  - Example: `src/components/OpenFace3Controls.tsx` uses server capability to show/hide toggles (landmarks, AUs, head pose, gaze, emotion) with tooltips.
  - If a pipeline didn’t run in the job, disable the toggle and show “Not in this job’s results”.
- Expected artifacts:
  - Use catalog to tailor which artifacts to fetch/display per pipeline (e.g., COCO, WebVTT, RTTM, scene JSON).
  - If a user enables a feature without corresponding data, prefer a gentle hint over silent no‑ops.

### 5) Diagnostics and Settings
- In `src/pages/CreateSettings.tsx`:
  - Add “Server Pipelines” diagnostics card:
    - Show server version, pipelines, per‑pipeline version/model.
    - Buttons: “Refresh Catalog”, “Clear Pipeline Cache”.
  - Show feature detection summary (e.g., “Introspection available”, “SSE available”).

### 6) Fallback and Migration
- If 1.2.x endpoints aren’t available:
  - Use the existing static pipeline list (keep a small constants map).
  - Hide advanced parameter UI; keep minimal selection UX.
  - Maintain current `/api/v1/pipelines` integration if that’s all we have.
- Visibly indicate “Limited mode” and link to docs for server upgrade.

### 7) Error Handling and UX
- Use `handleAPIError()` to map API errors → user‑friendly toasts/banners.
- Offline/degraded mode:
  - Use last cached catalog; mark it “stale” and allow manual refresh.
- Accessibility:
  - Ensure form controls convey errors and constraints to screen readers.

### 8) Testing
- Unit
  - Schema→UI mapping (enum to select, constraints to slider/input limits).
  - Zod validation against min/max/required.
  - Version negotiation/feature detection logic.
- Integration
  - Mocked responses for 1.2.1 and 1.2.2 (catalog + schema).
  - Fallback behavior with missing endpoints.
- E2E (Playwright)
  - Wizard loads catalog from server → parameters render → valid submit.
  - Viewer disables unsupported overlays and shows “not in job” hints.

### 9) Documentation
- Update `docs/CLIENT_SERVER_COLLABORATION_GUIDE.md` with pipeline discovery flow, local server instructions.
- Add a short “Dynamic Pipeline Introspection (v1.2.x)” snippet to `README.md`.
- Developer notes in `docs/DEVELOPER_GUIDE.md` covering schema→UI mapping and caching.

## File Touchpoints
- API/types
  - `src/types/pipelines.ts`
  - `src/lib/api/videoAnnotatorClient.ts` (or extend `src/api/client.ts`)
- Job wizard UI
  - `src/pages/CreateNewJob.tsx`
  - `src/pages/CreateSettings.tsx`
- Viewer UI
  - `src/components/OpenFace3Controls.tsx`
  - `src/components/UnifiedControls.tsx`
  - `src/components/Timeline.tsx`
- Hooks/Context
  - Add `usePipelineCatalog()` wrapper for React Query
  - Consider a small `PipelineCatalogProvider`
- Tests
  - `src/test/` unit/integration
  - `e2e/` Playwright smoke/flows

## Milestones
- M1: Types + client + feature detection; caching and settings diagnostics.
- M2: Dynamic job wizard (catalog + parameter forms + validation + submit).
- M3: Viewer capability‑aware overlays + expected artifacts mapping.
- M4: Tests (unit/integration/e2e) and docs updates.

## Success Criteria
- Pipeline discovery works against local VA v1.2.1/1.2.2.
- Job wizard reflects server catalog and enforces constraints client‑side.
- Viewer clearly communicates unsupported/unavailable features and job‑specific pipeline coverage.
- Fallback behavior is robust and obvious when introspection endpoints are missing.
