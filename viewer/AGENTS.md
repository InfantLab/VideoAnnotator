# AGENTS.md

Guidance for AI coding agents working in this repository. These instructions apply to the entire repo unless a more specific AGENTS.md exists in a subdirectory.

## Project Overview

Video Annotation Viewer is a React + TypeScript web app for visualizing multimodal outputs from the VideoAnnotator backend (COCO keypoints, WebVTT transcripts, RTTM speakers, scene detection). It also integrates with the VideoAnnotator API for job creation, monitoring (SSE), and results viewing.

- Tech stack: React 18, TypeScript, Vite, Tailwind, shadcn/ui
- Validation: Zod schemas
- Testing: Vitest (+ @testing-library), Playwright, Lighthouse CI
- Default API server: `http://localhost:18011` (configurable)

Key docs to review before changes:
- `README.md` (feature overview, usage)
- `docs/DEVELOPER_GUIDE.md` (architecture and structure)
- `docs/CLIENT_SERVER_COLLABORATION_GUIDE.md` (API integration/testing)
- `docs/development/ROADMAP_v0.4.0.md` (upcoming work; includes VideoAnnotator v1.2.x plan)
- `src/test/README.md` (testing guidance)

## Local Environment

Environment variables (create `.env` from `.env.example` when needed):
- `VITE_API_BASE_URL` (default `http://localhost:18011`)
- `VITE_API_TOKEN` (default `dev-token` for local/dev servers)

API client uses localStorage fallbacks:
- `videoannotator_api_url`
- `videoannotator_api_token`

Commands (Bun preferred; npm works too):
- Dev: `bun run dev` (or `npm run dev`)
- Build: `bun run build`
- Lint: `bun run lint`
- Tests: `bun test` (watch), `bun run test:run`, `bun run test:coverage`, `bun run test:ui`
- E2E: `bun run e2e` (install browsers first with `bun run e2e:install`)
- Preview: `bun run preview`

## Repository Structure (essentials)

- `src/components/` UI components (e.g., `VideoAnnotationViewer.tsx`, `VideoPlayer.tsx`, `Timeline.tsx`)
- `src/lib/parsers/` Data parsers (`coco.ts`, `webvtt.ts`, `rttm.ts`, `scene.ts`, `face.ts`, `merger.ts`)
- `src/lib/validation.ts` Zod schemas and validation helpers
- `src/types/annotations.ts` Shared TS types
- `src/api/client.ts` VideoAnnotator API client (jobs, pipelines, SSE, health)
- `src/hooks/useSSE.ts` SSE integration for live job updates
- `src/pages/` Job management pages and settings
- `src/test/` Vitest setup and tests
- `e2e/` Playwright smoke tests
- `scripts/` API debugging tools (`test_api_quick.py`, `browser_debug_console.js`)

## Coding Conventions

- Keep changes minimal, focused, and consistent with existing style.
- TypeScript strict mindset; avoid one-letter variable names; prefer descriptive names.
- Component names in PascalCase; functions in camelCase.
- Tailwind for styling; reuse shadcn/ui components in `src/components/ui/`.
- Validate external data with Zod (`src/lib/validation.ts`).
- For parsing/format support changes, follow the existing parser pattern in `src/lib/parsers/` and update `merger.ts`.
- Do not add license/copyright headers.
- Do not rename or relocate files unless necessary for the task.

## VideoAnnotator Integration

Default server: `http://localhost:18011/` (local dev). Configure via env or localStorage.

API client: see `src/api/client.ts`.
- Health: `/health`, `/api/v1/system/health`
- Jobs: `/api/v1/jobs`, `/api/v1/jobs/:id`
- Pipelines: `/api/v1/pipelines`
- SSE: `/api/v1/events/stream?token=...&job_id=...`
- Debug (optional): `/api/v1/debug/*`

v1.2.x pipeline introspection (roadmap):
- Add a typed client at `src/lib/api/videoAnnotatorClient.ts` (or extend `src/api/client.ts`) for pipeline catalog, parameter schemas, health/capabilities, and version detection.
- Feature-detect endpoints and gracefully degrade to static schemas if server <1.2.1.
- Surface pipeline availability/version/model info in UI; validate job configs against server-declared constraints.

## Testing Guidance

- Unit/integration tests: Vitest. See `src/test/` and `vitest.config.ts`. Prefer colocating tests under `src/test/` and follow file naming used there.
- E2E: Playwright in `e2e/`. Keep smoke tests fast and deterministic.
- Accessibility/performance: Lighthouse via `bun run lhci` (when configured in CI).
- When adding API features, add tests that mock network. For CI, avoid real network calls.
- For local API verification, you may use `scripts/test_api_quick.py` against `http://localhost:18011`.

### Task Tracking and Commits

**CRITICAL: Follow this workflow for feature implementation:**

- **Update tasks.md as you go**: When working from a specification with tasks.md checklist, mark tasks complete `[x]` as you implement them.
- **Commit code and tasks.md together**: Always stage and commit both code changes AND tasks.md updates in the same commit. This keeps progress tracking synchronized with implementation.
- **Commit message pattern**: Include task ID (e.g., "feat: T021-T023 - Config validation hook and component") so commits are traceable to specification.
- **Why this matters**: Keeps specification, implementation, and git history aligned. Allows reviewers to see what was intended vs. what was delivered in each commit.

Example workflow:
```bash
# Implement T021
# Edit code files...
# Update specs/XXX/tasks.md to mark T021 complete [x]
git add src/hooks/useConfigValidation.ts specs/XXX/tasks.md
git commit -m "feat: T021 - Create useConfigValidation hook with debouncing"
```

### QA Testing Guidelines

**CRITICAL: Agents must follow these QA rules:**

- **Don't mark your own homework**: NEVER mark QA checklist items as complete `[x]`. Only human testers can verify functionality works in practice.
- **Just the facts**: QA checklists are for manual verification steps, not detailed technical explanations of what was fixed.
- **Code fixes vs Testing**: Use comments like `**CODE FIXED**:` to note what was changed in code, but leave actual testing checkboxes `[ ]` unchecked for human verification.

QA testers need clean, actionable checklists - not implementation details.

## Documentation

- User-facing docs live in `docs/`. Developer roadmaps and plans in `docs/development/`.
- If you introduce user-visible changes or new developer workflows, update:
  - `CHANGELOG.md`
  - `README.md`
  - Relevant docs under `docs/` (e.g., client-server guide, roadmap)

## Common Tasks and Where to Change

- Add a new annotation format:
  - Types → `src/types/annotations.ts`
  - Parser → `src/lib/parsers/<format>.ts`
  - Detection/merge → `src/lib/parsers/merger.ts`
  - Rendering → `src/components/VideoPlayer.tsx` and timeline if needed
  - Validation → `src/lib/validation.ts`
  - Tests → `src/test/` (parser + integration)

- API UI changes (jobs/pipelines):
  - API calls → `src/api/client.ts`
  - State/hooks → `src/hooks/` (e.g., SSE, token status)
  - Pages → `src/pages/` (create, settings, jobs, job detail)
  - Tests → Vitest unit/integration and Playwright smoke

## Quality Bar

- Lint and typecheck before finishing work: `bun run lint`, `bunx tsc --noEmit` or `npm run typecheck` if present.
- Add or update tests for code you change.
- Ensure dev build (`bun run dev`) and production build (`bun run build`) succeed locally when feasible.

## Pitfalls and Gotchas

- SSE availability varies by server; handle missing SSE endpoint gracefully (fallback to polling if needed).
- Some older VideoAnnotator versions may lack endpoints used in docs; feature-detect and degrade.
- Large files: prefer progressive loading and avoid blocking the UI.

## Contact and Ownership

- Primary maintainer: listed in `README.md` (Contributors/Contact)
- Companion backend: https://github.com/InfantLab/VideoAnnotator

---

Document version: 1.0  
Last updated: 2025-09-18  
Scope: Entire repository

