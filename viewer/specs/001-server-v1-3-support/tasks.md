---
description: "Implementation tasks for VideoAnnotator Server v1.3.0 Client Support"
---

# Tasks: VideoAnnotator Server v1.3.0 Client Support

**Input**: Design documents from `/specs/001-server-v1-3-support/`  
**Prerequisites**: plan.md ✅, spec.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

**Tests**: Tests are included as recommended best practice (following existing test patterns in `src/test/`)

**Organization**: Tasks grouped by user story (P1, P2, P3) to enable independent implementation and testing.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3, US4, US5)
- Include exact file paths in descriptions

## Path Conventions

**Web application (SPA)**: `src/` and `src/test/` at repository root (following existing structure)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and dependency verification

- [x] T001 Verify VideoAnnotator server v1.3.0 is running at http://localhost:18011 and API key is available
- [x] T002 [P] Create `.env` file with VITE_API_BASE_URL and VITE_API_TOKEN from server console
- [x] T003 [P] Verify existing dependencies support new features (React 18, Vite 5, Zod, shadcn/ui)

**Checkpoint**: Development environment ready ✅

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story work begins

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

- [x] T004 Create TypeScript types in src/types/api.ts for v1.3.0 entities (ErrorEnvelope, ServerCapabilities, ConfigValidationResult, EnhancedHealthResponse, JobCancellationResponse)
- [x] T005 [P] Create Zod schemas in src/lib/validation.ts for all v1.3.0 response types (ErrorEnvelopeSchema, ConfigValidationResultSchema, EnhancedHealthResponseSchema, etc.)
- [x] T006 [P] Create src/lib/errorHandling.ts with parseApiError function for defensive ErrorEnvelope parsing (supports both v1.3.0 and legacy formats)
- [x] T007 [P] Create src/api/capabilities.ts with detectServerCapabilities function (inspects health endpoint for v1.3.0 fields)
- [x] T008 Extend src/api/client.ts with new method signatures (cancelJob, validateConfig, validatePipeline, enhanced getHealth)
- [x] T009 [P] Create src/hooks/useServerCapabilities.ts hook for capability detection with caching and refresh

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel ✅

---

## Phase 3: User Story 1 - Job Cancellation (Priority: P1) 🎯 MVP

**Goal**: Enable users to cancel running/queued jobs with confirmation dialog and optimistic UI updates

**Independent Test**: Submit a job via creation wizard, navigate to job detail page, click cancel button, verify job status updates to "cancelled" within 5 seconds with success toast

### Tests for User Story 1

> **Write these tests FIRST, ensure they FAIL before implementation**

- [x] T010 [P] [US1] Create unit test in src/test/api/client.v1.3.test.ts for cancelJob endpoint with mocked fetch responses (success, 400 already-cancelled, 404 not-found)
- [ ] T011 [P] [US1] Create unit test in src/test/hooks/useJobCancellation.test.ts for confirmation dialog flow and optimistic updates
- [ ] T012 [P] [US1] Create component test in src/test/components/JobCancelButton.test.tsx for button states (enabled, disabled, loading) and click handling
- [ ] T013 [P] [US1] Create integration test in src/test/integration/jobCancellation.test.ts for full cancel flow (API mock → optimistic update → cache invalidation → toast)

### Implementation for User Story 1

- [x] T014 [P] [US1] Implement cancelJob method in src/api/client.ts (POST /api/v1/jobs/{job_id}/cancel with Bearer auth)
- [ ] T015 [P] [US1] Create src/hooks/useJobCancellation.ts hook with confirmation dialog, optimistic updates, and error rollback
- [ ] T016 [US1] Create src/components/JobCancelButton.tsx with AlertDialog for confirmation, loading state, and disabled states for non-cancellable jobs
- [ ] T017 [US1] Integrate JobCancelButton into src/pages/CreateJobDetail.tsx (render conditionally based on job status)
- [ ] T018 [P] [US1] Add cancellation button to job list items in src/pages/CreateJobs.tsx (if applicable, with same conditional logic)
- [ ] T019 [US1] Update SSE handling in src/hooks/useSSE.ts to refresh job data when cancellation event received
- [ ] T020 [US1] Add toast notifications in JobCancelButton for success/error cases using src/hooks/use-toast.ts

**Checkpoint**: Job cancellation fully functional - users can cancel jobs from detail and list pages with immediate feedback

---

## Phase 4: User Story 2 - Configuration Validation (Priority: P1) 🎯 MVP

**Goal**: Provide real-time configuration validation with field-level errors, hints, and submit blocking for invalid configs

**Independent Test**: Open job creation wizard, enter invalid confidence_threshold (1.5), verify validation error displays with field name, message, hint, and submit button is disabled

### Tests for User Story 2

> **Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T021 [P] [US2] Create unit test in src/test/api/client.v1.3.test.ts for validateConfig and validatePipeline endpoints with various error/warning scenarios
- [ ] T022 [P] [US2] Create unit test in src/test/hooks/useConfigValidation.test.ts for debouncing (500ms), caching by config hash, and validation state management
- [ ] T023 [P] [US2] Create component test in src/test/components/ConfigValidationPanel.test.tsx for error/warning display, grouping by field, and hint rendering
- [ ] T024 [P] [US2] Create integration test in src/test/integration/configValidation.test.ts for full validation flow (config change → debounced API call → field-level error display → submit blocking)

### Implementation for User Story 2

- [ ] T025 [P] [US2] Implement validateConfig method in src/api/client.ts (POST /api/v1/config/validate with full config object)
- [ ] T026 [P] [US2] Implement validatePipeline method in src/api/client.ts (POST /api/v1/pipelines/{name}/validate with pipeline-specific config)
- [ ] T027 [P] [US2] Create src/hooks/useConfigValidation.ts with debounced validation (500ms), config hash caching, and result state management
- [ ] T028 [US2] Create src/components/ConfigValidationPanel.tsx to display errors/warnings with field names, messages, hints, and error codes (collapsible for dev mode)
- [ ] T029 [US2] Integrate ConfigValidationPanel into src/pages/CreateNewJob.tsx below the JSON config editor
- [ ] T030 [US2] Add submit button state management in CreateNewJob.tsx (disable if validationResult.valid === false)
- [ ] T031 [US2] Add warning confirmation dialog in CreateNewJob.tsx for valid-with-warnings case ("Submit Anyway?" prompt)
- [ ] T032 [US2] Add inline field-level error indicators in config editor (if using form inputs instead of JSON editor)

**Checkpoint**: Configuration validation fully functional - users get immediate feedback on invalid configs, submit is blocked, warnings allow submission with confirmation

---

## Phase 5: User Story 3 - Enhanced Authentication Management (Priority: P2)

**Goal**: Improve token setup UX with clear status indicators, server version display, and helpful guidance for auth issues

**Independent Test**: Configure valid token in settings, verify green "connected" indicator with server version; change to invalid token, verify clear error message with guidance

### Tests for User Story 3

> **Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T033 [P] [US3] Create unit test in src/test/api/capabilities.test.ts for server capability detection with v1.2.x and v1.3.0 health responses
- [ ] T034 [P] [US3] Create component test in src/test/components/TokenStatusIndicator.test.tsx for status states (connected, error, warning) and version display
- [ ] T035 [P] [US3] Create integration test in src/test/integration/authentication.test.ts for token setup flow, validation, and error handling

### Implementation for User Story 3

- [ ] T036 [P] [US3] Implement server capability detection in src/api/capabilities.ts (check for gpu_status/worker_info fields in health response)
- [ ] T037 [P] [US3] Create React Context in src/contexts/ServerCapabilitiesContext.tsx to share capabilities app-wide with refresh mechanism
- [ ] T038 [US3] Extend src/components/TokenStatusIndicator.tsx to display server version from capabilities context
- [ ] T039 [US3] Add authentication mode indicator (required/optional) to TokenStatusIndicator with appropriate icon/color
- [ ] T040 [US3] Update src/pages/CreateSettings.tsx with prominent token setup guide for first-time users (when no token configured)
- [ ] T041 [US3] Add "unsecured connection" warning in TokenStatusIndicator when auth is disabled (VIDEOANNOTATOR_REQUIRE_AUTH=false)
- [ ] T042 [US3] Add manual "Refresh Server Info" button in CreateSettings that calls detectServerCapabilities and updates context

**Checkpoint**: Authentication management improved - users see clear connection status, server version, and helpful setup guidance

---

## Phase 6: User Story 4 - Improved Error Handling (Priority: P2)

**Goal**: Provide consistent error display across all operations with field-level details, hints, and collapsible technical info

**Independent Test**: Trigger network error, validation error, and auth error; verify each displays with consistent format, appropriate detail, and actionable guidance

### Tests for User Story 4

> **Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T043 [P] [US4] Create unit test in src/test/lib/errorHandling.test.ts for parseApiError with v1.3.0 ErrorEnvelope, legacy formats (string, {error: ...}, {message: ...}), and edge cases
- [ ] T044 [P] [US4] Create component test in src/test/components/ErrorDisplay.test.tsx for structured error display (message, hint, code, request_id) and collapsible technical details
- [ ] T045 [P] [US4] Create integration test in src/test/integration/errorHandling.test.ts for error propagation from API client → UI components

### Implementation for User Story 4

- [ ] T046 [P] [US4] Implement parseApiError function in src/lib/errorHandling.ts with defensive Zod parsing for both formats
- [ ] T047 [P] [US4] Create or extend src/components/ErrorDisplay.tsx to handle ParsedError type with field-level rendering
- [ ] T048 [US4] Update ErrorDisplay to show hint text in muted color below main message
- [ ] T049 [US4] Add collapsible "Technical Details" section in ErrorDisplay for error_code, request_id, and full error object (hidden by default)
- [ ] T050 [US4] Integrate ErrorDisplay into all API error scenarios in CreateNewJob, CreateJobDetail, CreateJobs, and CreateSettings pages
- [ ] T051 [US4] Update toast notifications to include hint text when available (for ErrorEnvelope responses)
- [ ] T052 [US4] Add error boundary component to gracefully handle React rendering errors with ErrorDisplay

**Checkpoint**: Error handling unified - all errors display consistently with helpful details, hints, and accessible technical info

---

## Phase 7: User Story 5 - Enhanced Health and Diagnostics (Priority: P3)

**Goal**: Display comprehensive server health information in collapsible settings section with auto-refresh

**Independent Test**: Navigate to settings, expand diagnostics section, verify display of server version, GPU status, worker info, and diagnostics with 30s auto-refresh

### Tests for User Story 5

> **Write these tests FIRST, ensure they FAIL before implementation**

- [ ] T053 [P] [US5] Create unit test in src/test/api/client.v1.3.test.ts for enhanced getHealth response parsing with v1.3.0 fields
- [ ] T054 [P] [US5] Create component test in src/test/components/ServerDiagnostics.test.tsx for diagnostics display, auto-refresh, manual refresh, and error states
- [ ] T055 [P] [US5] Create integration test in src/test/integration/serverDiagnostics.test.ts for full diagnostics flow including cache and refresh behavior

### Implementation for User Story 5

- [ ] T056 [P] [US5] Update getHealth method in src/api/client.ts to parse enhanced v1.3.0 response fields (gpu_status, worker_info, diagnostics)
- [ ] T057 [P] [US5] Create src/components/ServerDiagnostics.tsx with Collapsible component (shadcn/ui) for expandable diagnostics section
- [ ] T058 [US5] Implement GPU info display in ServerDiagnostics (availability, CUDA version, memory as percentage)
- [ ] T059 [US5] Implement worker info display in ServerDiagnostics (active jobs, max concurrent, queue depth with color-coded status)
- [ ] T060 [US5] Implement diagnostics display (database, storage, FFmpeg status with icons)
- [ ] T061 [US5] Add auto-refresh mechanism (React Query with 30s refetchInterval when section is expanded)
- [ ] T062 [US5] Add manual "Refresh Now" button in ServerDiagnostics that triggers immediate refetch
- [ ] T063 [US5] Format uptime as human-readable string (e.g., "3 days, 4 hours") instead of raw seconds
- [ ] T064 [US5] Add "stale data" indicator if last refresh > 2 minutes ago
- [ ] T065 [US5] Integrate ServerDiagnostics into src/pages/CreateSettings.tsx in collapsible section

**Checkpoint**: Server diagnostics fully functional - advanced users can view comprehensive health info with auto-refresh

---

## Phase 8: Backward Compatibility & Testing

**Purpose**: Ensure v1.2.x server support and comprehensive test coverage

- [ ] T066 [P] Create backward compatibility test suite in src/test/integration/backwardCompat.test.ts with mocked v1.2.x responses
- [ ] T067 [P] Test capability detection with v1.2.x health endpoint (missing gpu_status/worker_info fields)
- [ ] T068 [P] Test graceful degradation when cancellation endpoint returns 404 (hide cancel buttons for v1.2.x)
- [ ] T069 [P] Test validation endpoint fallback (disable validation UI if endpoint returns 404)
- [ ] T070 [P] Test legacy error format parsing (string and {error: "..."} formats)
- [ ] T071 Create E2E test in e2e/server-v1-3.spec.ts for job cancellation flow with real VideoAnnotator v1.3.0 server
- [ ] T072 [P] Create E2E test for configuration validation flow with invalid config
- [ ] T073 [P] Create E2E test for authentication setup and token validation
- [ ] T074 [P] Create E2E test for server diagnostics display and refresh
- [ ] T075 Run full test suite and ensure 100% pass rate (bun run test:run && bun run e2e)

**Checkpoint**: All tests passing, backward compatibility verified

---

## Phase 9: Documentation & Polish

**Purpose**: Update documentation, ensure code quality, and prepare for release

- [ ] T076 Update CHANGELOG.md with new features for v0.5.0 (job cancellation, config validation, enhanced auth, improved errors, diagnostics)
- [ ] T077 [P] Update docs/CLIENT_SERVER_COLLABORATION_GUIDE.md with v1.3.0 API endpoint documentation and usage examples
- [ ] T078 [P] Add JSDoc comments to all new functions, hooks, and components
- [ ] T079 [P] Update README.md with v1.3.0 feature descriptions if user-visible changes warrant it
- [ ] T080 Run linter and fix all issues (bun run lint)
- [ ] T081 Run TypeScript compiler and ensure zero errors (bunx tsc --noEmit)
- [ ] T082 Run production build and verify bundle size increase < 50KB gzipped (bun run build && analyze bundle)
- [ ] T083 Run Lighthouse CI and ensure performance score > 90, accessibility score 100 (bun run lhci)
- [ ] T084 [P] Review all console warnings/errors in dev mode and fix
- [ ] T085 [P] Test application in Chrome, Firefox, Edge (ensure cross-browser compatibility)
- [ ] T086 Create PR description summarizing changes, linking to spec, and listing QA verification steps

**Checkpoint**: Feature complete, documented, tested, and ready for code review

---

## Task Dependencies & Execution Order

### Critical Path (Sequential Phases)

1. **Phase 1** (Setup) → **Phase 2** (Foundational) must complete first
2. **Phase 2** enables parallel execution of **Phase 3-7** (user stories)
3. **Phase 8** (Testing) can run in parallel with **Phase 3-7** as stories complete
4. **Phase 9** (Documentation) must wait for all implementation to complete

### User Story Independence

- **US1** (Job Cancellation) ✅ Independent - can be implemented/tested alone
- **US2** (Config Validation) ✅ Independent - can be implemented/tested alone
- **US3** (Enhanced Auth) ⚠️ Depends on US1/US2 for full integration testing but core implementation is independent
- **US4** (Error Handling) ⚠️ Cross-cutting concern - affects US1, US2, US3 but can be implemented independently
- **US5** (Diagnostics) ✅ Independent - can be implemented/tested alone

### Parallel Execution Opportunities

**After Phase 2 completes**, these can run in parallel:

- **Team A**: US1 (Job Cancellation) - Tasks T010-T020
- **Team B**: US2 (Config Validation) - Tasks T021-T032
- **Team C**: US3 (Enhanced Auth) - Tasks T033-T042
- **Team D**: US4 (Error Handling) - Tasks T043-T052 (but benefits from US1-US3 context)
- **Team E**: US5 (Diagnostics) - Tasks T053-T065

**Example MVP Scope** (Minimum Viable Product):
- Phase 1 (Setup)
- Phase 2 (Foundational)
- Phase 3 (US1 - Job Cancellation) only
- Phase 8 (Testing for US1)
- Phase 9 (Documentation)
= **~10-12 hours of work**, delivers immediate value

---

## Implementation Strategy

### Recommended Approach: Incremental MVP Delivery

1. **Sprint 1** (MVP): Phase 1 → Phase 2 → Phase 3 (US1) → Partial Phase 8 (US1 tests) → Deploy
   - **Value**: Users can cancel jobs immediately
   - **Effort**: ~10-12 hours

2. **Sprint 2**: Phase 4 (US2) → Partial Phase 8 (US2 tests) → Deploy
   - **Value**: Configuration validation prevents wasted processing
   - **Effort**: ~8-10 hours

3. **Sprint 3**: Phase 5 (US3) + Phase 6 (US4) → Partial Phase 8 → Deploy
   - **Value**: Better auth UX and error handling across all features
   - **Effort**: ~6-8 hours

4. **Sprint 4**: Phase 7 (US5) + Phase 8 (complete) + Phase 9 → Deploy
   - **Value**: Advanced diagnostics + full test coverage + documentation
   - **Effort**: ~8-10 hours

**Total Estimated Effort**: 32-40 hours (1-2 weeks for single developer, 3-5 days for 2-person team)

---

## Testing Strategy

### Test-First Workflow (Recommended)

For each user story:
1. Write tests (T0XX tasks) for expected behavior
2. Run tests → verify they FAIL (red)
3. Implement feature (T0XX+N tasks)
4. Run tests → verify they PASS (green)
5. Refactor if needed
6. Move to next user story

### Test Coverage Goals

- **Unit Tests**: >80% coverage for new code
- **Integration Tests**: All user story workflows covered
- **E2E Tests**: Critical paths (job cancellation, config validation, auth setup)
- **Backward Compat Tests**: All v1.2.x scenarios verified

### Test Execution

```bash
# Unit/integration tests (watch mode during development)
bun test

# Single run with coverage
bun run test:coverage

# E2E tests (requires VideoAnnotator v1.3.0 server running)
bun run e2e

# Interactive test UI
bun run test:ui
```

---

## Summary

- **Total Tasks**: 86 tasks
- **Setup**: 3 tasks (Phase 1)
- **Foundational**: 6 tasks (Phase 2)
- **US1 (Job Cancellation)**: 11 tasks (P1 - MVP candidate)
- **US2 (Config Validation)**: 12 tasks (P1 - MVP candidate)
- **US3 (Enhanced Auth)**: 10 tasks (P2)
- **US4 (Error Handling)**: 10 tasks (P2)
- **US5 (Diagnostics)**: 13 tasks (P3)
- **Testing**: 10 tasks (Phase 8)
- **Documentation**: 11 tasks (Phase 9)

**Parallel Opportunities**: 45+ tasks marked with [P] can run in parallel (52% of total)

**Independent Testing**: Each user story (US1-US5) includes clear independent test criteria and can be verified standalone

**MVP Recommendation**: Implement US1 (Job Cancellation) first for immediate user value, then US2 (Config Validation) for efficiency gains

**Next Steps**: Start with Phase 1 (Setup) → Phase 2 (Foundational) → Choose US1 or US2 for first sprint

---

**Tasks Document Complete**: Ready for implementation! 🚀
