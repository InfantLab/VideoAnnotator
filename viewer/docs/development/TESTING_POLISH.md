# Testing & CI Polish — JOSS Submission Readiness

Tracked task list for achieving a green test suite, meaningful coverage, and robust CI before JOSS submission.

Last updated: 2026-02-27

---

## 1. Fix Failing Tests — COMPLETE (52/52 fixed, 262/262 passing)

### 1A. API client DNS rewrite drift — 4 failures FIXED
- **File:** `src/test/api/client.v1.3.test.ts`
- **Root cause:** `APIClient` rewrites `localhost` → `127.0.0.1` but tests asserted `localhost` URLs. `isValidToken()` rejects `test-token`.
- **Fix:** Used `http://127.0.0.1:18011` constant, `va_test12345678` token format, mocked localStorage for consistent URL/token.
- [x] Fix URL assertions
- [x] Fix token format

### 1B. `parseApiError` refactor broke field error tests — 2 failures FIXED
- **File:** `src/lib/errorHandling.ts` (source fix)
- **Root cause:** `parseApiError()` had inline Zod parsing that bypassed `parseErrorEnvelope()`, leaving `fieldErrors` unpopulated.
- **Fix:** Restored delegation to `parseErrorEnvelope()` in the Zod success branch.
- [x] Align source for `fieldErrors` handling
- [x] Align source for field-prefixed message format

### 1C. Stub/mock components never test real code — 18 failures FIXED
- **File:** `src/test/components/TokenStatusIndicator.test.tsx` (complete rewrite)
- **File:** `src/test/integration/errorHandling.test.tsx`
- **Root cause:** Tests rendered inline stub components instead of real ones.
- **Fix (TokenStatusIndicator):** Rewrote to import real component, mock `useTokenStatus()` and `useServerCapabilitiesContext()` hooks.
- **Fix (errorHandling):** Added try/catch + `parseApiError()` + state management in test component.
- [x] Rewrite TokenStatusIndicator tests against real component
- [x] Fix errorHandling integration test component

### 1D. Fake timer + userEvent deadlock — 22 failures FIXED
- **File:** `src/test/integration/config-validation.test.tsx`
- **File:** `src/test/components/ServerDiagnostics.test.tsx`
- **File:** `src/test/integration/serverDiagnostics.test.tsx`
- **Root cause:** `vi.useFakeTimers()` froze `waitFor()` polling. Text assertions didn't match actual component output.
- **Fix:** Added `shouldAdvanceTime: true` to fake timers, `advanceTimers` to userEvent setup, updated text matchers, fixed stale data tests with `mockRejectedValue`, used `getAllByText`/`getAllByRole` for multiple matches.
- [x] Fix config-validation timer deadlocks
- [x] Fix ServerDiagnostics component timer deadlocks + text mismatches
- [x] Fix serverDiagnostics integration timer deadlocks + text mismatches

### 1E. Interface mismatch / missing implementation — 6 failures FIXED
- **File:** `src/test/pipeline-integration.test.tsx`
- **File:** `src/test/integration/job-cancellation.test.tsx`
- **File:** `src/test/integration/authentication.test.tsx`
- **Root cause:** `isLoading` false when `enabled: false`, `showErrorToast()` uses `{ title: 'Error' }` not `{ title: 'Cancellation failed' }`, test component didn't use localStorage.
- **Fix (pipeline):** Changed `isLoading` assertion to `false`, passed `{ enabled: true }`, replaced `setTimeout(0)` with `waitFor`.
- **Fix (job-cancellation):** Updated toast assertion to use `expect.objectContaining({ title: 'Error' })`.
- **Fix (authentication):** Added localStorage read/write to test component, used controlled promise for loading state, spied on `localStorage` directly instead of `Storage.prototype`.
- [x] Fix pipeline integration test defaults
- [x] Fix job-cancellation toast assertion
- [x] Fix authentication localStorage and async timing

---

## 2. CI Workflow Hardening

### 2A. Add build verification step
- **Problem:** `vite build` is never run in CI — TypeScript and bundle errors aren't caught.
- **Fix:** Add `bun run build` step after lint, before tests.
- [ ] Add build step to `.github/workflows/tests.yml`

### 2B. Remove `continue-on-error` from test coverage step
- **Problem:** Failing tests silently pass CI.
- **Fix:** Remove `continue-on-error: true` from the coverage step (keep it on e2e/lighthouse which are supplementary).
- [ ] Update `.github/workflows/tests.yml`

### 2C. Add coverage threshold
- **Problem:** No minimum coverage enforced — regressions go unnoticed.
- **Fix:** Add `thresholds` to `vitest.config.ts` coverage config (e.g., 40% statements as a starting floor).
- [ ] Configure coverage thresholds in `vitest.config.ts`

### 2D. Consider build matrix
- **Nice-to-have:** Test on multiple Node/Bun versions for broader compatibility signaling.
- [ ] Evaluate adding Node 18/20 matrix to CI

---

## 3. Expand Test Coverage — High-Value Targets

### 3A. Core parsers (easy unit tests, high impact)
- `src/lib/parsers/coco.ts` — COCO JSON format parser
- `src/lib/parsers/cocoOpenface3.ts` — OpenFace3 COCO variant
- `src/lib/parsers/face.ts` — Face detection parser
- `src/lib/parsers/openface3Parser.ts` — OpenFace3 CSV parser
- [ ] Add `coco.test.ts`
- [ ] Add `face.test.ts`
- [ ] Add `openface3Parser.test.ts`

### 3B. Pure utility functions (easy wins)
- `src/lib/formatters.ts` — Display formatting helpers
- `src/lib/validation.ts` — Input validation functions
- `src/lib/toastHelpers.tsx` — Toast message formatting
- `src/lib/utils.ts` — General utility functions
- [ ] Add `formatters.test.ts`
- [ ] Add `validation.test.ts`

### 3C. Key hooks
- `src/hooks/useTokenStatus.ts` — Token status management
- `src/hooks/useServerCapabilities.ts` — Server capability detection
- `src/hooks/useSystemHealth.ts` — System health monitoring
- [ ] Add tests for critical hooks

### 3D. Context providers
- `src/contexts/ServerCapabilitiesContext.tsx` / `ServerCapabilitiesProvider.tsx`
- [ ] Add ServerCapabilities context tests

---

## 4. E2E Test Expansion

### 4A. Expand Playwright smoke test
- **Current state:** Single test — checks page title loads.
- **Target:** Cover main user flows:
  - [ ] Load demo dataset and verify viewer renders
  - [ ] Upload a file via drag-and-drop
  - [ ] Navigate between pages (Dashboard, Library, Create)

---

## 5. Documentation & Badges

### 5A. Codecov badge
- [ ] Verify Codecov token is configured in GitHub repo secrets
- [ ] Add coverage badge to README if not present

### 5B. CI status badge
- [ ] Ensure CI status badge in README points to correct workflow

---

## Priority Order

1. ~~**Fix 52 failing tests** (Sections 1A–1E) — reviewer will run tests~~ DONE
2. **Add build step to CI** (Section 2A) — proves the app compiles
3. **Remove continue-on-error from coverage** (Section 2B) — ensures green CI means green tests
4. **Add parser tests** (Section 3A) — core contribution of the project
5. **Add utility tests** (Section 3B) — easy coverage wins
6. **Coverage threshold** (Section 2C) — prevent regressions
7. **E2E expansion** (Section 4A) — nice-to-have
8. **Badges & docs** (Section 5) — nice-to-have
