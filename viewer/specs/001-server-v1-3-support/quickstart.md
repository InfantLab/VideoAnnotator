# Quickstart: VideoAnnotator Server v1.3.0 Integration

**Target Audience**: Developers implementing VideoAnnotator v1.3.0 client support  
**Prerequisites**: Familiarity with React, TypeScript, Vite, and existing video-annotation-viewer codebase  
**Estimated Setup Time**: 15 minutes

---

## Overview

This guide walks you through setting up your development environment and implementing the VideoAnnotator server v1.3.0 features. You'll add job cancellation, configuration validation, enhanced authentication, improved error handling, and server diagnostics to the client application.

---

## Quick Setup

### 1. Checkout Feature Branch

```bash
git checkout 001-server-v1-3-support
bun install  # or npm install
```

### 2. Start Local VideoAnnotator Server (v1.3.0)

Ensure you have a running VideoAnnotator v1.3.0 server for testing:

```bash
# In VideoAnnotator repository
git checkout 001-videoannotator-v1-3
uv sync
uv run videoannotator server

# Note the API key printed in console:
# "[INFO] API Key: va_api_abc123..."
```

### 3. Configure Client Environment

Create `.env` file (or update existing):

```bash
VITE_API_BASE_URL=http://localhost:18011
VITE_API_TOKEN=va_api_abc123  # Use key from server console
```

### 4. Start Development Server

```bash
bun run dev  # or npm run dev
# Open http://localhost:3000
```

### 5. Verify Connection

Navigate to Settings page → Server Diagnostics section. You should see:
- Server Version: 1.3.0
- GPU Status: Available (or "Not Available")
- Worker Status: Healthy

---

## Implementation Checklist

Follow this order for smooth development:

### Phase 1: API Client Extension (~2-3 hours)

- [ ] **1.1** Add v1.3.0 types to `src/types/api.ts`
  - Copy interfaces from `specs/001-server-v1-3-support/data-model.md`
  - `ErrorEnvelope`, `ConfigValidationResult`, `JobCancellationResponse`, `EnhancedHealthResponse`

- [ ] **1.2** Add Zod schemas to `src/lib/validation.ts`
  - Import schemas from data-model.md
  - Export for use in API client

- [ ] **1.3** Extend `src/api/client.ts` with new endpoints:
  ```typescript
  // Job cancellation
  async cancelJob(jobId: string): Promise<JobCancellationResponse>
  
  // Configuration validation
  async validateConfig(config: unknown): Promise<ConfigValidationResult>
  async validatePipeline(name: string, config: unknown): Promise<PipelineValidationResult>
  
  // Enhanced health (update existing getHealth method)
  async getHealth(): Promise<EnhancedHealthResponse>
  ```

- [ ] **1.4** Create `src/api/capabilities.ts` for server detection:
  ```typescript
  export async function detectServerCapabilities(apiUrl: string): Promise<ServerCapabilities>
  ```

- [ ] **1.5** Create `src/lib/errorHandling.ts` for ErrorEnvelope parsing:
  ```typescript
  export function parseApiError(response: unknown): ParsedError
  ```

### Phase 2: React Hooks (~2-3 hours)

- [ ] **2.1** Create `src/hooks/useServerCapabilities.ts`:
  - Fetch and cache server capabilities
  - Provide refresh mechanism
  - Export `{ capabilities, isLoading, refresh }`

- [ ] **2.2** Create `src/hooks/useJobCancellation.ts`:
  - Handle confirmation dialog
  - Call cancelJob API
  - Optimistic updates
  - Export `{ cancelJob, isCancelling }`

- [ ] **2.3** Create `src/hooks/useConfigValidation.ts`:
  - Debounced validation (500ms)
  - Cache results by config hash
  - Export `{ validationResult, isValidating, validate }`

### Phase 3: UI Components (~4-5 hours)

- [ ] **3.1** Create `src/components/JobCancelButton.tsx`:
  - AlertDialog for confirmation (shadcn/ui)
  - Disable for non-cancellable states
  - Loading state during cancellation
  - Toast notifications

- [ ] **3.2** Create `src/components/ConfigValidationPanel.tsx`:
  - Display errors/warnings grouped by field
  - Inline field-level error display
  - Show hints in muted color
  - Auto-refresh on config change

- [ ] **3.3** Extend `src/components/ErrorDisplay.tsx`:
  - Handle both ErrorEnvelope and legacy formats
  - Display hint text when available
  - Collapsible technical details
  - Show error codes in dev mode

- [ ] **3.4** Create `src/components/ServerDiagnostics.tsx`:
  - Collapsible section (shadcn/ui)
  - Display GPU info, worker status, diagnostics
  - Auto-refresh every 30s when expanded
  - Manual refresh button

- [ ] **3.5** Extend `src/components/TokenStatusIndicator.tsx`:
  - Show server version
  - Indicate v1.3.0 vs v1.2.x
  - Warning for unsecured connections

### Phase 4: Page Integration (~2-3 hours)

- [ ] **4.1** Update `src/pages/CreateJobDetail.tsx`:
  - Add `<JobCancelButton jobId={id} />` component
  - Conditional rendering based on job status
  - Update SSE handling for cancellation events

- [ ] **4.2** Update `src/pages/CreateNewJob.tsx`:
  - Integrate `<ConfigValidationPanel />` below config editor
  - Disable submit button if validation errors exist
  - Show validation warnings with "Submit Anyway" confirmation

- [ ] **4.3** Update `src/pages/CreateSettings.tsx`:
  - Add `<ServerDiagnostics />` section
  - Position after API token configuration
  - Add "Refresh Capabilities" button

### Phase 5: Testing (~4-6 hours)

- [ ] **5.1** Unit tests for API client:
  - `src/test/api/client.v1.3.test.ts` - New endpoints
  - `src/test/api/capabilities.test.ts` - Feature detection
  - `src/test/api/errorHandling.test.ts` - ErrorEnvelope parsing

- [ ] **5.2** Component tests:
  - `src/test/components/JobCancelButton.test.tsx`
  - `src/test/components/ConfigValidationPanel.test.tsx`
  - `src/test/components/ServerDiagnostics.test.tsx`

- [ ] **5.3** Integration tests:
  - `src/test/integration/jobCancellation.test.ts` - Full cancel flow
  - `src/test/integration/configValidation.test.ts` - Full validation flow
  - `src/test/integration/backwardCompat.test.ts` - v1.2.x compatibility

- [ ] **5.4** E2E tests:
  - `e2e/server-v1-3.spec.ts` - Playwright smoke tests

### Phase 6: Documentation & Finalization (~1-2 hours)

- [ ] **6.1** Update `CHANGELOG.md` with new features
- [ ] **6.2** Update `README.md` if user-visible changes
- [ ] **6.3** Update `docs/CLIENT_SERVER_COLLABORATION_GUIDE.md` with v1.3.0 sections
- [ ] **6.4** Add JSDoc comments to new functions/components
- [ ] **6.5** Run full test suite: `bun test && bun run e2e`
- [ ] **6.6** Run linter: `bun run lint`
- [ ] **6.7** Build check: `bun run build`

---

## Key Implementation Patterns

### Pattern 1: Defensive API Error Parsing

Always handle both v1.3.0 and legacy error formats:

```typescript
function parseApiError(response: unknown): ParsedError {
  // Try v1.3.0 ErrorEnvelope first
  const envelopeResult = ErrorEnvelopeSchema.safeParse(response);
  if (envelopeResult.success) {
    return {
      message: envelopeResult.data.message,
      code: envelopeResult.data.error_code,
      field: envelopeResult.data.field,
      hint: envelopeResult.data.hint,
      isStructured: true,
    };
  }
  
  // Fallback to legacy format
  const legacyResult = LegacyErrorSchema.safeParse(response);
  if (legacyResult.success) {
    // Extract message from various legacy formats
    const msg = typeof legacyResult.data === 'string' 
      ? legacyResult.data 
      : (legacyResult.data as any).error || (legacyResult.data as any).message;
    return { message: msg, isStructured: false };
  }
  
  return { message: 'An unexpected error occurred', isStructured: false };
}
```

### Pattern 2: Server Capability Detection

Check for v1.3.0 fields in health response:

```typescript
async function detectServerCapabilities(): Promise<ServerCapabilities> {
  const health = await fetchHealthEndpoint();
  
  // v1.3.0 includes gpu_status, worker_info
  const isV13 = 'gpu_status' in health || 'worker_info' in health;
  
  if (isV13) {
    return {
      version: health.version || '1.3.0',
      supportsCancellation: true,
      supportsValidation: true,
      supportsEnhancedErrors: true,
      supportsEnhancedHealth: true,
      detectedAt: new Date(),
    };
  }
  
  // Fallback to v1.2.x
  return {
    version: '1.2.x',
    supportsCancellation: false,
    supportsValidation: false,
    supportsEnhancedErrors: false,
    supportsEnhancedHealth: false,
    detectedAt: new Date(),
  };
}
```

### Pattern 3: Debounced Configuration Validation

Avoid overwhelming server with validation requests:

```typescript
function useConfigValidation(config: PipelineConfig) {
  const [validationResult, setValidationResult] = useState<ValidationResult | null>(null);
  const [isValidating, setIsValidating] = useState(false);
  
  const validateDebounced = useMemo(
    () => debounce(async (cfg: PipelineConfig) => {
      setIsValidating(true);
      try {
        const result = await apiClient.validateConfig(cfg);
        setValidationResult(result);
      } catch (error) {
        setValidationResult({ valid: false, errors: [parseApiError(error)] });
      } finally {
        setIsValidating(false);
      }
    }, 500), // 500ms debounce
    []
  );
  
  useEffect(() => {
    if (config && Object.keys(config).length > 0) {
      validateDebounced(config);
    }
  }, [config, validateDebounced]);
  
  return { validationResult, isValidating };
}
```

### Pattern 4: Optimistic Job Cancellation

Provide immediate UI feedback:

```typescript
async function cancelJob(jobId: string) {
  const confirmed = await showConfirmDialog({
    title: 'Cancel Job',
    message: 'Are you sure? This cannot be undone.',
  });
  
  if (!confirmed) return;
  
  // Optimistic update
  queryClient.setQueryData(['job', jobId], (old: Job) => ({
    ...old,
    status: 'cancelling',
  }));
  
  try {
    const result = await apiClient.cancelJob(jobId);
    queryClient.setQueryData(['job', jobId], result);
    toast.success('Job cancelled successfully');
  } catch (error) {
    // Revert on error
    queryClient.invalidateQueries(['job', jobId]);
    toast.error(parseApiError(error).message);
  }
}
```

---

## Testing Guide

### Running Tests

```bash
# Unit tests (watch mode)
bun test

# Unit tests (single run with coverage)
bun run test:coverage

# E2E tests (requires server running)
bun run e2e

# Interactive test UI
bun run test:ui
```

### Test Server Setup

For integration/E2E tests, you'll need both v1.2.x and v1.3.0 server instances:

```bash
# Terminal 1: v1.2.x server
cd ../VideoAnnotator
git checkout main
uv run videoannotator server --port 18011

# Terminal 2: v1.3.0 server
cd ../VideoAnnotator
git checkout 001-videoannotator-v1-3
uv run videoannotator server --port 18012

# Terminal 3: Run tests
cd ../video-annotation-viewer
VITE_API_BASE_URL=http://localhost:18011 bun run e2e  # v1.2.x tests
VITE_API_BASE_URL=http://localhost:18012 bun run e2e  # v1.3.0 tests
```

### Mocking Strategy

Use MSW (Mock Service Worker) or Vitest fetch mocks for unit tests:

```typescript
// Mock v1.3.0 error response
mockFetch.mockResolvedValueOnce({
  ok: false,
  status: 400,
  json: async () => ({
    error_code: 'VALUE_OUT_OF_RANGE',
    message: 'Invalid value',
    field: 'confidence_threshold',
    hint: 'Try using 0.5',
  }),
});
```

---

## Troubleshooting

### Issue: Validation endpoint returns 404

**Cause**: Server is v1.2.x (no validation endpoint)  
**Solution**: Check `useServerCapabilities().supportsValidation` before showing validation UI

### Issue: Cancellation doesn't work

**Cause**: Server is v1.2.x or job is not in cancellable state  
**Solution**: Disable cancel button based on capabilities and job status

### Issue: Error messages not displaying hints

**Cause**: Server returning legacy error format  
**Solution**: `parseApiError` should handle both formats; check `isStructured` flag

### Issue: Tests failing with "Network Error"

**Cause**: VideoAnnotator server not running or wrong port  
**Solution**: Start server and verify `VITE_API_BASE_URL` in tests

### Issue: TypeScript errors after adding types

**Cause**: Missing imports or schema mismatches  
**Solution**: Run `bunx tsc --noEmit` and fix import paths

---

## Performance Checklist

- [ ] Validation debounced (500ms minimum)
- [ ] Server capabilities cached (5-minute TTL)
- [ ] Diagnostics lazy-loaded (`React.lazy`)
- [ ] Optimistic updates for cancellation (no spinner blocking)
- [ ] Health endpoint not called on every page load
- [ ] Bundle size increase < 50KB gzipped
- [ ] Lighthouse performance score > 90

---

## Accessibility Checklist

- [ ] Cancel button has aria-label: "Cancel job"
- [ ] Confirmation dialog keyboard-navigable
- [ ] Validation errors announced to screen readers
- [ ] Loading states have aria-live="polite"
- [ ] Error messages have sufficient color contrast
- [ ] Focus management in dialogs
- [ ] All interactive elements keyboard-accessible

---

## Git Workflow

### Committing Changes

Follow conventional commits:

```bash
git add src/api/client.ts
git commit -m "feat(api): add job cancellation endpoint"

git add src/components/ConfigValidationPanel.tsx
git commit -m "feat(validation): add configuration validation panel"

git add src/test/integration/backwardCompat.test.ts
git commit -m "test(compat): add v1.2.x backward compatibility tests"
```

### Pre-Push Checklist

```bash
bun run lint           # No linting errors
bun run test:run       # All unit tests pass
bun run build          # Production build succeeds
bun run e2e            # E2E tests pass (if server available)
```

---

## Next Steps After Implementation

1. **Create Pull Request**: Reference spec in PR description
2. **Request Code Review**: Tag maintainers
3. **Update Documentation**: Ensure all docs reflect new features
4. **Deploy to Staging**: Test with production-like data
5. **User Acceptance Testing**: Follow QA checklist in spec
6. **Merge to Main**: After approval
7. **Tag Release**: Create v0.5.0 tag (or appropriate version)

---

## Resources

- **Feature Spec**: `specs/001-server-v1-3-support/spec.md`
- **Data Model**: `specs/001-server-v1-3-support/data-model.md`
- **API Contracts**: `specs/001-server-v1-3-support/contracts/*.yaml`
- **Research Doc**: `specs/001-server-v1-3-support/research.md`
- **Server Update Guide**: `docs/development/Server_V1.3.0_client_updating.md`
- **Existing API Client**: `src/api/client.ts`
- **Existing Tests**: `src/test/`

---

**Questions?** Check the research doc or reach out to the maintainer listed in README.md.

**Happy Coding!** 🚀
