# Research: VideoAnnotator Server v1.3.0 Integration

**Feature**: VideoAnnotator Server v1.3.0 Client Support  
**Phase**: 0 (Research & Technical Investigation)  
**Date**: 2025-10-27

## Purpose

This document consolidates technical research for integrating VideoAnnotator server v1.3.0 features into the video-annotation-viewer client. Key areas investigated: server capability detection strategies, error handling patterns, configuration validation approaches, backward compatibility mechanisms, and React state management for new features.

---

## 1. Server Capability Detection Strategy

### Decision: Health Endpoint Inspection with Feature Detection Fallback

**Rationale**: 
- VideoAnnotator v1.3.0 enhanced health endpoint includes server version, GPU status, and worker info
- Can detect v1.3.0 by presence of new fields in health response structure
- Graceful degradation: if new endpoints return 404, assume v1.2.x server
- No dedicated `/version` endpoint needed; leverage existing health checks

**Implementation Approach**:
```typescript
// Pseudo-code for capability detection
interface ServerCapabilities {
  version: string;
  supportsCancellation: boolean;
  supportsValidation: boolean;
  supportsEnhancedErrors: boolean;
  supportsEnhancedHealth: boolean;
}

async function detectServerCapabilities(apiUrl: string): Promise<ServerCapabilities> {
  const health = await fetchHealthEndpoint();
  
  // v1.3.0 health includes gpu_status, worker_info, diagnostics
  const isV13 = 'gpu_status' in health || 'worker_info' in health;
  
  if (isV13) {
    return {
      version: health.version || '1.3.0',
      supportsCancellation: true,
      supportsValidation: true,
      supportsEnhancedErrors: true,
      supportsEnhancedHealth: true,
    };
  }
  
  // Fallback: attempt validation endpoint as feature probe
  const supportsValidation = await probeEndpoint('/api/v1/config/validate');
  
  return {
    version: '1.2.x',
    supportsCancellation: false,
    supportsValidation,
    supportsEnhancedErrors: false,
    supportsEnhancedHealth: false,
  };
}
```

**Alternatives Considered**:
- **Dedicated `/api/v1/version` endpoint**: Rejected - requires server changes; health endpoint is authoritative
- **User agent header inspection**: Rejected - servers don't send version in headers
- **Try-catch on every endpoint**: Rejected - creates unnecessary network overhead and poor UX

**Best Practices Applied**:
- Cache capability detection results in React context (avoid repeated checks)
- Provide manual refresh mechanism in settings for admins
- Log detected capabilities to console (dev mode) for troubleshooting

---

## 2. Error Response Format Handling (ErrorEnvelope)

### Decision: Defensive Parsing with Backward Compatibility

**Rationale**:
- v1.3.0 introduces `ErrorEnvelope` with structured fields: `error_code`, `message`, `field`, `hint`
- v1.2.x returns plain error strings or simple `{error: "message"}` objects
- Must parse both formats to avoid breaking existing error displays

**Implementation Approach**:
```typescript
// Zod schema with fallback
const ErrorEnvelopeSchema = z.object({
  error_code: z.string().optional(),
  message: z.string(),
  field: z.string().optional(),
  hint: z.string().optional(),
  status: z.number().optional(),
  request_id: z.string().optional(),
});

const LegacyErrorSchema = z.union([
  z.string(),
  z.object({ error: z.string() }),
  z.object({ message: z.string() }),
]);

function parseApiError(response: unknown): ParsedError {
  // Try v1.3.0 format first
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
    const msg = typeof legacyResult.data === 'string' 
      ? legacyResult.data 
      : (legacyResult.data as any).error || (legacyResult.data as any).message;
    return {
      message: msg,
      isStructured: false,
    };
  }
  
  return { message: 'An unexpected error occurred', isStructured: false };
}
```

**Alternatives Considered**:
- **Strict v1.3.0 parsing only**: Rejected - breaks v1.2.x compatibility
- **Try-catch with type guards**: Rejected - Zod provides better validation and type safety
- **Separate error handlers per API version**: Rejected - increases code complexity unnecessarily

**Best Practices Applied**:
- Use Zod for runtime validation (matches project conventions)
- Provide `isStructured` flag for conditional UI rendering
- Toast notifications show hint text when available (v1.3.0) but degrade gracefully (v1.2.x)

---

## 3. Configuration Validation Integration

### Decision: Real-time Validation with Debouncing

**Rationale**:
- Form fields trigger validation on blur and submit
- Debounce validation API calls (500ms) to avoid overwhelming server
- Display field-level errors inline; block submit on validation failures
- Cache validation results per configuration hash (avoid redundant calls)

**Implementation Approach**:
```typescript
// React hook with debouncing
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
    }, 500),
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

**Alternatives Considered**:
- **Validate on every keystroke**: Rejected - excessive API calls; poor server load
- **Validate only on submit**: Rejected - poor UX; no early feedback
- **Client-side validation only**: Rejected - validation rules might differ from server

**Best Practices Applied**:
- Debounce with lodash or custom hook (project uses React hooks extensively)
- Display loading spinner during validation (clear user feedback)
- Cache results using `useMemo` and configuration hash
- Provide manual "Validate Now" button for impatient users

---

## 4. Job Cancellation UI/UX Patterns

### Decision: Confirmation Dialog with Optimistic Updates

**Rationale**:
- Cancellation is destructive; require confirmation to prevent accidental clicks
- Show optimistic UI update immediately (status → "cancelling...") for responsiveness
- Revert on API error; show toast notification on success
- Disable cancel button for completed/failed/cancelled jobs

**Implementation Approach**:
```typescript
function useJobCancellation(jobId: string) {
  const [isCancelling, setIsCancelling] = useState(false);
  const queryClient = useQueryClient(); // React Query for cache management
  
  const cancelJob = async () => {
    // Confirmation dialog
    const confirmed = await showConfirmDialog({
      title: 'Cancel Job',
      message: 'Are you sure? This cannot be undone.',
      confirmText: 'Cancel Job',
      cancelText: 'Keep Running',
    });
    
    if (!confirmed) return;
    
    setIsCancelling(true);
    
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
      // Revert optimistic update
      queryClient.invalidateQueries(['job', jobId]);
      toast.error(parseApiError(error).message);
    } finally {
      setIsCancelling(false);
    }
  };
  
  return { cancelJob, isCancelling };
}
```

**Alternatives Considered**:
- **No confirmation dialog**: Rejected - too easy to accidentally cancel long-running jobs
- **Polling for status updates**: Rejected - SSE already handles real-time updates
- **No optimistic updates**: Rejected - feels sluggish; users expect immediate feedback

**Best Practices Applied**:
- shadcn/ui AlertDialog component for confirmation (consistent with project UI)
- React Query for cache invalidation (if used) or manual state updates
- Toast notifications from existing `src/hooks/use-toast.ts`
- Accessible button states (disabled, loading) with aria-labels

---

## 5. Server Diagnostics Display Strategy

### Decision: Collapsible Section in Settings with Auto-Refresh

**Rationale**:
- Advanced feature; not needed for primary workflows; belongs in settings
- Auto-refresh every 30 seconds when section is expanded (stay current)
- Display GPU info, worker status, server version prominently
- Gracefully handle missing data (v1.2.x servers or connection errors)

**Implementation Approach**:
```typescript
function ServerDiagnostics() {
  const [isExpanded, setIsExpanded] = useState(false);
  const { data: health, refetch } = useQuery(
    ['server-health'],
    () => apiClient.getHealth(),
    {
      enabled: isExpanded,
      refetchInterval: isExpanded ? 30000 : false, // 30s when expanded
    }
  );
  
  return (
    <Collapsible open={isExpanded} onOpenChange={setIsExpanded}>
      <CollapsibleTrigger>
        <h3>Server Diagnostics</h3>
      </CollapsibleTrigger>
      <CollapsibleContent>
        {health ? (
          <>
            <DiagnosticItem label="Server Version" value={health.version || 'Unknown'} />
            <DiagnosticItem label="GPU Available" value={health.gpu_status ? 'Yes' : 'No'} />
            <DiagnosticItem label="Active Workers" value={health.worker_info?.active || 0} />
            <Button onClick={() => refetch()}>Refresh Now</Button>
          </>
        ) : (
          <p>Loading diagnostics...</p>
        )}
      </CollapsibleContent>
    </Collapsible>
  );
}
```

**Alternatives Considered**:
- **Dedicated diagnostics page**: Rejected - overkill for single collapsible section
- **No auto-refresh**: Rejected - stale data less useful for troubleshooting
- **Real-time SSE for diagnostics**: Rejected - unnecessary overhead; polling sufficient

**Best Practices Applied**:
- React Query (or SWR) for automatic refetching and caching
- shadcn/ui Collapsible component for consistent accordion UX
- Manual refresh button for immediate updates
- Display "stale data" indicator if last refresh > 2 minutes ago

---

## 6. Backward Compatibility Testing Strategy

### Decision: Dual Test Suites + Mock Server Versions

**Rationale**:
- Must ensure zero regressions when connecting to v1.2.x servers
- Create mock API responses for both v1.2.x and v1.3.0 in tests
- Vitest tests mock fetch/axios at network layer
- Playwright E2E tests use real VideoAnnotator instances (both versions)

**Implementation Approach**:
```typescript
// Vitest test with mocked responses
describe('API Client Backward Compatibility', () => {
  it('handles v1.2.x error format gracefully', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({ error: 'Invalid configuration' }), // v1.2.x format
    });
    
    const result = await apiClient.submitJob(config);
    expect(result.error).toBe('Invalid configuration');
  });
  
  it('handles v1.3.0 ErrorEnvelope format', async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 400,
      json: async () => ({
        error_code: 'INVALID_CONFIG',
        message: 'Invalid configuration',
        field: 'scene_detection.threshold',
        hint: 'Value must be between 0 and 100',
      }),
    });
    
    const result = await apiClient.submitJob(config);
    expect(result.error).toBe('Invalid configuration');
    expect(result.hint).toBe('Value must be between 0 and 100');
  });
});
```

**Alternatives Considered**:
- **Only test v1.3.0 format**: Rejected - risks breaking existing deployments
- **Separate codepaths per version**: Rejected - increases maintenance burden
- **User-selectable "server version" setting**: Rejected - auto-detection is better UX

**Best Practices Applied**:
- MSW (Mock Service Worker) or jest/vitest fetch mocks for network mocking
- Separate test suites: `*.v1.2.test.ts` and `*.v1.3.test.ts` for clarity
- Playwright tests tagged with `@v1.2` and `@v1.3` for selective execution
- CI pipeline runs both test suites against mock servers

---

## 7. React State Management for New Features

### Decision: React Query for Server State + Zustand for UI State (if needed)

**Rationale**:
- Server state (jobs, health, validation results) best handled by React Query
- Provides caching, refetching, optimistic updates, and error handling out of box
- UI state (modals, expanded sections) handled by component-local useState
- If global UI state needed (e.g., "show diagnostics on all pages"), use Zustand (lightweight)

**Implementation Approach**:
```typescript
// React Query setup for job data
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30000, // 30s
      cacheTime: 300000, // 5m
      refetchOnWindowFocus: true,
    },
  },
});

// Custom hook for job with cancellation
function useJob(jobId: string) {
  const query = useQuery(['job', jobId], () => apiClient.getJob(jobId));
  const cancelMutation = useMutation(
    () => apiClient.cancelJob(jobId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries(['job', jobId]);
        queryClient.invalidateQueries(['jobs']); // Refresh list
      },
    }
  );
  
  return {
    job: query.data,
    isLoading: query.isLoading,
    error: query.error,
    cancelJob: cancelMutation.mutate,
    isCancelling: cancelMutation.isLoading,
  };
}
```

**Alternatives Considered**:
- **Redux/Redux Toolkit**: Rejected - overkill for this feature; React Query handles server state better
- **Context API only**: Rejected - lacks caching, refetching, and optimistic update patterns
- **Manual fetch + useState**: Rejected - reinvents wheel; React Query is standard for API state

**Best Practices Applied**:
- React Query v4+ (matches modern React practices)
- Custom hooks encapsulate queries and mutations (separation of concerns)
- Optimistic updates for cancellation (better perceived performance)
- Cache invalidation strategies (keep list and detail views in sync)

---

## 8. Bundle Size & Performance Considerations

### Decision: Lazy Load Diagnostics + Code Splitting

**Rationale**:
- Server diagnostics is P3 (nice-to-have); shouldn't bloat initial bundle
- Use React.lazy() and dynamic imports for diagnostics components
- Validation and cancellation are P1; must be in main bundle
- Target: <50KB increase in main bundle (gzipped)

**Implementation Approach**:
```typescript
// Lazy load diagnostics component
const ServerDiagnostics = React.lazy(() => import('./components/ServerDiagnostics'));

// In settings page
function SettingsPage() {
  return (
    <div>
      {/* Other settings */}
      <Suspense fallback={<Skeleton />}>
        <ServerDiagnostics />
      </Suspense>
    </div>
  );
}
```

**Alternatives Considered**:
- **Bundle everything in main chunk**: Rejected - increases initial load time
- **Lazy load all new features**: Rejected - cancellation/validation are critical path
- **Preload diagnostics on settings page mount**: Considered - implement if users complain about load delay

**Best Practices Applied**:
- Vite's built-in code splitting (automatic chunk generation)
- Lighthouse performance monitoring in CI
- Bundle analyzer to verify size constraints
- Suspense boundaries with loading skeletons (shadcn/ui Skeleton component)

---

## Summary of Technical Decisions

| Area | Decision | Key Benefit |
|------|----------|------------|
| **Capability Detection** | Health endpoint inspection | No server changes needed; graceful degradation |
| **Error Handling** | Defensive parsing with Zod | Backward compatible; type-safe |
| **Config Validation** | Debounced real-time validation | Good UX without server overload |
| **Job Cancellation** | Confirmation dialog + optimistic updates | Prevents accidents; feels responsive |
| **Server Diagnostics** | Collapsible section with auto-refresh | Advanced feature doesn't clutter UI |
| **Backward Compatibility** | Feature detection + dual test suites | Zero regressions on v1.2.x servers |
| **State Management** | React Query for server state | Industry standard; reduces boilerplate |
| **Performance** | Lazy load diagnostics | Keeps main bundle small |

---

## Open Questions & Risks Mitigated

### Risks Addressed
1. ✅ **Server version detection reliability** → Health endpoint structure differs between versions
2. ✅ **Error parsing fragility** → Defensive Zod schemas with fallbacks
3. ✅ **Validation performance** → Debouncing and caching strategies
4. ✅ **Cancellation race conditions** → Optimistic updates with rollback
5. ✅ **Bundle size bloat** → Code splitting for non-critical features

### Remaining Considerations
- **Network resilience**: Implement retry logic with exponential backoff for transient failures
- **Accessibility**: Ensure all new components meet WCAG 2.1 AA standards (test with Lighthouse)
- **Internationalization**: Error messages and hints are English-only (acceptable for v1; i18n is out of scope)

---

**Research Complete**: All technical unknowns resolved. Ready for Phase 1 (Design & Contracts).
