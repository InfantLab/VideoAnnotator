# QA Checklist for v0.5.0

Manual testing checklist for VideoAnnotator v1.3.0 support features.

**Test Date:** _____________  
**Tester:** _____________  
**Environment:** [ ] Local Dev [ ] Production Build [ ] Deployed  
**Server Version:** [ ] v1.2.x [ ] v1.3.0+  
**Browser:** [ ] Chrome [ ] Firefox [ ] Safari [ ] Edge

---

## Pre-Testing Setup

### Environment Verification
- [x] Server is running (http://localhost:18011 or configured URL)
- [x] Server version confirmed via `/api/v1/system/health`
- [x] API token configured (env or localStorage)
- [x] Demo videos available in `demo/videos/` directory
- [x] Frontend dev server running (`bun run dev`)

### Known Issues to Note
- [x] Reviewed CHANGELOG.md Known Issues section
- [x] 8/24 ServerDiagnostics timing tests may fail (does not affect functionality)

---

## Onboarding & Connection Error Handling

### Initial Connection Guidance
- [ ] **Stop server** - With server offline, navigate to `/create`
- [ ] **Connection banner** appears at top of page with clear error message
- [ ] **User-friendly troubleshooting** displayed:
  - [ ] Step 1: Check server is running (with explanation of what to look for)
  - [ ] Step 2: Verify server URL (with link to settings)
  - [ ] Step 3: Test URL in browser (clickable link to /health endpoint)
  - [ ] Step 4: Server restart instructions with TWO options:
    - [ ] Standard mode: `uv run videoannotator server` (works for common ports automatically)
    - [ ] Dev mode: `uv run videoannotator server --dev` (allows all origins for testing)
    - [ ] Explanation that standard mode works for React/Vite/Vue/Angular ports
    - [ ] Clear note that dev mode is for testing/custom ports
  - [ ] NO mention of raw "CORS" jargon without context
  - [ ] NO complex `--allow-origins` flags (server handles this automatically now)
- [ ] **Action buttons work**:
  - [ ] "Check API Settings" → navigates to Settings tab
  - [ ] "Retry Connection" → retries without page reload
  - [ ] "Server Setup Guide" → opens docs in new tab
- [ ] **Test the /health link** - Click the health endpoint link, opens server health in new tab
- [ ] **Browser console** - Check F12 console for original error details (for debugging)
- [ ] **Retry works** - Start server with `uv run videoannotator server` or `--dev`, click retry, banner disappears

### ServerDiagnostics CORS Error
- [ ] Navigate to `/create/settings` with server offline
- [ ] ServerDiagnostics component shows simplified error:
  - [ ] "Cannot connect to server" heading
  - [ ] Shows server URL in code block
  - [ ] "Quick fixes" section with 3 actionable items
  - [ ] NO technical jargon like "CORS policy" or "Access-Control-Allow-Origin"
  - [ ] Language is beginner-friendly
  - [ ] "Retry Connection" button available
- [ ] Start server with `uv run videoannotator server` → click refresh → diagnostics load successfully
- [ ] Test with dev mode: `uv run videoannotator server --dev` → works on any port
- [ ] No console errors about missing error handling

### Token Setup Onboarding
- [ ] First visit with no token configured
- [ ] Token status indicator shows "Disconnected" (red)
- [ ] Click indicator → token setup dialog opens
- [ ] Dialog provides clear instructions
- [ ] Save valid token → indicator updates to "Connected" (green)

---

## US1: Job Cancellation (T014-T024)

### Basic Cancellation Flow
- [ ] **T014**: Navigate to Jobs page, verify "Cancel" button appears for in-progress jobs
- [ ] **T015**: Click "Cancel" on an in-progress job
- [ ] **T016**: Confirmation dialog appears with job ID/name
- [ ] **T017**: Click "Cancel" in dialog - job status updates to "cancelling"
- [ ] **T018**: Wait for server to confirm - status updates to "cancelled"
- [ ] **T019**: Toast notification shows "Job cancelled successfully"
- [ ] **T020**: Verify cancelled job no longer shows "Cancel" button

### Error Handling
- [ ] **T021**: Attempt to cancel already-completed job (should show appropriate message)
- [ ] **T022**: Cancel with network error (offline/server down) - error toast appears
- [ ] **T023**: Cancel with server error (500) - error details displayed
- [ ] **T024**: Rapid double-click "Cancel" - only one request sent

### Edge Cases
- [ ] Cancel job immediately after creation (within first few seconds)
- [ ] Cancel job that's 90% complete
- [ ] Cancel multiple jobs in sequence (not parallel)
- [ ] Cancel with browser console open - check for console errors (should be none)

---

## US2: Configuration Validation (T025-T036)

### Real-Time Validation
- [ ] **T025**: Navigate to Create Job page
- [ ] **T026**: Select a pipeline from dropdown
- [ ] **T027**: Enter invalid configuration (e.g., negative number, empty required field)
- [ ] **T028**: Validation error appears within 500ms (debounced)
- [ ] **T029**: Error message is clear and actionable
- [ ] **T030**: Fix the error - validation clears automatically

### Field-Level Validation
- [ ] **T031**: Test each required field with empty value
- [ ] **T032**: Test numeric fields with:
  - [ ] Negative numbers (if invalid)
  - [ ] Zero (if invalid)
  - [ ] Very large numbers (e.g., 999999)
  - [ ] Decimal values (if invalid)
  - [ ] Non-numeric text (e.g., "abc")
- [ ] **T033**: Test string fields with:
  - [ ] Empty string
  - [ ] Very long string (>255 chars)
  - [ ] Special characters
- [ ] **T034**: Test boolean fields (toggle switches)

### Submit Validation
- [ ] **T035**: Attempt to submit with validation errors - blocked with clear message
- [ ] **T036**: Submit with valid config - succeeds
- [ ] Submit button disabled while validation in progress
- [ ] Submit button shows loading state during submission

### Pipeline-Specific Validation
- [ ] Test "pose_estimation" pipeline with invalid confidence threshold
- [ ] Test "transcription" pipeline with invalid language code
- [ ] Test "face_detection" pipeline with invalid min_face_size
- [ ] Test "scene_detection" pipeline with invalid threshold

### Performance
- [ ] Type rapidly in validated field - debouncing works (max 1 validation per 500ms)
- [ ] Switch pipelines rapidly - old validations cancelled
- [ ] No memory leaks (check browser DevTools Performance tab)

---

## US3: Enhanced Authentication (T037-T048)

### Token Status Indicator
- [ ] **T037**: Navigate to any page - token status indicator visible in header
- [ ] **T038**: With valid token: Green checkmark, "Connected"
- [ ] **T039**: With invalid token: Red X, "Disconnected"
- [ ] **T040**: Hover over indicator - tooltip shows token status details
- [ ] **T041**: Click indicator - opens token configuration dialog

### Token Configuration Dialog
- [ ] **T042**: Open dialog - current token masked (e.g., "dev-***ken")
- [ ] **T043**: Click "Show" - token revealed in full
- [ ] **T044**: Click "Hide" - token masked again
- [ ] **T045**: Enter new valid token - "Validate" button enabled
- [ ] **T046**: Click "Validate" - validation status shown
- [ ] **T047**: Save valid token - stored in localStorage
- [ ] **T048**: Close and reopen dialog - token persists

### Token Validation Flow
- [ ] Enter valid token → validation succeeds → green checkmark
- [ ] Enter invalid token → validation fails → red X with error message
- [ ] Enter empty token → validation disabled or shows "required" message
- [ ] Change API URL → token re-validated against new URL

### Error Handling
- [ ] Test with server offline - validation fails gracefully
- [ ] Test with network timeout (slow connection) - loading state shown
- [ ] Test with 401 Unauthorized response - clear error message
- [ ] Test with 403 Forbidden response - clear error message

### Persistence and Refresh
- [ ] Set token → refresh page → token still valid
- [ ] Set token → close/reopen browser → token persists
- [ ] Change token in localStorage (DevTools) → indicator updates on page load
- [ ] Token expiration simulation (if server supports) → indicator updates to "Disconnected"

---

## US4: Improved Error Handling (T050-T052)

### Error Display Component
- [ ] **T050**: Trigger API error (e.g., invalid token) - ErrorDisplay component shown
- [ ] **T051**: Error includes:
  - [ ] Clear error message
  - [ ] Actionable hint (e.g., "Check your API token")
  - [ ] Collapsible technical details (error code, request ID)
  - [ ] "Copy" button for technical details
- [ ] **T052**: Click "Copy" - technical details copied to clipboard

### Field-Level Errors
- [ ] Validation error shows field name + specific issue
- [ ] Multiple field errors displayed (not just first one)
- [ ] Field errors highlighted in red
- [ ] Error cleared when field corrected

### Network Errors
- [ ] Server offline → clear "Server unreachable" message
- [ ] Network timeout → "Request timed out" with retry suggestion
- [ ] CORS error → clear explanation (if applicable)

### API Error Codes
- [ ] 400 Bad Request → shows specific validation failures
- [ ] 401 Unauthorized → prompts token configuration
- [ ] 403 Forbidden → explains permission issue
- [ ] 404 Not Found → clear "resource not found" message
- [ ] 409 Conflict → explains conflict (e.g., duplicate job)
- [ ] 429 Too Many Requests → explains rate limit
- [ ] 500 Internal Server Error → generic message with request ID
- [ ] 503 Service Unavailable → suggests retry

### User Experience
- [ ] Errors auto-dismiss after 5 seconds (toasts)
- [ ] Persistent errors remain until user action
- [ ] Error messages use plain language (no jargon)
- [ ] Errors include next steps or actions

---

## US5: Enhanced Diagnostics (T053-T065)

### ServerDiagnostics Component
- [ ] **T056**: Navigate to Create/Settings page
- [ ] **T057**: ServerDiagnostics card visible (collapsed by default)
- [ ] **T058**: Click header to expand - diagnostics load
- [ ] **T059**: Server info displayed:
  - [ ] Server version (e.g., "1.3.0")
  - [ ] Uptime in human-readable format (e.g., "3 days, 4 hours")
  - [ ] Status: healthy/degraded/unhealthy with color indicator

### GPU Information
- [ ] **T060**: GPU section visible (if server has GPU)
- [ ] **T061**: GPU info includes:
  - [ ] Device name (e.g., "NVIDIA GeForce RTX 3090")
  - [ ] CUDA version (e.g., "12.1")
  - [ ] Memory usage percentage
  - [ ] Memory used / total (e.g., "5 GB / 24 GB")
- [ ] GPU status color-coded (green <50%, yellow <80%, red ≥80%)
- [ ] No GPU → "No GPU detected" message shown

### Worker Information
- [ ] **T062**: Worker section displayed
- [ ] Worker info includes:
  - [ ] Active jobs count
  - [ ] Queued jobs count
  - [ ] Max concurrent workers
  - [ ] Worker status: idle/busy/overloaded
- [ ] Worker status color-coded:
  - [ ] Idle: green
  - [ ] Busy: yellow
  - [ ] Overloaded: red

### System Diagnostics
- [ ] **T063**: Diagnostics section displayed
- [ ] Database status: healthy/degraded/unhealthy with icon
- [ ] Storage info:
  - [ ] Disk usage percentage
  - [ ] Used / total space (e.g., "50 GB / 100 GB")
- [ ] FFmpeg status: available/unavailable
- [ ] Each diagnostic has status icon (CheckCircle/AlertTriangle/XCircle)

### Auto-Refresh Functionality
- [ ] **T064**: Expand diagnostics - data auto-refreshes every 30 seconds
- [ ] Wait 30 seconds - "Last updated" timestamp changes
- [ ] Data updates without page reload
- [ ] Collapse diagnostics - auto-refresh pauses
- [ ] Re-expand diagnostics - auto-refresh resumes

### Manual Refresh
- [ ] **T065**: Click "Refresh Now" button
- [ ] Loading spinner appears on button
- [ ] Data refreshes immediately
- [ ] "Last updated" timestamp updates
- [ ] Button re-enabled after refresh completes

### Stale Data Indicator
- [ ] Keep diagnostics open for 3+ minutes
- [ ] After 2 minutes, yellow "Data may be stale" alert appears
- [ ] Manual refresh clears stale indicator
- [ ] Auto-refresh clears stale indicator

### Error Handling
- [ ] Server offline → error message displayed with ErrorDisplay
- [ ] Server timeout → appropriate error message
- [ ] v1.2.x server (no diagnostics) → graceful degradation or clear message
- [ ] API error → retry button available

---

## Backward Compatibility (Phase 8 - DEFERRED)

### v1.2.x Server Compatibility
- [ ] Connect to v1.2.x server
- [ ] Job cancellation: feature-detect and disable if unavailable
- [ ] Config validation: feature-detect and disable if unavailable
- [ ] Server diagnostics: graceful degradation (show basic info only)
- [ ] Health endpoint: fallback to `/health` if `/api/v1/system/health` unavailable
- [ ] No console errors related to missing endpoints

### Feature Detection
- [ ] Check browser console - feature detection logs visible (if debug enabled)
- [ ] Unavailable features hidden or disabled (not just error-prone)
- [ ] Clear messaging: "Feature requires server v1.3.0+"

---

## Cross-Browser Testing

### Chrome (Latest)
- [ ] All US1-US5 features work
- [ ] No console errors
- [ ] Performance acceptable (<100ms interactions)
- [ ] DevTools network tab shows correct API calls

### Firefox (Latest)
- [ ] All US1-US5 features work
- [ ] No console errors
- [ ] Token show/hide works
- [ ] Clipboard copy works

### Safari (Latest)
- [ ] All US1-US5 features work
- [ ] No console errors
- [ ] Date formatting correct
- [ ] Styles render correctly

### Edge (Latest)
- [ ] All US1-US5 features work
- [ ] No console errors
- [ ] All interactions work

---

## Accessibility (WCAG 2.1 AA)

### Keyboard Navigation
- [ ] All interactive elements accessible via Tab key
- [ ] Tab order logical and intuitive
- [ ] Enter/Space activates buttons
- [ ] Escape closes dialogs
- [ ] Focus indicators visible

### Screen Reader (Test with NVDA/JAWS/VoiceOver)
- [ ] Error messages announced
- [ ] Success toasts announced
- [ ] Button states announced (loading, disabled)
- [ ] Form labels associated with inputs
- [ ] Landmarks properly defined (role="alert", etc.)

### Color Contrast
- [ ] Error text meets 4.5:1 contrast ratio
- [ ] Status indicators readable with colorblindness simulation
- [ ] Focus indicators meet 3:1 contrast ratio

### Visual Indicators
- [ ] Color not sole indicator (icons + text)
- [ ] Loading states clear (spinner + text)
- [ ] Status changes visible

---

## Performance

### Page Load
- [ ] Initial page load <3 seconds
- [ ] Time to Interactive <5 seconds
- [ ] No layout shift during load

### Interactions
- [ ] Button clicks respond <100ms
- [ ] Form validation <500ms (debounced)
- [ ] API calls show loading state immediately
- [ ] No UI blocking during network requests

### Memory Usage
- [ ] Open DevTools → Performance → Memory tab
- [ ] Use app for 5 minutes (navigate, create jobs, cancel, validate)
- [ ] No significant memory increase (heap should stabilize)
- [ ] Close all dialogs → memory released

### Network Efficiency
- [ ] Check DevTools Network tab
- [ ] No unnecessary duplicate requests
- [ ] API calls cached when appropriate (React Query)
- [ ] Debouncing prevents request spam

---

## Lighthouse Audit (Optional)

### Run Lighthouse
```bash
bun run lhci
```

### Target Scores
- [ ] Performance: ≥90
- [ ] Accessibility: ≥95
- [ ] Best Practices: ≥90
- [ ] SEO: ≥90

### Common Issues to Check
- [ ] Images optimized (if applicable)
- [ ] Proper meta tags
- [ ] No console errors
- [ ] Accessible names for interactive elements

---

## Regression Testing

### Existing Features (v0.3.x)
- [ ] Video playback works
- [ ] Timeline displays annotations
- [ ] COCO keypoints render correctly
- [ ] WebVTT transcripts display
- [ ] RTTM speaker segments work
- [ ] Scene detection markers visible
- [ ] Face detection (OpenFace3) works
- [ ] File upload and parsing works
- [ ] Multiple annotation types merge correctly

### No Breaking Changes
- [ ] Old job results still viewable
- [ ] Local file loading still works
- [ ] All existing parsers functional
- [ ] Settings persist correctly

---

## Edge Cases and Stress Testing

### Rapid Interactions
- [ ] Rapid button clicks (cancel, validate, refresh)
- [ ] Rapid form input changes
- [ ] Rapid page navigation
- [ ] No UI freezing or crashes

### Large Data
- [ ] Job list with 100+ jobs
- [ ] Very long error messages
- [ ] Long server uptime (e.g., 30+ days)
- [ ] Large GPU memory (e.g., 80GB)

### Boundary Values
- [ ] Uptime: 0 seconds, 59 seconds, 60 seconds, 86400 seconds (1 day)
- [ ] Memory: 0%, 1%, 50%, 99%, 100%
- [ ] Worker count: 0 active, 0 queued, max concurrent
- [ ] Empty job list

### Network Conditions
- [ ] Test with throttled network (Chrome DevTools → Network → Slow 3G)
- [ ] Test with intermittent connectivity (online/offline toggling)
- [ ] Test with high latency (500ms+)

---

## Security Considerations

### Token Handling
- [ ] Token not visible in URL
- [ ] Token not logged to console (in production build)
- [ ] Token stored securely (localStorage, not cookies)
- [ ] Token masked in UI by default

### Input Sanitization
- [ ] XSS attempts in form fields (e.g., `<script>alert('XSS')</script>`)
- [ ] SQL injection attempts in text fields (if applicable)
- [ ] Special characters handled correctly

### HTTPS (Production)
- [ ] All API calls use HTTPS (in production)
- [ ] No mixed content warnings
- [ ] Secure WebSocket (wss://) for SSE if applicable

---

## Documentation Verification

### User-Facing Docs
- [ ] README.md reflects v0.5.0 features
- [ ] CHANGELOG.md accurately describes changes
- [ ] CLIENT_SERVER_COLLABORATION_GUIDE.md includes v1.3.0 endpoints

### Developer Docs
- [ ] AGENTS.md updated with new components/hooks
- [ ] JSDoc comments complete and accurate
- [ ] Test coverage documented

---

## Final Sign-Off

### Pre-Release Checklist
- [ ] All critical bugs fixed
- [ ] All high-priority features working
- [ ] No console errors in production build
- [ ] Tests passing (70+ tests)
- [ ] Linter clean (or documented exceptions)
- [ ] TypeScript compilation successful
- [ ] Build succeeds (`bun run build`)
- [ ] Production preview works (`bun run preview`)

### Tester Notes
```
[Add any observations, issues found, or recommendations here]




```

### Approval
- [ ] **QA Approved** - Ready for release  
  Signed: _____________ Date: _____________

---

**Document Version:** 1.0  
**Last Updated:** 2025-10-29  
**Related Spec:** `specs/001-server-v1-3-support/spec.md`  
**Related Tasks:** `specs/001-server-v1-3-support/tasks.md`
