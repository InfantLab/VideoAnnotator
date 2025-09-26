# Manual QA Testing – v0.4.0 Pipeline Integration

Target build: v0.4.0 (dynamic VideoAnnotator v1.2.x integration)
Last updated: 2025-09-18

## Test Environment
- VideoAnnotator server v1.2.1 or v1.2.2 running at `http://localhost:18011`
- `VITE_API_BASE_URL` and `VITE_API_TOKEN` configured (or stored via app settings)
- Fresh browser session (clear localStorage to test onboarding flows)

## Pipeline Discovery & Diagnostics
- [ ] Load Settings → confirm “Server Pipelines” diagnostics card renders with server version, catalog timestamp, and feature flags
- [ ] Click “Refresh Catalog” → catalog refetches without full page reload
- [ ] Disconnect server (stop VA process) → UI shows reachable error, cached catalog marked as stale
- [ ] Reconnect server → “Refresh Catalog” restores live status without stale banner
- [ ] Verify “Clear Pipeline Cache” (or equivalent) removes local cache and forces full refetch next load

## Job Wizard – Dynamic Catalog
- [ ] Open “Create New Job” → pipeline list populated from server catalog (no static defaults)
- [ ] Pipelines grouped correctly (e.g., Face, Person, Audio, Scenes); descriptions/version labels match server response
- [ ] Toggle pipeline enabled state → selection persists across wizard steps
- [ ] Pipelines disabled when server returns `defaultEnabled = false` and no toggle applied
- [ ] Catalog updates reflected after manual refresh without reloading the page

## Parameter Forms & Validation
- [ ] For pipelines that expose schemas, parameter controls reflect server types (switch, slider, select, text)
- [ ] Required parameters enforce input before submission (inline errors)
- [ ] Min/max/step constraints applied to numeric sliders/inputs
- [ ] Enum options show labels/descriptions when provided
- [ ] Invalid inputs block job submission with helpful message referencing field name
- [ ] Submit valid configuration → payload sent without client-side errors

## Viewer Capability Awareness
- [ ] Load job results containing specific pipelines → related overlay toggles enabled
- [ ] Pipelines not run in the job show disabled toggles with “Not available” helper text
- [ ] OpenFace3 controls reflect server-declared capabilities (hide features server marks as disabled)
- [ ] Switching between jobs updates overlay availability without refresh
- [ ] Attempt to enable overlay lacking data → viewer surfaces warning instead of silent failure

## Artifact Expectations & Results Mapping
- [ ] Viewer fetches expected artifact types per pipeline (COCO/WebVTT/RTTM/etc.) based on catalog metadata
- [ ] Missing artifact handled gracefully with user-facing notification
- [ ] “Open in Viewer” from job list pre-selects overlays matching pipelines actually run

## Fallback & Degraded Mode
- [ ] Point app to pre-1.2.1 server (or simulate by disabling catalog endpoint) → UI shows limited mode banner
- [ ] Legacy pipeline list still available; advanced parameter forms hidden/disabled
- [ ] Job submission works with fallback configuration
- [ ] Viewer overlays continue to function using legacy assumptions

## Regression Checks
- [ ] Legacy REST calls (health, job list, submission) still succeed
- [ ] SSE job monitoring unaffected (connect/disconnect, progress events)
- [ ] Demo data loading unaffected
- [ ] Unit tests (`bun run test:run`) and E2E smoke (`bun run e2e`) pass

## Notes & Findings
- Record screenshots of new diagnostics card and dynamic pipeline forms
- Capture console/network logs for any failed requests or validation issues
- File issues referencing specific pipeline IDs and schema payloads when mismatches occur

