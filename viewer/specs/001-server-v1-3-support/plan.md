# Implementation Plan: VideoAnnotator Server v1.3.0 Client Support

**Branch**: `001-server-v1-3-support` | **Date**: 2025-10-27 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-server-v1-3-support/spec.md`

**Note**: This plan covers client-side integration of VideoAnnotator server v1.3.0 features including job cancellation, configuration validation, enhanced authentication, improved error handling, and server diagnostics.

## Summary

Update the video-annotation-viewer web application to support new VideoAnnotator server v1.3.0 capabilities, providing users with job cancellation controls, real-time configuration validation with helpful error messages, enhanced authentication management with clear status indicators, consistent error handling across all operations, and comprehensive server health diagnostics. The implementation must maintain backward compatibility with v1.2.x servers while progressively enhancing the UI when v1.3.0 features are detected.

**Technical Approach**: Extend existing API client (`src/api/client.ts`) with new endpoints, implement server capability detection via health endpoint inspection, create React components for job cancellation and validation feedback, enhance error handling middleware to parse ErrorEnvelope format, and update UI components to display server diagnostics. Use feature flags based on detected server version to gracefully degrade functionality for older servers.

## Technical Context

**Language/Version**: TypeScript 5.x (strict mode)  
**Primary Dependencies**: React 18, Vite 5, Tailwind CSS, shadcn/ui, Zod (validation), React Router  
**Storage**: Browser localStorage for API token/URL persistence; no database required  
**Testing**: Vitest + @testing-library/react (unit/integration), Playwright (E2E), Lighthouse CI (performance/accessibility)  
**Target Platform**: Modern web browsers (Chrome 90+, Firefox 88+, Safari 14+, Edge 90+)  
**Project Type**: Single-page web application (SPA) with Vite build system  
**Performance Goals**: 
- API responses render within 1 second
- Configuration validation feedback within 1 second of input
- Job cancellation confirmation within 5 seconds
- Initial page load < 3 seconds on 3G connection
- Lighthouse performance score > 90

**Constraints**:
- Must maintain backward compatibility with VideoAnnotator v1.2.x servers
- No breaking changes to existing user workflows
- Authentication required mode must work seamlessly
- Bundle size increase < 50KB (gzipped)
- Zero console errors in production builds

**Scale/Scope**:
- Single feature branch (~500-800 LOC additions)
- 5 new API client methods
- 3-4 new React components/hooks
- 8-12 new test files
- Expected completion: 2-3 weeks

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Note**: Project constitution is pending creation. Applying principles from AGENTS.md and existing codebase conventions:

### ✅ Type Safety & Validation
- **Requirement**: Use Zod schemas for all API response validation
- **Status**: PASS - Plan includes Zod schema creation for ErrorEnvelope, validation results, and enhanced health responses
- **Evidence**: Existing `src/lib/validation.ts` pattern will be extended

### ✅ Testing Discipline
- **Requirement**: Add tests for all new code; unit + integration + E2E coverage
- **Status**: PASS - Plan includes comprehensive test strategy with Vitest, Playwright, and API mocking
- **Evidence**: Following existing patterns in `src/test/` directory

### ✅ Format Compatibility
- **Requirement**: Support standard formats with graceful degradation
- **Status**: PASS - Plan includes backward compatibility with v1.2.x servers and feature detection
- **Evidence**: Server version detection strategy outlined in Phase 0 research

### ✅ API Integration First
- **Requirement**: VideoAnnotator API as primary backend; respect existing client patterns
- **Status**: PASS - Extending `src/api/client.ts` with new v1.3.0 endpoints
- **Evidence**: All new endpoints documented in server v1.3.0 update guide

### ✅ Progressive Enhancement
- **Requirement**: Feature detection and graceful degradation for missing capabilities
- **Status**: PASS - Plan includes capability detection via health endpoint and conditional UI rendering
- **Evidence**: FR-005, FR-014 require backward compatibility handling

### ✅ Code Quality Standards
- **Requirement**: Lint/typecheck pass; minimal changes; consistent style
- **Status**: PASS - Following existing TypeScript strict mode, ESLint config, and component patterns
- **Evidence**: Using shadcn/ui components and Tailwind conventions

### ✅ Documentation Discipline
- **Requirement**: Update docs and CHANGELOG with all user-visible changes
- **Status**: PASS - Plan includes CHANGELOG updates and user documentation for new features
- **Evidence**: Phase 1 includes quickstart.md for developers

**Pre-Phase 0 Gate**: ✅ PASS - All principles satisfied; no violations to justify

## Project Structure

### Documentation (this feature)

```text
specs/001-server-v1-3-support/
├── plan.md              # This file (implementation plan)
├── spec.md              # Feature specification (complete)
├── research.md          # Phase 0 output (technical research)
├── data-model.md        # Phase 1 output (data structures)
├── quickstart.md        # Phase 1 output (developer guide)
├── contracts/           # Phase 1 output (API contracts)
│   ├── cancellation.yaml       # Job cancellation endpoint spec
│   ├── validation.yaml         # Config validation endpoints spec
│   ├── error-envelope.yaml     # Enhanced error format spec
│   └── health-enhanced.yaml    # Enhanced health endpoint spec
├── checklists/
│   └── requirements.md  # Spec quality checklist (complete)
└── tasks.md             # Phase 2 output (/speckit.tasks - NOT YET CREATED)
```

### Source Code (repository root)

This is a web application following the existing video-annotation-viewer structure:

```text
src/
├── api/
│   ├── client.ts                    # [EXTEND] Add v1.3.0 endpoints
│   ├── types.ts                     # [NEW] ErrorEnvelope, ValidationResult types
│   └── capabilities.ts              # [NEW] Server capability detection
├── components/
│   ├── JobCancelButton.tsx          # [NEW] Cancel job UI control
│   ├── ConfigValidationPanel.tsx    # [NEW] Validation feedback display
│   ├── ErrorDisplay.tsx             # [EXTEND] Handle ErrorEnvelope format
│   ├── TokenStatusIndicator.tsx     # [EXTEND] Show server version
│   └── ServerDiagnostics.tsx        # [NEW] Health/diagnostics display
├── hooks/
│   ├── useJobCancellation.ts        # [NEW] Job cancel logic
│   ├── useConfigValidation.ts       # [NEW] Config validation logic
│   └── useServerCapabilities.ts     # [NEW] Feature detection hook
├── lib/
│   ├── validation.ts                # [EXTEND] Add v1.3.0 response schemas
│   └── errorHandling.ts             # [NEW] ErrorEnvelope parsing utilities
├── pages/
│   ├── CreateJobDetail.tsx          # [EXTEND] Add cancel button
│   ├── CreateNewJob.tsx             # [EXTEND] Add validation feedback
│   └── CreateSettings.tsx           # [EXTEND] Add diagnostics section
└── types/
    └── api.ts                       # [NEW] v1.3.0 API types

src/test/
├── api/
│   ├── client.v1.3.test.ts          # [NEW] v1.3.0 endpoint tests
│   ├── capabilities.test.ts         # [NEW] Feature detection tests
│   └── errorHandling.test.ts        # [NEW] ErrorEnvelope parsing tests
├── components/
│   ├── JobCancelButton.test.tsx     # [NEW] Cancel button tests
│   ├── ConfigValidationPanel.test.tsx # [NEW] Validation display tests
│   └── ServerDiagnostics.test.tsx   # [NEW] Diagnostics tests
└── integration/
    ├── jobCancellation.test.ts      # [NEW] End-to-end cancel flow
    ├── configValidation.test.ts     # [NEW] End-to-end validation flow
    └── backwardCompat.test.ts       # [NEW] v1.2.x compatibility tests

e2e/
└── server-v1-3.spec.ts              # [NEW] Playwright E2E tests

docs/
└── CLIENT_SERVER_COLLABORATION_GUIDE.md  # [EXTEND] Document v1.3.0 features
```

**Structure Decision**: 

This is a single-page web application using the existing React + TypeScript structure. The implementation extends existing modules rather than creating new top-level directories:

- **API Client Extension**: New v1.3.0 endpoints added to `src/api/client.ts` with capability detection in separate module
- **UI Components**: New components follow shadcn/ui patterns in `src/components/`
- **Hooks**: Custom React hooks encapsulate cancellation, validation, and capability detection logic
- **Type Safety**: Zod schemas in `src/lib/validation.ts`; TypeScript types in `src/types/api.ts`
- **Testing**: Following existing `src/test/` structure with unit, integration, and E2E coverage

No new top-level directories needed; all changes integrate with existing architecture.

## Complexity Tracking

> **No violations to justify** - All changes follow existing patterns and architectural decisions.

---

## Post-Phase 1 Constitution Check

*GATE: Re-evaluate after design artifacts complete.*

### ✅ Design Validates Constitution Compliance

**Review Date**: 2025-10-27  
**Reviewer**: AI Agent (plan generation)  
**Status**: ✅ PASS - All principles satisfied post-design

#### Type Safety & Validation
- ✅ **Evidence**: All API responses validated with Zod schemas (see `data-model.md`)
- ✅ **Evidence**: TypeScript strict mode types defined for all new entities
- ✅ **Evidence**: Runtime validation prevents invalid state

#### Testing Discipline
- ✅ **Evidence**: Comprehensive test plan in `quickstart.md` (16+ test files planned)
- ✅ **Evidence**: Unit, integration, and E2E coverage specified
- ✅ **Evidence**: Backward compatibility test suite for v1.2.x servers

#### Format Compatibility
- ✅ **Evidence**: Defensive parsing handles both v1.3.0 and legacy formats
- ✅ **Evidence**: Feature detection ensures graceful degradation
- ✅ **Evidence**: No breaking changes to existing workflows

#### API Integration First
- ✅ **Evidence**: All endpoints documented in OpenAPI contracts
- ✅ **Evidence**: Extends existing `src/api/client.ts` pattern
- ✅ **Evidence**: Maintains VideoAnnotator as single source of truth

#### Progressive Enhancement
- ✅ **Evidence**: Server capability detection via health endpoint
- ✅ **Evidence**: Conditional UI rendering based on detected features
- ✅ **Evidence**: V1.2.x fallback behavior specified

#### Code Quality Standards
- ✅ **Evidence**: Following existing component patterns (shadcn/ui)
- ✅ **Evidence**: React hooks for logic encapsulation
- ✅ **Evidence**: Consistent naming and file organization

#### Documentation Discipline
- ✅ **Evidence**: `quickstart.md` provides developer guide
- ✅ **Evidence**: API contracts document all endpoints
- ✅ **Evidence**: CHANGELOG and CLIENT_SERVER_COLLABORATION_GUIDE updates planned

**Post-Design Gate**: ✅ PASS - Design artifacts complete; ready for implementation (`/speckit.tasks`)

---

## Phase Summary

### Phase 0: Research ✅ COMPLETE

**Output**: `research.md` (8 technical decisions documented)

**Key Findings**:
- Health endpoint inspection for capability detection
- Defensive Zod parsing for error handling
- Debounced validation to avoid server overload
- Optimistic UI updates for cancellation
- React Query for state management
- Code splitting for performance

**All Unknowns Resolved**: No NEEDS CLARIFICATION items remaining

---

### Phase 1: Design & Contracts ✅ COMPLETE

**Outputs**:
- `data-model.md` - 7 core entities with TypeScript types and Zod schemas
- `contracts/cancellation.yaml` - Job cancellation endpoint specification
- `contracts/validation.yaml` - Configuration validation endpoints
- `contracts/health-enhanced.yaml` - Enhanced health endpoint with diagnostics
- `contracts/error-envelope.yaml` - Standardized error format documentation
- `quickstart.md` - Developer implementation guide with 6-phase checklist
- `.github/copilot-instructions.md` - Updated agent context

**API Endpoints Defined**: 4 new/enhanced endpoints
**Data Entities**: 7 core types with validation rules
**Test Strategy**: Unit + Integration + E2E coverage planned

---

## Next Steps

### For Implementation (Use `/speckit.tasks`)

The planning phase is complete. Generate implementation tasks:

```bash
/speckit.tasks
```

This will create `tasks.md` with:
- Task breakdown organized by user story (P1, P2, P3)
- Parallelizable tasks marked with [P]
- Dependencies clearly specified
- Test-first workflow (write tests → implement → verify)

### For Developers Starting Work

1. **Read**: `quickstart.md` for setup and implementation checklist
2. **Reference**: `data-model.md` for types and schemas
3. **Test Against**: `contracts/*.yaml` for API endpoint specifications
4. **Follow**: Phase-by-phase checklist in `quickstart.md`

### For Reviewers

1. **Spec Quality**: `checklists/requirements.md` (✅ all passed)
2. **Constitution Compliance**: Both pre- and post-design gates passed
3. **Technical Decisions**: `research.md` documents all rationale
4. **API Contracts**: OpenAPI specs in `contracts/` directory

---

## Estimated Effort

| Phase | Effort | Timeline |
|-------|--------|----------|
| **Phase 1**: API Client Extension | 2-3 hours | Day 1 |
| **Phase 2**: React Hooks | 2-3 hours | Day 1-2 |
| **Phase 3**: UI Components | 4-5 hours | Day 2-3 |
| **Phase 4**: Page Integration | 2-3 hours | Day 3-4 |
| **Phase 5**: Testing | 4-6 hours | Day 4-5 |
| **Phase 6**: Documentation | 1-2 hours | Day 5 |
| **Total** | **15-22 hours** | **1-2 weeks** |

**Note**: Timeline assumes one developer working part-time (3-4 hours/day)

---

## Risk Mitigation

| Risk | Mitigation | Status |
|------|------------|--------|
| Server version detection unreliable | Health endpoint structure differs; fallback to feature probing | ✅ Addressed in research.md |
| Error parsing breaks on format changes | Defensive Zod schemas with legacy fallbacks | ✅ Addressed in data-model.md |
| Validation causes server overload | 500ms debouncing + caching by config hash | ✅ Addressed in research.md |
| Cancellation race conditions | Optimistic updates with rollback on error | ✅ Addressed in research.md |
| Bundle size bloat | Code splitting for diagnostics (lazy load) | ✅ Addressed in research.md |
| Backward compatibility breaks | Dual test suites + feature detection | ✅ Addressed in quickstart.md |

---

## Artifacts Summary

### Generated Documents

| Document | Status | Lines | Purpose |
|----------|--------|-------|---------|
| `plan.md` | ✅ Complete | 173 | This file - implementation plan |
| `research.md` | ✅ Complete | ~600 | Technical research and decisions |
| `data-model.md` | ✅ Complete | ~550 | Types, schemas, validation rules |
| `quickstart.md` | ✅ Complete | ~450 | Developer implementation guide |
| `contracts/cancellation.yaml` | ✅ Complete | ~150 | Job cancellation API spec |
| `contracts/validation.yaml` | ✅ Complete | ~250 | Config validation API specs |
| `contracts/health-enhanced.yaml` | ✅ Complete | ~200 | Enhanced health endpoint spec |
| `contracts/error-envelope.yaml` | ✅ Complete | ~180 | Error format documentation |
| `.github/copilot-instructions.md` | ✅ Updated | N/A | Agent context with new tech stack |

### Ready for Generation

| Document | Command | Purpose |
|----------|---------|---------|
| `tasks.md` | `/speckit.tasks` | Implementation task breakdown |

---

## Success Criteria Validation

All success criteria from `spec.md` are achievable with this design:

- **SC-001** (5-second cancellation): ✅ Optimistic updates + async API call
- **SC-002** (1-second validation): ✅ Debounced validation + caching
- **SC-003** (95% auth setup): ✅ Clear UI guidance + token status indicator
- **SC-004** (80% self-resolution): ✅ ErrorEnvelope with hints + field-level errors
- **SC-005** (v1.2.x/v1.3.0 compat): ✅ Feature detection + defensive parsing
- **SC-006** (2-second diagnostics): ✅ Cached health data + auto-refresh
- **SC-007** (zero data loss): ✅ Optimistic updates with rollback + error handling

---

**Plan Complete**: All research resolved, design artifacts generated, and gates passed. Ready for task generation and implementation. 🎯
