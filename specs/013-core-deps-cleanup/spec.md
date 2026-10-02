# Feature Specification: Core Dependency Clean-up and Tooling

**Feature Branch**: `1.6-dev` (v1.6.0 development branch)
**Created**: 2026-10-01
**Status**: Implemented 2026-10-01 on `1.6-dev` (81d1221); CI passes
**Input**: User description: "Core dependency clean-up and tooling (v1.6.0 Phase 1, spec 2 of the dependency audit's sequence). Remove core dependencies nothing imports and the dead visualization package; remove unused extras dependencies and the empty `annotation` extra; move numba to the `audio` extra; upgrade core dependencies within constraints, with tests and the pipeline-output comparison as the guard; upgrade pre-commit hooks, make the mypy hook run CI's check, drop dead hooks, declare dev tools once, and move GitHub Actions off deprecated versions. Core install must get smaller; no pipeline output may change."

**Background**: [`docs/development/dependency_audit_v1.6.0.md`](../../docs/development/dependency_audit_v1.6.0.md)
§3 (core), §4 (extras), §6 (tooling). Twelve core dependencies are imported nowhere in the
package; the core lock dates from 2025; the pre-commit type check excludes most of the package and
missed errors CI later caught.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A smaller, faster core install (Priority: P1)

A researcher installs VideoAnnotator with only the pipelines they need. The core install, which
every user gets, downloads and installs only what the core actually uses.

**Why this priority**: v1.5.0's promise was "slim by default"; a dozen unused packages (including
a plotting library, a spreadsheet library and an image-processing suite) undercut it on every
install, in every Docker image and every CI run.

**Independent Test**: install the core alone in a clean environment before and after; compare
package count and size; start the server and list pipelines.

**Acceptance Scenarios**:

1. **Given** a clean environment, **When** the user installs the core only, **Then** the install
   has fewer packages and takes less disk than before, and the server, CLI and viewer work as
   before.
2. **Given** a core install, **When** the user adds any pipeline extra, **Then** that pipeline
   works exactly as before (a dependency moved from core to an extra is still there when needed).

---

### User Story 2 - Current, secure core libraries without changed results (Priority: P1)

A lab installing v1.6.0 gets current releases of the web server, database and data libraries, and
their existing results are reproduced exactly.

**Why this priority**: year-old web and crypto-adjacent libraries accumulate security fixes the
project doesn't have; results changing silently would break the labs' comparisons (constitution
principles III and V).

**Independent Test**: re-lock, run the full suite, and compare every pipeline's output on the demo
video before and after.

**Acceptance Scenarios**:

1. **Given** the upgraded lock, **When** the test suite runs on every supported platform and Python,
   **Then** it passes as before.
2. **Given** the demo video, **When** every pipeline runs before and after the upgrade, **Then** the
   outputs are identical (or within the run-to-run variation the pipeline already has).
3. **Given** a library major version that breaks a test or an output, **Then** it is held back with
   its reason recorded next to the constraint, not upgraded.

---

### User Story 3 - Checks before commit match CI (Priority: P2)

A maintainer committing a change gets the same type-check verdict from the pre-commit hooks as
from CI, so a change doesn't pass locally and fail on push.

**Why this priority**: three type errors reached the release unnoticed because the local hook
checked a fraction of the package with a two-year-old type checker.

**Independent Test**: introduce a type error in a module the current hook excludes (for example a
pipeline); commit; the hook must reject it, as CI would.

**Acceptance Scenarios**:

1. **Given** a type error anywhere in the package, **When** the maintainer commits, **Then** the
   pre-commit hook reports it.
2. **Given** the hook configuration, **Then** it has no hooks that check nothing (wrong paths) or
   that depend on an unmaintained source.

---

### User Story 4 - CI on supported action versions (Priority: P3)

CI runs without deprecation warnings about its own building blocks, so it won't break when those
are removed.

**Independent Test**: a CI run shows no Node 20 deprecation warnings.

**Acceptance Scenarios**:

1. **Given** a push, **When** CI runs, **Then** no step warns that its action runs on a deprecated
   runtime.

### Edge Cases

- A package "nothing imports" may still be needed at runtime by name (FastAPI needs
  `python-multipart` for uploads without importing it). Each removal is checked against runtime
  behaviour, not only imports; `python-multipart` stays.
- A dependency removed from core but still needed transitively by an extra (for example `tqdm`
  via Whisper) must keep arriving through that extra.
- Users or scripts that import the removed `videoannotator.visualization` package: it can't be
  imported today (broken import path), so removal breaks nothing that works.
- `version.py`'s dependency report names some removed packages; it must not error when they're
  absent.
- Windows keeps its existing CI allowance.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The core install MUST NOT include packages the core doesn't use: the twelve listed in
  the audit (§3) unless a runtime need is found, in which case the need is recorded.
- **FR-002**: The dead `visualization` package MUST be removed.
- **FR-003**: Extras MUST NOT declare packages their pipelines don't use (`imutils`, `supervision`);
  the empty `annotation` extra MUST be removed (and from `all`).
- **FR-004**: `numba` MUST move from core to the `audio` extra.
- **FR-005**: Core dependencies MUST be upgraded to their latest releases that pass the test suite
  and the output comparison; any held back MUST carry its reason next to its constraint.
- **FR-006**: Every pipeline's output on the demo video MUST be unchanged by this spec (identical,
  or within existing run-to-run variation), checked with `scripts/compare_pipeline_outputs.py`.
- **FR-007**: The pre-commit type check MUST check the same files with the same configuration and
  type-checker version as CI.
- **FR-008**: Pre-commit hooks that check nothing or depend on unmaintained sources MUST be removed;
  the rest MUST be upgraded to current releases.
- **FR-009**: Development tools MUST be declared in one place.
- **FR-010**: GitHub Actions MUST be on versions that don't run on a deprecated runtime.
- **FR-011**: The CHANGELOG MUST list removed core dependencies, so anyone who relied on one being
  present transitively knows to add it.

### Key Entities

- **Core dependency set**: what every install gets. Shrinks.
- **Extras**: per-pipeline dependency groups. Gain `numba` (audio); lose `imutils`, `supervision`;
  `annotation` removed.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: The core-only install has at least 20% fewer packages and uses less disk than before
  (measured in a clean environment, before and after).
- **SC-002**: The test suite passes on every supported platform and Python version, with no fewer
  passing tests than before (except tests of removed dead code).
- **SC-003**: Every pipeline's output on the demo video is identical before and after, or within
  the run-to-run variation measured beforehand.
- **SC-004**: A type error introduced in any package module is caught by the pre-commit hook.
- **SC-005**: A CI run produces no deprecated-runtime warnings from its actions.

## Assumptions

- Removing a core dependency that a user imported in their own scripts (relying on it arriving with
  VideoAnnotator) is acceptable in a minor release when listed in the CHANGELOG; the core never
  promised them.
- "Upgrade within constraints" covers core and dev tools only. torch, pyannote, transformers,
  opencv and the pipeline libraries are later specs.
- The pipeline-output comparison runs on the demo video on Linux with GPU, as for spec 012.
