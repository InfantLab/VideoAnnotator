# Feature Specification: Python 3.13 Support

**Feature Branch**: `1.6-dev` (v1.6.0 development branch)
**Created**: 2026-10-01
**Status**: Implemented 2026-10-01 on `1.6-dev`; CI passes on 3.12 and 3.13. Docker image builds not yet run (no Docker in the dev container)
**Input**: User description: "Python 3.13 support (v1.6.0 Phase 1, spec 1 of the dependency audit's sequence). Allow VideoAnnotator to install and run on Python 3.13 while keeping Python 3.12 working: widen requires-python to >=3.12,<3.14 with no other library change. CI runs the test suite on both 3.12 and 3.13; the dev container and the Docker images move to 3.13. Update classifiers, lint and type-check targets, and the install docs. Out of scope: any library upgrade, dropping Python 3.12 (v1.7.0), and Python 3.14 (blocked by TensorFlow until the face-stack spec removes it). Still to verify: speaker_diarization on 3.13 with a Hugging Face token."

**Background**: [`docs/development/dependency_audit_v1.6.0.md`](../../docs/development/dependency_audit_v1.6.0.md)
§2 and §7. Python 3.12 has had security fixes only since 2025. A trial on 3.13.15 (2026-10-01)
installed every extra, imported all ten pipelines, passed the test suite (1335 passed, 0 failed)
and ran five pipelines on a real video, with no library version changed.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Install and run on Python 3.13 (Priority: P1)

A researcher whose machine or lab image has Python 3.13 installs VideoAnnotator, with whichever
pipeline extras they need, starts the server and runs jobs. Today the install is refused outright
because only Python 3.12 is accepted.

**Why this priority**: it's the feature. New systems increasingly ship 3.13, and a tool that
refuses the Python a lab already has loses that lab at the first step.

**Independent Test**: on a clean machine with only Python 3.13, install the core and each extras
group, start the server, and run a job with every available pipeline on a short video.

**Acceptance Scenarios**:

1. **Given** a clean Python 3.13 environment, **When** the user installs VideoAnnotator with all
   extras, **Then** the install succeeds without building anything that needs a compiler or system
   headers beyond what the install docs list.
2. **Given** that install, **When** the user runs a job with every pipeline on a short test video,
   **Then** every pipeline completes, including speaker diarization (given a Hugging Face token
   with the model licences accepted).
3. **Given** that install, **When** the user lists pipelines, **Then** every installed pipeline is
   reported as available, exactly as on Python 3.12.

---

### User Story 2 - Existing Python 3.12 installs keep working (Priority: P1)

A lab already running VideoAnnotator on Python 3.12 upgrades to the release with this change and
carries on, with no reinstall of Python and no change in results.

**Why this priority**: no users should be broken by adding a Python version (constitution
principle V, backward compatibility by default).

**Independent Test**: upgrade an existing 3.12 install in place and rerun a previously completed
job on the same video; compare outputs.

**Acceptance Scenarios**:

1. **Given** a working Python 3.12 install, **When** the user upgrades VideoAnnotator, **Then** the
   upgrade succeeds on 3.12 and every pipeline still runs.
2. **Given** the same video and settings, **When** a job runs before and after the upgrade on
   3.12, **Then** the outputs are identical, apart from timestamps and run identifiers.

---

### User Story 3 - Both versions are checked on every change (Priority: P2)

A maintainer pushing a change sees the test suite run on Python 3.12 and 3.13, so a change that
breaks either is caught before it's merged.

**Why this priority**: support for a version nobody tests decays within weeks.

**Independent Test**: open a pull request and confirm the checks list a test run for each
supported Python version, and that a deliberate 3.13-only failure turns the run red.

**Acceptance Scenarios**:

1. **Given** a pull request, **When** CI runs, **Then** the test suite runs on Python 3.12 and
   3.13, and either failing blocks the merge (Windows keeps its existing allowance to fail).

---

### User Story 4 - Development and container environments use 3.13 (Priority: P3)

A contributor opening the dev container, or a user running the Docker image, gets Python 3.13
without extra steps.

**Why this priority**: the shipped and development environments should match the newest supported
version, so problems surface where we work.

**Independent Test**: rebuild the dev container and build each Docker image; check the Python
version inside and run the test suite in the dev container.

**Acceptance Scenarios**:

1. **Given** a freshly rebuilt dev container, **When** the contributor checks the project's Python,
   **Then** it is 3.13, and the test suite passes.
2. **Given** a freshly built CPU or GPU Docker image, **When** the server starts in it, **Then** it
   runs on Python 3.13 and serves the viewer and API as before.

---

### Edge Cases

- A user on Python 3.11 or earlier, or on 3.14: the install refuses with a message naming the
  supported versions (3.12 and 3.13), not a dependency-resolution traceback.
- Standard-library audio modules that 3.13 removed (`audioop`, `aifc`, `sunau`, `chunk`) are still
  imported by audio libraries: the install must bring their backports automatically, only on 3.13.
- Packages whose published metadata stops at 3.12 (for example `open-clip-torch`, `tf-keras`,
  `webvtt-py`) install and work on 3.13 in the trial; treat a regression there as a release
  blocker, not a warning.
- Pipelines that download model weights at first use behave the same on 3.13 (same cache
  locations, same readiness reporting).
- The bundled viewer and the TypeScript side are unaffected.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: VideoAnnotator MUST install and run on Python 3.12 and 3.13, and MUST refuse other
  versions at install time with a clear statement of the supported range.
- **FR-002**: This change MUST NOT upgrade, downgrade or newly pin any library, other than the
  standard-library backports Python 3.13 itself requires. Library upgrades are later specs.
- **FR-003**: Every pipeline extras group MUST install on both supported versions from published
  binary packages (or the one existing source build, `openai-whisper`), with no new system
  requirements.
- **FR-004**: Every pipeline MUST produce the same outputs on 3.12 before and after this change.
- **FR-005**: CI MUST run the test suite on both supported Python versions on every pull request
  and every push to the main branch; a failure on either MUST fail the run (Windows keeps its
  existing allowance).
- **FR-006**: The dev container MUST provide Python 3.13 for the project environment.
- **FR-007**: The CPU and GPU Docker images MUST run the server on Python 3.13.
- **FR-008**: Package metadata (supported-version classifiers), lint and type-check settings MUST
  reflect the supported range.
- **FR-009**: The README and installation docs MUST state the supported versions as exactly 3.12
  and 3.13. They currently say "3.12+", which is wrong today.
- **FR-010**: Speaker diarization MUST be verified on Python 3.13 with a real Hugging Face token
  before this spec is marked done (it was the one pipeline the trial couldn't run). *Verified
  2026-10-01 on the demo video: completes on 3.13, and its four segments are identical to 3.12's.*
- **FR-011**: The CHANGELOG MUST record the new supported version, and that Python 3.12 support is
  planned to end in v1.7.0.

### Key Entities

- **Supported Python range**: the versions VideoAnnotator declares, tests and documents. Today
  3.12 only; after this spec 3.12 and 3.13.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A clean Python 3.13 machine goes from nothing to a completed job with every pipeline
  by following the install docs alone, with no step that differs from the 3.12 instructions apart
  from the Python version.
- **SC-002**: The full test suite passes on both supported versions in CI, with the same number of
  passing tests on each (allowing for extras that are installed on one runner only).
- **SC-003**: For a fixed test video and settings, every pipeline's output on 3.12 is unchanged by
  this release, and the 3.13 output matches the 3.12 output apart from timestamps and run
  identifiers (or any difference is explained in the CHANGELOG).
- **SC-004**: An install attempt on an unsupported Python version fails within the first step,
  with a message that names the supported versions.

## Assumptions

- Python 3.12 stays supported throughout v1.6.0; dropping it is a v1.7.0 decision.
- Python 3.14 is out of scope: TensorFlow (via DeepFace in `face_analysis`) has no 3.14 release.
  The face-stack spec removes TensorFlow and adds 3.14.
- The trial's environment (Linux x86_64 with CUDA) is representative for the install; macOS and
  Windows are covered by CI.
- Ubuntu 24.04, the Docker base, ships Python 3.12, so 3.13 in the images comes from the project's
  package manager rather than the OS.
- Output equality (SC-003) allows for nondeterminism the pipelines already have on 3.12 (for
  example GPU floating-point differences); such differences are within the existing run-to-run
  variation, measured on 3.12 first.
