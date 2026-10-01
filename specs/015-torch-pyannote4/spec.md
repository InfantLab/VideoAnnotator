# Feature Specification: torch 2.11, pyannote.audio 4 and CUDA 12.6

**Feature Branch**: `1.6-dev` (v1.6.0 development branch)
**Created**: 2026-10-01
**Status**: Implemented 2026-10-01 on `1.6-dev` (pending CI)
**Input**: v1.6.0 Phase 1, spec 3 of the dependency audit's sequence
([`dependency_audit_v1.6.0.md`](../../docs/development/dependency_audit_v1.6.0.md) §1, §4, §5).
torch is held at 2.6.0 only because torchaudio ≥ 2.9 removed `AudioMetaData`, which pyannote.audio 3
uses; so upgrading torch means migrating to pyannote.audio 4, and current torch is built for CUDA
12.6 and newer, not 12.4. One change, since each forces the others.

**Found while specifying** (trial, 2026-10-01): pyannote.audio 4 imports torchaudio, whose last
release (2.11; the project is discontinued) is compiled against torch 2.11 and fails to load on
torch ≥ 2.12. So the target is **torch 2.11** (with torchvision 0.26, torchaudio 2.11 and
torchcodec 0.11, the builds matching torch 2.11), not the 2.14 the audit assumed. torch moves past
2.11 when pyannote.audio drops torchaudio.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Current GPU stack, same results (Priority: P1)

A lab installs v1.6.0 and gets a current, supported torch with GPU wheels matching the Docker
images' CUDA, and every pipeline gives the results it gave before (or a documented, explained
difference).

**Why this priority**: torch 2.6 is a year and a half behind; its CUDA 12.4 wheels don't match our
own CUDA 12.6 images; and it blocks every library that has moved on.

**Independent Test**: install every extra; run the test suite and every pipeline on the demo video;
compare outputs with the previous lock.

**Acceptance Scenarios**:

1. **Given** a GPU machine with a driver supporting CUDA 12.6, **When** the user installs with all
   extras, **Then** torch uses the GPU and every pipeline runs on it.
2. **Given** the demo video, **When** every pipeline runs before and after, **Then** outputs are
   identical or within the run-to-run variation measured before; any other difference is
   explained in the CHANGELOG.
3. **Given** a CPU-only machine (macOS, Windows, Linux without GPU), **Then** everything still
   installs and runs on CPU.

---

### User Story 2 - Diarization keeps working on pyannote.audio 4 (Priority: P1)

A researcher running speaker diarization gets speaker turns as before, with the same default model
(`speaker-diarization-3.1`), the same output format, and no new system requirement.

**Why this priority**: diarization is one of the most-used pipelines; pyannote.audio 4 changes how
models are loaded (`token` instead of `use_auth_token`), what a pipeline returns, and how audio is
read (torchcodec, needing FFmpeg's shared libraries).

**Independent Test**: diarize the demo video with a token; compare speaker turns with pyannote 3.

**Acceptance Scenarios**:

1. **Given** a valid Hugging Face token, **When** diarization runs, **Then** it uses
   `speaker-diarization-3.1` by default and produces RTTM-format turns as before.
2. **Given** an install where FFmpeg's shared libraries are absent (only the `ffmpeg` command-line
   tool, as from a static build), **Then** diarization still works.
3. **Given** no token or a token without the model licence accepted, **Then** the error says so,
   as today.

---

### User Story 3 - No data leaves the machine unless the user says so (Priority: P1)

A lab processing recordings of infants installs VideoAnnotator and runs diarization; nothing about
their files is sent anywhere.

**Why this priority**: constitution principle I (local-first). pyannote.audio 4 ships with
anonymous usage telemetry **on by default** (`metrics_enabled: true`, sending pipeline names, file
durations and speaker counts to `otel.pyannote.ai`).

**Independent Test**: run diarization with network access monitored; no connection to the
telemetry endpoint.

**Acceptance Scenarios**:

1. **Given** a default install and no telemetry setting, **When** diarization runs, **Then** no
   telemetry is sent.
2. **Given** a user who explicitly sets `PYANNOTE_METRICS_ENABLED=1`, **Then** their choice is
   respected.

### Edge Cases

- torchaudio's last release (2.11) declares no torch requirement, so a resolver will happily pair it
  with a newer torch that it then fails to load against; the constraints must pin the pair.
- `openface-test==0.1.13`, `openai-whisper` (and its Triton kernels), ultralytics and open_clip
  must all work with torch 2.11.
- Older NVIDIA drivers that supported CUDA 12.4 but not 12.6: the install docs state the new
  minimum; on such a driver torch falls back to CPU (already handled by pipelines' CPU fallback).
- Users who pinned `pyannote/speaker-diarization-community-1` in config: supported, since
  pyannote.audio 4 runs it; the default stays 3.1 until Phase 5's benchmark.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Every extra that uses torch MUST use torch 2.11.x (one range shared by all groups) with
  the matching torchvision, torchaudio and torchcodec; Linux GPU wheels MUST come from the CUDA 12.6
  index. The cap and its reason MUST be recorded beside the constraint.
- **FR-002**: The `audio` extra MUST use pyannote.audio 4.x; our code MUST work with its API
  (`token`, its return type) and MUST NOT depend on FFmpeg shared libraries being present.
- **FR-003**: The default diarization model MUST stay `pyannote/speaker-diarization-3.1`.
- **FR-004**: pyannote's telemetry MUST be off unless the user explicitly enables it; this MUST be
  tested.
- **FR-005**: All pipelines' outputs on the demo video MUST be identical or within measured
  run-to-run variation; differences beyond that MUST be investigated and explained.
- **FR-006**: The install docs and CHANGELOG MUST state the new minimum NVIDIA driver (CUDA 12.6
  level) and the telemetry default.
- **FR-007**: The full test suite MUST pass on every supported platform and Python.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Every pipeline that used the GPU before still does on a CUDA 12.6-capable machine, and
  everything runs on CPU elsewhere. (Diarization has always run on CPU: the pyannote pipeline is
  never moved to the GPU. Changing that is out of scope.)
- **SC-002**: Diarization of the demo video gives the same speaker turns as with pyannote.audio 3
  (start, end, speaker), or the differences are explained.
- **SC-003**: Zero network requests to pyannote's telemetry endpoint during a default run.
- **SC-004**: CI passes on Ubuntu and macOS for Python 3.12 and 3.13.

## Assumptions

- Phase 5 decides whether `speaker-diarization-community-1` becomes the default; this spec only
  makes it available.
- The minimum driver is the one NVIDIA documents for CUDA 12.6 minor-version compatibility.
