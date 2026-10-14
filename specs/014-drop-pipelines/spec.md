# Feature Specification: Drop Obsolete Pipelines

**Feature Branch**: `1.6-dev` (v1.6.0 development branch)
**Created**: 2026-10-01
**Status**: Implemented 2026-10-01 on `1.6-dev`
**Input**: v1.6.0 Phase 1, spec 4 of the dependency audit's sequence. The pipeline review
([`pipeline_review_v1.6.0.md`](../../docs/development/pipeline_review_v1.6.0.md), decided
2026-10-01) drops `laion_voice` (16–32 GB of weights, abandoned upstream, adult acted speech) and
`face_laion_clip` (abandoned upstream, unvalidated on infants), and drops `audio_processing`
(duplicates `speech_recognition` + `speaker_diarization`), keeping its name as an alias for one
release.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - The pipeline list only offers tools worth running (Priority: P1)

A researcher choosing pipelines in the viewer or CLI sees only maintained pipelines that answer a
real question, without a 16–32 GB voice model or abandoned face models among them.

**Why this priority**: every pipeline listed is an implicit recommendation; two of today's ten
are not ones we'd recommend, and one duplicates two others.

**Independent Test**: install all extras, list pipelines (API, CLI, viewer); `laion_voice`,
`face_laion_clip` and `audio_processing` aren't offered.

**Acceptance Scenarios**:

1. **Given** every extra installed, **When** the user lists pipelines, **Then** `laion_voice`,
   `face_laion_clip` and `audio_processing` are not listed, and the `audio-laion` and
   `face-laion` extras no longer exist.
2. **Given** the installed extras list, **When** the user installs "everything" (`[all]`),
   **Then** the LAION models are not part of it.

---

### User Story 2 - Old jobs and configs fail clearly or keep working (Priority: P1)

A lab with a v1.5.0 config or script naming one of these pipelines either keeps working
(`audio_processing`) or gets a clear message saying what happened and what to use instead (the
LAION pipelines).

**Why this priority**: constitution principle V (backward compatibility within v1.x); silent
failures or tracebacks for a removed pipeline would cost a lab hours.

**Independent Test**: submit jobs naming each of the three; check the response.

**Acceptance Scenarios**:

1. **Given** a job or config naming `laion_voice` or `face_laion_clip`, **When** it's submitted,
   **Then** it's rejected with a message saying the pipeline was removed in v1.6.0, why, and the
   alternative (OpenFace 3 / `face_analysis` for faces; none yet for voice emotion).
2. **Given** a job or config naming `audio_processing`, **When** it runs, **Then** it produces the
   same outputs as in v1.5.0 and records a deprecation warning saying it will be removed in
   v1.7.0 and to use `speech_recognition` + `speaker_diarization`.

---

### User Story 3 - Short family names mean a predictable pipeline (Priority: P2)

A user who asks for the `audio` or `face` family by its short name gets the same pipeline
whatever else is installed.

**Why this priority**: today the short name `audio` resolves to `laion_voice` when its extra is
installed (a 16–32 GB download) and to `speaker_diarization` otherwise, because ties between
"stable" pipelines are broken alphabetically.

**Independent Test**: resolve each family's short name with only that family's extra installed
and with everything installed; the result is the same.

**Acceptance Scenarios**:

1. **Given** any combination of installed extras, **When** a family's short name is resolved,
   **Then** it names the pipeline that family declares as its default.
2. **Given** the `audio` family, **Then** its short name means speech recognition plus
   diarization (`audio_processing` while deprecated); after v1.7.0 the short name is removed
   along with it.

### Edge Cases

- Existing job records in the database naming a removed pipeline: listing and viewing them still
  works (results are files on disk); only re-running them fails, with the removal message.
- The bundled config files (`configs/*.yaml`) have `audio_processing:` sections: they keep
  working (alias) and are updated to the two separate pipelines.
- The viewer still parses LAION output files from old jobs; removing its parsers is the viewer
  clean-up's job, not this spec's.
- CLIP weights in `scene_detection` are named after the LAION-2B dataset (`laion2b_s34b_b79k`);
  unrelated, unchanged.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The `laion_voice` and `face_laion_clip` pipelines, their code, metadata, tests and
  the `audio-laion` and `face-laion` extras MUST be removed.
- **FR-002**: Requesting a removed pipeline (API, CLI, config) MUST fail before any processing
  with a message naming the version it was removed in, the reason, and the alternative.
- **FR-003**: `audio_processing` MUST keep running with unchanged outputs, MUST NOT appear in
  pipeline listings, and MUST record a deprecation warning (job log and API response) pointing to
  `speech_recognition` + `speaker_diarization` and naming v1.7.0 for removal.
- **FR-004**: Each pipeline family MUST declare its default in metadata; short family names MUST
  resolve to that default regardless of installed extras.
- **FR-005**: Bundled configs, examples and docs MUST use `speech_recognition` +
  `speaker_diarization` instead of `audio_processing`, and MUST NOT mention the LAION pipelines
  except in the CHANGELOG and archive.
- **FR-006**: Remaining pipelines' outputs MUST be unchanged (demo-video comparison).
- **FR-007**: The CHANGELOG MUST record the removals, the deprecation, and the `audio` short-name
  change.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: With every extra installed, the pipeline list has 7 entries (was 10), and an
  everything-install no longer downloads the 16–32 GB LAION voice weights on first use.
- **SC-002**: Every request for a removed pipeline gets the removal message, in all three entry
  points (API, CLI, config file); none produces a traceback.
- **SC-003**: A v1.5.0 job using `audio_processing` produces identical outputs after the change.
- **SC-004**: Each family's short name resolves to the same pipeline under every combination of
  installed extras tested (each extra alone, and all).

## Assumptions

- "Removed" pipelines can return as v1.7.0 plugins; nothing here prevents that.
- Old LAION output files on disk stay readable by the viewer.
- Deprecating rather than removing `audio_processing` is chosen to honour principle V; the
  pipeline review's "keep the name as an alias for one release" is read as "keep it working".
