# Feature Specification: Output Provenance and Overlay Attribution

**Feature Branch**: `017-output-provenance` (work lands on `1.6-dev`)
**Created**: 2026-10-04
**Status**: Draft
**Input**: User description: "Output provenance and overlay attribution. Every pipeline output file records who made it: pipeline name, VideoAnnotator version, model name and weight revision, the settings that change results (pipeline config, deterministic mode and torch settings), and a real creation time (COCO files currently hard-code date_created 2025-01-01, and WebVTT/RTTM files record nothing). The job also keeps this per pipeline. The viewer labels every overlay with the pipeline and version that drew it (constitution Principle VI), and shows the full provenance on demand. This is the foundation for Phase 3's methods paragraph (GET /api/v1/jobs/{id}/methods), which is out of scope here except that the provenance must hold everything it needs (model revision SHA, prompt SHA-256 for VLM, quantisation). Must stay backward compatible: existing readers of COCO/WebVTT/RTTM keep working, and the viewer still opens older files that have no provenance (showing "version not recorded")."

## Context

Today a VideoAnnotator output file can't say what made it:

- COCO files (`person_tracking`, `scene_detection`, `face_analysis`, `vlm_annotation`, OpenFace 3)
  carry VideoAnnotator's version but not the pipeline, the model or the settings, and every one
  claims it was created on 2025-01-01 (a fixed value).
- WebVTT transcripts and RTTM speaker turns carry nothing at all.
- The job record keeps each pipeline's status, timing and annotation count, but not the model or
  settings it ran with.

The constitution requires that "every overlay MUST be attributable to the pipeline name and version
that produced it" (Principle VI) and that results be reproducible (Principle III). Neither can be
met until outputs record their own provenance. Phase 3's methods paragraph, which turns a job into
a ready-to-paste methods section, needs the same record.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - A result file says what made it (Priority: P1)

A researcher finds an output file months later, in a shared drive or a supplementary-materials
archive, separated from the job that made it. Opening it, they can tell which pipeline produced
it, which VideoAnnotator version, which model and weights, with which settings, and when.

**Why this priority**: everything else (attribution in the viewer, the methods paragraph,
reproducing a result) reads this record. Files leave the tool; their provenance has to go with
them.

**Independent Test**: run a job with every pipeline on the demo clip; for each output file, read
its provenance without the server or the job, and check every field below is present and correct.

**Acceptance Scenarios**:

1. **Given** a completed job, **When** the researcher opens any of its output files (COCO JSON,
   WebVTT, RTTM, OpenFace detail JSON, person-tracks JSON), **Then** the file itself, or a
   companion file stored and shipped next to it, names the pipeline, the VideoAnnotator version,
   the model and its weight revision, the settings that change results, and the real time it was
   created.
2. **Given** a WebVTT or RTTM output with provenance, **When** it is opened in a tool that
   reads those formats (ELAN, a subtitle player, pyannote's RTTM loader), **Then** it opens
   exactly as before and shows the same cues or speaker turns.
3. **Given** a COCO output with provenance, **When** it is loaded with the standard COCO API,
   **Then** it loads as before.
4. **Given** a job run with deterministic mode on, **When** its outputs are inspected, **Then**
   their provenance says so, along with the numerical settings in force.

---

### User Story 2 - Every overlay in the viewer names its source (Priority: P1)

A researcher reviewing a video in the viewer sees, for every overlay and timeline track (person
boxes, face landmarks, scene bands, transcript, speaker turns, VLM labels, ELAN tiers), which
pipeline and which version drew it. They can open the full provenance for any of them.

**Why this priority**: it is a non-negotiable constitution principle, and it is what makes the
viewer an audit tool. A researcher comparing two runs, or showing a result to a colleague, must
not mistake one model's output for another's.

**Independent Test**: open a job made after this feature in the viewer; every visible overlay and
track shows its pipeline and version without extra clicks, and one action shows the full
provenance.

**Acceptance Scenarios**:

1. **Given** a job opened in the viewer, **When** any overlay or track is shown, **Then** its
   pipeline name and VideoAnnotator version are visible next to its controls or legend.
2. **Given** an overlay with provenance, **When** the user asks for details, **Then** the viewer
   shows every recorded field (model, weight revision, settings, deterministic mode, creation
   time) as recorded, without rewording values.
3. **Given** two loaded files from the same pipeline but different versions or models, **When**
   both are shown, **Then** each is labelled with its own source, and the user can tell them
   apart.
4. **Given** an ELAN ground-truth file, **When** its tiers are shown, **Then** they are labelled
   as ground truth from that file, not as a pipeline's output.

---

### User Story 3 - Older files still open, honestly labelled (Priority: P2)

A researcher opens results made before this feature (v1.5.x and earlier, and the bundled demo
data). They open and display exactly as before, and their overlays say "version not recorded"
(with whatever the file does say, e.g. a COCO file's VideoAnnotator version) rather than guessing.

**Why this priority**: existing labs have months of results. Breaking them, or labelling them with
invented provenance, would violate backward compatibility (Principle V) and Principle VI.

**Independent Test**: open the committed v1.5.0-era fixtures and the demo datasets; everything
displays, and each overlay's label states what is known and that the rest wasn't recorded.

**Acceptance Scenarios**:

1. **Given** an output with no provenance, **When** it is opened in the viewer, **Then** it
   displays as before, and its label reads "version not recorded" (with the pipeline name when the
   file type identifies it).
2. **Given** an older COCO file that carries a VideoAnnotator version but no pipeline record,
   **When** it is shown, **Then** the label shows that version and marks the rest as not recorded.

---

### User Story 4 - The job keeps each pipeline's provenance (Priority: P2)

A researcher or an agent asks the server about a finished job and gets, for each pipeline, the
same provenance its output files carry, without downloading the files. This is what the Phase 3
methods paragraph will read.

**Why this priority**: the methods paragraph, reruns ("same settings"), and comparing runs all
need provenance per job; reading it from files is slower and fails when files are moved.

**Independent Test**: after a job finishes, the job's results from the API and the CLI include a
provenance record per pipeline, equal to the one in that pipeline's files.

**Acceptance Scenarios**:

1. **Given** a completed job, **When** its results are requested, **Then** each pipeline's entry
   includes its provenance.
2. **Given** a pipeline that failed, **When** its results are requested, **Then** its entry
   includes whatever provenance was known before it failed (e.g. version, model, settings), so a
   failure can be reported with its context.
3. **Given** a job from before this feature, **When** its results are requested, **Then** the
   provenance field is absent or empty and nothing else changes.

---

### Edge Cases

- A model whose weights carry no revision identifier (a file with no published hash): the
  provenance records a content hash of the weights file actually loaded.
- A model served by another process (VLM through Ollama): the provenance records the model name,
  the digest the server reports, the quantisation it reports, the server address, and the SHA-256
  of the prompt text used. If the server reports no digest, that field says so.
- A pipeline config holding a secret (an API token, a Hugging Face token): never written into
  provenance; recorded as present but redacted.
- Several files from one pipeline (OpenFace 3 writes two, person tracking two): each carries the
  same provenance record.
- The deprecated `audio_processing` pipeline: its WebVTT and RTTM name `audio_processing` as the
  pipeline that made them, plus the sub-pipeline.
- A file edited by hand after export: provenance can't detect this, and the viewer doesn't claim
  to. (Integrity checking is out of scope.)
- Clock and time zone: creation times are recorded in UTC with an explicit offset.
- `videoannotator process` (no server): writes the same provenance as a server job.

## Requirements *(mandatory)*

### Functional Requirements

**Recording**

- **FR-001**: Every output file a pipeline writes MUST carry a provenance record, embedded in the
  file where its format allows it without breaking standard readers, otherwise in a companion
  file written beside it and included wherever the output file is (job folder, artifacts ZIP,
  `--output` copies, per-file downloads).
- **FR-002**: The provenance record MUST hold: pipeline name; pipeline version (when the pipeline
  declares one); VideoAnnotator version; for each model used, its name, source and weight revision
  (a published revision identifier, or a content hash of the weights loaded); the effective
  pipeline settings (defaults merged with the job's config); deterministic mode and the numerical
  settings in force; the creation time in UTC; the job ID when made by a job; and the input
  video's name and content hash.
- **FR-003**: For VLM output, the record MUST also hold the SHA-256 of the prompt text, the model
  digest and quantisation as reported by the model server, and the server address.
- **FR-004**: COCO outputs MUST record their real creation time; the fixed `2025-01-01` value
  MUST be removed.
- **FR-005**: Secrets in settings (tokens, keys, passwords) MUST NOT be written to provenance;
  they are recorded as redacted.
- **FR-006**: Adding provenance MUST NOT change any annotation value in any output. The existing
  output baseline MUST still pass, with provenance fields excluded from comparison.
- **FR-007**: Every output MUST remain readable by the standard tools for its format (COCO API,
  WebVTT parsers, RTTM loaders, ELAN for the files it imports) exactly as before.

**Job record**

- **FR-008**: The job MUST store each pipeline's provenance record, and the job-results response
  (API and CLI) MUST include it per pipeline.
- **FR-009**: Jobs created before this feature MUST keep working; their provenance is absent, not
  invented.

**Viewer**

- **FR-010**: The viewer MUST label every overlay and timeline track with the pipeline name and
  VideoAnnotator version that produced it, visible without extra interaction.
- **FR-011**: The viewer MUST show the complete provenance record of any overlay or track on
  request, with values as recorded.
- **FR-012**: When a file has no provenance, the viewer MUST open it as before and label its
  overlays "version not recorded", showing any partial information the file does carry, and MUST
  NOT infer or fill in missing fields.
- **FR-013**: ELAN ground-truth tracks MUST be labelled as ground truth from the named file.
- **FR-014**: The viewer contract fixtures MUST be refreshed to include provenance, and the
  contract test MUST cover both files with provenance and files without.

### Key Entities

- **Provenance record**: who made one pipeline's output: pipeline (name, version), VideoAnnotator
  version, models (name, source, weight revision), effective settings (secrets redacted),
  determinism settings, creation time, job ID, input video (name, hash), plus VLM fields (prompt
  SHA-256, model digest, quantisation, server). Versioned by a schema version so later fields can
  be added.
- **Output file**: one file a pipeline writes; carries or sits beside one provenance record.
- **Pipeline result** (existing, per job and pipeline): gains the provenance record.
- **Overlay attribution** (viewer): the label and details shown for one overlay or track, derived
  only from the provenance record of the file it was drawn from.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: For a job running every pipeline on the demo clip, 100% of the output files carry a
  complete provenance record (every FR-002 field present, or explicitly marked unavailable with a
  reason).
- **SC-002**: 100% of overlays and tracks in the viewer show a pipeline and version label, for new
  jobs; for files without provenance, 100% show "version not recorded" and none shows an invented
  value.
- **SC-003**: All committed older fixtures and demo datasets open and display identically to
  before (the existing viewer tests and contract test pass unchanged in their annotation
  assertions).
- **SC-004**: The output baseline test passes with no annotation value changed.
- **SC-005**: Given only one output file, a researcher can name its pipeline, model, weight
  revision and VideoAnnotator version in under a minute, without the server.
- **SC-006**: The provenance in the job record and in the output files of the same pipeline are
  identical for every pipeline of the demo job.

## Assumptions

- Companion files are used only where a format can't carry the record safely. WebVTT allows
  `NOTE` blocks, which readers ignore; RTTM has no comment syntax that every loader skips, so it
  is expected to use a companion file. The plan confirms each format against real readers.
- "Version" in the viewer label means VideoAnnotator's version; pipeline versions and model
  revisions are in the details view.
- Integrity (signing, tamper detection) is out of scope.
- The methods paragraph (prose, BibTeX, the `/methods` endpoint) is out of scope; this feature
  only guarantees the record holds what it needs.
- Hashing the input video happens once per job and is cheap relative to running pipelines.
- Provenance is recorded for pipelines this repository ships. Third-party pipelines (v1.7
  plugins) get the common fields automatically and may add model fields.

## Dependencies

- Builds on 2026-10-04's per-pipeline torch settings (`utils/torch_settings.py`) for the
  determinism fields and on the registry metadata's `outputs[].file` for which files a pipeline
  writes.
- Phase 3's methods paragraph and tidy export (`model` column) consume this record.
