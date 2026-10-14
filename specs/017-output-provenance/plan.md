# Implementation Plan: Output Provenance and Overlay Attribution

**Branch**: `1.6-dev` (spec `017-output-provenance`) | **Date**: 2026-10-04 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/017-output-provenance/spec.md`

## Summary

Every pipeline output records what made it (pipeline, VideoAnnotator version, models and weight
revisions, effective settings, determinism, real creation time, job, input video, and for VLM the
prompt hash and model digest). The record is built once per pipeline run in the job runner and
stamped into each file the pipeline wrote: a top-level `provenance` key in JSON, a `NOTE` block in
WebVTT, a companion `.provenance.json` for RTTM. The job stores the same record per pipeline and
the results API returns it. The viewer reads it from each file and labels every overlay with its
pipeline and version, with the full record on demand; files without one say "version not
recorded". COCO's hard-coded `date_created` becomes the real time.

## Technical Context

**Language/Version**: Python 3.12/3.13 (server, pipelines); TypeScript/React (viewer, Bun)
**Primary Dependencies**: existing only: FastAPI, SQLAlchemy, Pydantic; torch (determinism
fields); huggingface_hub (Hub revision); ollama client (model digest). No new dependency.
**Storage**: SQLite/SQLAlchemy `pipeline_results` gains a nullable JSON `provenance` column
(additive migration); file backend serialises the same field; output files on disk.
**Testing**: pytest (unit, API, contract, real-model baseline); Vitest (viewer parsers, controls,
contract test against fixtures with and without provenance).
**Target Platform**: Linux/macOS/Windows server; evergreen browsers for the viewer.
**Project Type**: web application (Python backend + React viewer in one repo).
**Performance Goals**: stamping adds < 2 s to a job on the demo clip; weight files hashed once per
server process (cache by path, size, mtime); input video hashed once per job.
**Constraints**: output files stay readable by pycocotools, WebVTT parsers, pyannote's RTTM loader;
annotation values unchanged (baseline test); no secrets in records.
**Scale/Scope**: 8 registered pipelines, 9 output file kinds; viewer: 8 overlay kinds.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Assessment |
|---|---|
| I. Local-First Execution | Pass. Nothing leaves the machine; hashing is local; the VLM digest comes from the configured (local by default) Ollama server. |
| II. Stable Pipeline Contract | Pass. Additive only: a new top-level JSON key, a WebVTT comment block, a companion file. Annotation content unchanged; standard readers verified in tests (FR-007). `info.date_created` changes value, not shape. |
| III. Provenance & Reproducibility | This feature implements it: model revisions, settings, determinism and input hash per output. |
| IV. Modular by Construction | Pass. Stamping is central and driven by registry `outputs[].file`, so plugin pipelines get the common fields; model fields come from an optional `provenance_models()` hook with a default of none. |
| V. Backward Compatibility | Pass. Nullable DB column via additive migration; old jobs return `provenance: null`; old files open in the viewer as "version not recorded"; legacy fixtures kept and tested. |
| VI. Faithful Annotation Display | This feature implements its attribution clause; the details view shows recorded values verbatim and never infers missing ones. |
| Engineering Standards | Tests for every new surface (API field, CLI output, file formats, viewer); typecheck, lint, bundle check, baseline all stay green. |

No violations. Re-checked after Phase 1 design: unchanged.

## Project Structure

### Documentation (this feature)

```
specs/017-output-provenance/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/provenance-in-files.md
└── tasks.md             # /speckit-tasks
```

### Source Code (repository root)

```
src/videoannotator/
├── provenance.py                    # NEW: ProvenanceRecord/ModelRef builders, redaction,
│                                    #      weight hashing (cached), file stamping per format
├── batch/
│   ├── job_execution.py             # build record after initialize(); stamp files after process();
│   │                                #      keep it on PipelineResult (also on failure)
│   ├── result_files.py              # list companion .provenance.json files too
│   └── types.py                     # PipelineResult.provenance
├── storage/
│   ├── models.py, sqlite_backend.py # pipeline_results.provenance JSON + additive migration
│   └── file_backend.py              # serialise provenance
├── api/v1/jobs.py                   # PipelineResultResponse.provenance
├── exporters/native_formats.py      # real date_created / year
└── pipelines/
    ├── base_pipeline.py             # provenance_models() default hook
    └── */                           # provenance_models() per pipeline (R3)

viewer/src/
├── lib/provenance.ts                # NEW: extract from JSON / VTT NOTE / companion; ProvenanceInfo
├── lib/parsers/*.ts, merger.ts      # carry provenance per track into StandardAnnotationData
├── types/annotations.ts             # ProvenanceRecord, ProvenanceInfo
└── components/UnifiedControls.tsx   # attribution line + details popover per overlay

tests/
├── unit/test_provenance.py          # builder, redaction, hashing cache, per-format stamping
├── unit/storage/…                   # migration adds column on an old DB
├── api/test_result_files.py         # provenance in results; companion listed
├── contract/test_output_readers.py  # pycocotools / pyannote RTTM / webvtt read stamped files
├── integration/test_output_baseline.py  # strips provenance & dates before comparing
└── fixtures/viewer_contract/        # regenerated with provenance; legacy/ keeps old ones
```

**Structure Decision**: existing layout; one new backend module (`provenance.py`) and one new
viewer module (`lib/provenance.ts`).

## Complexity Tracking

None.
