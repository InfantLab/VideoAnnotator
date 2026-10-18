# Implementation Plan: Pipeline Guide

**Branch**: `1.6-dev` (spec `026-pipeline-guide`) | **Date**: 2026-10-17 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/026-pipeline-guide/spec.md`
**Roadmap**: v1.6.0 Phase 3b

## Summary

Give every pipeline a short guide, written once in its metadata file: a behavioural title and
summary, settings split into behavioural (shown) and technical (collapsed) with plain labels and
units, settings hidden when they don't apply, and an About block (homepage, published benchmarks,
citations). The registry parses the new optional keys leniently; the API returns them additively;
a new `videoannotator pipelines describe` prints them with no server; the viewer's Pipelines page
and job wizard show them through one `PipelineGuide` component; the generated
`docs/pipelines_spec.md` includes them. A metadata check keeps the built-ins complete. Two settings
the code never reads are removed. Decisions: [research.md](research.md).

## Technical Context

**Language/Version**: Python 3.12/3.13 (core); TypeScript/React (viewer, Bun)
**Primary Dependencies**: existing only (PyYAML, FastAPI, Typer; React, shadcn/ui `Collapsible`)
**Storage**: none; metadata YAML files in `src/videoannotator/registry/metadata/`
**Testing**: pytest (registry parsing, metadata check, API via TestClient, CLI via Typer runner,
generated-doc drift); Vitest (components)
**Target Platform**: as the rest of VideoAnnotator (Linux/macOS/Windows, container)
**Project Type**: Python package + bundled web viewer
**Performance Goals**: none new (metadata is loaded once)
**Constraints**: additive only (Principle V); third-party metadata without the new keys still
loads; benchmark figures quoted from sources, never computed
**Scale/Scope**: 8 metadata files (7 listed + deprecated `audio_processing`), 52 settings

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design: still passes.*

| Principle | Check | Status |
|---|---|---|
| I. Local-First, No Telemetry | Metadata on disk; `describe` needs no server or network; links in About are opened by the user, nothing is fetched. | Pass |
| II. Stable Contract, Open Formats | No output change. Metadata gains optional keys (allowed in a minor version); API fields added, none renamed. | Pass |
| III. Provenance & Reproducibility | No change to what runs or what's recorded. Benchmarks cite their sources; figures quoted, never computed. | Pass |
| IV. Modular by Construction | Guide lives in each pipeline's own metadata; core reads it through the registry only; no pipeline imports. | Pass |
| V. Backward Compatibility | Existing metadata, configs and API clients unaffected. The two removed settings were never read; a config that still has them runs, with the validator's existing "will be ignored" warning (R6). | Pass |
| VI. Faithful Annotation Display | No annotation display change. Applies in spirit: a setting that does nothing is no longer offered; a changed setting is never hidden. | Pass |

No violations; Complexity Tracking is empty.

## Project Structure

### Documentation (this feature)

```
specs/026-pipeline-guide/
├── plan.md  research.md  data-model.md  quickstart.md
├── contracts/  metadata.md  api.md  cli.md  viewer.md
├── checklists/requirements.md
└── tasks.md            # /speckit-tasks
```

### Source Code (repository root)

```
src/videoannotator/
├── registry/
│   ├── pipeline_registry.py        # PipelineMetadata/PipelineConfigField gain fields; Benchmark dataclass; lenient parsing
│   └── metadata/*.yaml             # title, summary, homepage, benchmarks, quality_note; per-setting group/label/unit/min/max/applies_when
├── api/v1/pipelines.py             # PipelineInfo + schema route: new fields
└── cli.py                          # `pipelines describe`; Title column in `pipelines list`

scripts/generate_pipeline_specs.py  # per-pipeline guide sections in docs/pipelines_spec.md

viewer/src/
├── components/PipelineGuide.tsx            # new
├── components/DynamicPipelineParameters.tsx # groups, counts, appliesWhen
├── pages/Pipelines.tsx, pages/NewJob.tsx    # use PipelineGuide (PipelineSelectionStep)
├── api/client.ts, types/pipelines.ts        # new fields, appliesWhen mapping
└── test/components/PipelineGuide.test.tsx, DynamicPipelineParameters.test.tsx

tests/
├── unit/registry/test_pipeline_guide.py     # parsing, built-in completeness check, unread-key check, API/CLI agreement
├── api/test_pipeline_guide_api.py
└── unit/cli/test_cli_pipelines_describe.py
```

**Structure Decision**: existing layout; one new viewer component, one new CLI command, no new
modules on the Python side beyond tests.

## Implementation order

1. Registry: dataclasses and parsing (+ unit tests for lenient parsing).
2. Metadata content for all 8 pipelines: titles, summaries, homepages, classification
   ([data-model.md](data-model.md) first pass), labels/units, `applies_when`, removal of the two
   unread keys. Benchmarks sourced one pipeline at a time (R5), coordinated with spec 030's
   research where a default may change.
3. Metadata check test (completeness, ≤ 5 behavioural, unread keys).
4. API fields + tests.
5. CLI `describe` and the list's Title column + tests.
6. Viewer: types/client, `DynamicPipelineParameters`, `PipelineGuide`, the two pages + tests;
   rebuild `viewer_static`.
7. Generated doc; CHANGELOG; roadmap v1.6.0 Phase 3b tick.

## Risks

- **Benchmark figures**: the slowest part, and easy to get wrong. Mitigation: R5's rules, a source
  URL per figure, and review of each entry against its source before commit.
- **Classification disagreements**: a researcher may want `confidence_threshold` visible. The
  first pass is recorded with its judgement calls (data-model.md); confirm with the maintainer at
  step 2.
- **Spec 030 changing defaults**: whichever lands second updates the affected guide entries.

## Complexity Tracking

None.
