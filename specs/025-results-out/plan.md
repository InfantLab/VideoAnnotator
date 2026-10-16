# Implementation Plan: Results Out

**Branch**: `1.6-dev` (spec `025-results-out`) | **Date**: 2026-10-16 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/025-results-out/spec.md`

## Summary

Turn the per-pipeline output files into things a researcher uses directly: one long-format table
(CSV always, Parquet with an optional extra), one ELAN `.eaf` per video, and a methods paragraph with
BibTeX, plus R and Python notebooks over the demo clip's table. Everything is read from the files on
disk and their spec 017 provenance records by one exporter package, so the CLI (on any folder, no
server), the API (job, run, dataset) and the viewer (an Export menu) produce identical results.
Native outputs don't change. Decisions: [research.md](research.md).

## Technical Context

**Language/Version**: Python 3.12/3.13 (core); TypeScript/React (viewer, Bun); R ≥ 4.2 for one notebook
**Primary Dependencies**: core only (stdlib `csv`, `xml.etree`, `zipfile`; PyYAML; FastAPI). Optional
`pyarrow` in a new `parquet` extras group. Dev: `xmlschema` (EAF schema validation). Notebooks:
pandas + matplotlib (Python), readr/dplyr/tidyr/ggplot2 (R)
**Storage**: none new; reads results folders (spec 022) and job folders; one new storage query
(`list_jobs_by_dataset`)
**Testing**: pytest (unit over the committed fixtures, API via TestClient, CLI via Typer's runner,
notebook execution); Vitest contract test for the exported `.eaf`
**Target Platform**: Linux/macOS/Windows laptops and the container (spec 024)
**Project Type**: Python package + bundled web viewer
**Performance Goals**: 100-video run with every pipeline exported in < 1 min (SC-003); status routes
stay responsive meanwhile
**Constraints**: offline-capable; slim core install imports the exporters without pyarrow/pandas;
bounded memory (one video's rows at a time)
**Scale/Scope**: up to ~10⁶ rows per export

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design: still passes.*

| Principle | Check | Status |
|---|---|---|
| I. Local-First, No Telemetry | Exports read local files; CLI needs no server or network; no external calls. ELAN links to the local video path. | Pass |
| II. Stable Contract, Open Formats | Native outputs unchanged (FR-036). Adds open formats (CSV, Parquet, EAF 3.0, BibTeX). Metadata gains an optional `citations` key: additive, allowed in a minor version. The viewer's ELAN contract test covers the new `.eaf` files. The table is versioned (FR-014). | Pass |
| III. Provenance & Reproducibility | Built on spec 017 records; `model`/`videoannotator_version` on every row; Parquet carries every source's record; the methods text cites exact revisions; gaps say "not recorded", never inferred. | Pass |
| IV. Modular by Construction | pyarrow only in the `parquet` extra; exporters import nothing from pipeline packages (they read files); citations live with each pipeline's metadata. Viewer: one self-contained Export menu, code-split dialog. | Pass |
| V. Backward Compatibility | Pre-017 and pre-022 results export (FR-037, SC-007); no route, CLI command or file changes, only additions. | Pass |
| VI. Faithful Annotation Display | No viewer display change. ELAN merging of frame detections into "in view" intervals is an export, documented, with the raw rows in the table; it never invents labels or values (VLM samples hold until the next sample, the rule Principle VI already allows). | Pass |

No violations; Complexity Tracking is empty.

## Project Structure

### Documentation (this feature)

```
specs/025-results-out/
├── plan.md  research.md  data-model.md  quickstart.md
├── contracts/  table.md  api.md  cli.md  elan-and-methods.md
├── checklists/requirements.md
└── tasks.md            # /speckit-tasks
```

### Source Code (repository root)

```
src/videoannotator/
├── exporters/
│   ├── results_set.py      # NEW: folders/jobs → VideoResults (files, provenance, skipped, missing)
│   ├── readers.py          # NEW: one reader per native format → Row
│   ├── table.py            # NEW: rows → CSV / Parquet, ordering, ExportSummary
│   ├── elan.py             # NEW: rows → EAF 3.0
│   ├── methods.py          # NEW: provenance + metadata → prose, BibTeX, prompts
│   └── citation.py         # NEW: VideoAnnotator's own BibTeX entry
├── registry/
│   ├── pipeline_registry.py  # PipelineMetadata.citations (optional)
│   └── metadata/*.yaml       # citations: for all 8 shipped pipelines
├── storage/{base,sqlite_backend,file_backend}.py  # list_jobs_by_dataset
├── api/v1/exports.py       # NEW: table/elan/methods routes for jobs, batches, datasets
├── api/v1/__init__.py      # include exports router
└── cli.py                  # `export` Typer group: table, elan, methods

viewer/src/
├── api/client.ts           # exportTable / exportElan / getMethods (+ summary header parsing)
├── components/ExportMenu.tsx      # NEW
├── components/MethodsDialog.tsx   # NEW (lazy)
├── pages/JobDetail.tsx, pages/BatchDetail.tsx, pages/Datasets.tsx  # mount the menu
└── test/contract/…         # exported .eaf fixtures open and show tiers

examples/notebooks/
├── demo_clip_table.csv          # NEW: table export of the demo clip fixtures
├── results_table_python.ipynb   # NEW
├── results_table_r.Rmd          # NEW
└── README.md                    # NEW
scripts/make_demo_table.py       # NEW: regenerates demo_clip_table.csv and the .eaf fixtures

tests/
├── fixtures/elan/EAFv3.0.xsd           # NEW
├── fixtures/viewer_contract/exported/  # NEW: demo_clip.eaf, demo_clip_edited_in_elan.eaf
├── unit/exporters/test_readers.py, test_table.py, test_elan.py, test_methods.py, test_results_set.py
├── unit/test_cli_export.py, unit/test_notebooks.py, unit/registry/test_citations.py
├── api/test_export_routes.py
└── manual/results_out_r_notebook.md

docs/usage/exporting_results.md   # NEW: the table's columns and measures, ELAN, methods
```

**Structure Decision**: a new set of modules in the existing `exporters/` package (today only
`native_formats.py`), a new API router file beside `batches.py`/`jobs.py`, and a CLI Typer group in
`cli.py`, following the existing `dataset`/`prompts` groups.

## Phasing (for /speckit-tasks)

1. **Foundation**: reader table + `results_set` + `Row`; fixtures; `list_jobs_by_dataset`;
   `parquet` extra; `xmlschema` dev dep.
2. **US1 Table (P1, MVP)**: readers for all seven formats, CSV/Parquet writer, summary; CLI
   `export table`; API table routes; viewer Export menu (table items); docs page.
3. **US2 Methods (P2)**: `citations` in metadata + YAML for 8 pipelines; VideoAnnotator citation;
   `methods.py`; CLI + API; MethodsDialog.
4. **US3 ELAN (P3)**: `elan.py`; CLI + API; viewer menu item; `.eaf` fixtures in the viewer contract
   test.
5. **US4 Notebooks (P4)**: demo table script + CSV; Python notebook + its test; R notebook + manual
   check.
6. **Polish**: CHANGELOG, roadmap Phase 3 ticks, README "Get your results out", rebuild viewer bundle,
   regenerate viewer API types, full pre-commit and test runs.

## Complexity Tracking

None.
