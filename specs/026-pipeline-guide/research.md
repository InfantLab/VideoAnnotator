# Research: Pipeline Guide

Decisions for [plan.md](plan.md). Each: decision, rationale, alternatives.

## R1. Where the guide lives in the metadata

**Decision**: new optional top-level keys in each `registry/metadata/<pipeline>.yaml`, beside
`citations` (spec 025): `title`, `summary`, `homepage`, `benchmarks`, `quality_note`. New optional
keys per `config_schema` entry: `group`, `label`, `unit`, `min`, `max`, `applies_when`.

**Rationale**: matches how spec 025 added `citations` (flat, optional, parsed leniently); one file
per pipeline stays the single source for the viewer, API, CLI and the generated
`docs/pipelines_spec.md`. Additive keys keep every existing metadata file and third-party file
loading (Principle V; FR-011).

**Alternatives**: a nested `guide:` block (one more level for no gain, and `citations` is already
top level); a separate guide file per pipeline (two files to keep in step); guide text in the
viewer (the CLI and API wouldn't have it, breaking SC-004).

## R2. A new `title`, not a changed `display_name`

**Decision**: add `title` (behavioural, e.g. "Faces: expression, age, gender") and keep
`display_name` ("Face Analysis (DeepFace)") unchanged. Surfaces that show a pipeline to a researcher
use `title`, falling back to `display_name`.

**Rationale**: `display_name` names the method, which is what the methods paragraph (spec 025) and
provenance want; the guide wants the behaviour (spec 023 Decision 1). Changing `display_name`
would change the methods text of existing results.

**Alternatives**: rewrite `display_name` (changes exports and methods text); derive a title from
`tasks` (reads as a tag list, not a title).

## R3. Two groups, defaulting to technical

**Decision**: `group` is `behavioural` or `technical`; absent means technical. A metadata check
(test) requires every built-in pipeline's settings to have an explicit group.

**Rationale**: an unclassified setting should never crowd the visible list (spec Edge Cases); the
check stops built-ins from relying on the default. Two groups only (spec Assumptions).

**Alternatives**: default behavioural (third-party files would show everything); a third
"advanced" group (recreates the long list).

## R4. `applies_when`

**Decision**: `applies_when: {<other setting>: [<values>]}`, all listed settings must match (AND);
values compare as strings. Evaluated in the viewer against the current form values, and in the CLI's
describe output as a note ("only when detection_backend is opencv"). A setting that doesn't apply
but differs from its default stays visible, marked "not used with these settings" (FR-005).

**Rationale**: covers the cases found (OpenCV-only thresholds, burst-only offsets in the VLM:
`burst_offsets` applies when `sampling_mode` is `frame_burst`) without an expression language.

**Alternatives**: a JSON-Schema `if/then` (heavy for the viewer to evaluate); free-text notes only
(can't hide anything).

## R5. Benchmarks: format and sourcing

**Decision**: each entry is `{name, measure, figure, source, note?}`; `figure` is a string quoted
from the source ("91.4%", "WER 4.3"), never a computed number. `benchmarks: []` is an explicit "no
published benchmark". `quality_note` covers model-dependent pipelines (the VLM: quality depends on
the model chosen; link the model's page). Figures are gathered during implementation from each
method's paper or its official model card, one task per pipeline, with the source URL recorded.

**Rationale**: FR-007 forbids computing or rounding beyond the source; strings keep the source's
own form (units, decimals). Spec 030's research (model defaults, in progress) collects upstream
evaluations for candidate replacements; where a default changes, its benchmark entry changes in the
same commit (spec 030's own requirement).

**Alternatives**: numeric `value` + `unit` (invites rounding and comparisons across incomparable
benchmarks); linking to Papers with Code only (sunset, and not the method's own claim).

## R6. Settings the code doesn't read (FR-012)

**Finding** (word search of each pipeline package for each `config_schema` key, 2026-10-17):
- `audio_processing.enable_diarization`: not read. `audio_processing` is deprecated (spec 014).
- `scene_detection.enable_clip_labels`: not read.
- `face_analysis.scale_factor`, `min_neighbors`: read only on the OpenCV path → `applies_when:
  {detection_backend: [opencv]}`, technical.
- `vlm_annotation.burst_offsets`: read only in `frame_burst` mode → `applies_when`.

**Decision**: remove the two unread keys from metadata unless a saved preset or job config in the
repo's fixtures uses them (none do: searched `tests/`, `examples/`, `configs/`); a submitted config
that still contains them keeps working: the validator warns "Unknown field '…' will be ignored"
(`validation/validator.py`), which is now true of what those keys always did. Re-run the word
search as a test (`tests/unit/registry/test_pipeline_guide.py`) so a new unread key fails CI.

**Rationale**: an option that does nothing is the faithful-display problem (spec 023 fix
`41a38d2`) in settings form.

**Alternatives**: keep and mark them technical (still shown as if they do something).

## R7. API shape

**Decision**: `GET /api/v1/pipelines` and `GET /api/v1/pipelines/{name}` gain `title`, `summary`,
`homepage`, `benchmarks`, `quality_note` and `citations` (key + BibTeX); each `config_schema` entry
gains `group`, `label`, `unit`, `min`, `max`, `applies_when`. `GET /{name}/schema`'s parameters gain
`group`, `unit`, `min`, `max`, `applies_when`, and `label` comes from metadata (falling back to
today's title-cased key). The pipeline descriptor gains `title` and `summary`. All additions;
nothing renamed or removed.

**Rationale**: the viewer's `PipelineParameterSchema` type already has `group`, `unit`, `min`,
`max` (unused until now), so the schema route is the natural carrier; the info route serves the CLI
and agents.

## R8. CLI

**Decision**: `videoannotator pipelines describe <name> [--json]` reads the local registry (no
server needed, like spec 025's `export`); `pipelines list` (server-backed) shows `title` beside
`name` in its table and markdown output when the server provides it.

**Rationale**: describing a pipeline needs only the metadata on disk; agents get `--json`
(roadmap principle "agents are users too").

## R9. Viewer

**Decision**: one `PipelineGuide` component used on the Pipelines page and in the job wizard's
pipeline choice: title, summary, availability badge, then `DynamicPipelineParameters` split into
"What to measure" (behavioural, open) and "Technical settings (n, m changed)" (collapsed), then an
About block (homepage, benchmarks with source links, quality note, citations with Copy BibTeX).
`applies_when` filtering lives in `DynamicPipelineParameters`. OpenFace 3 and VLM's one-off
controls are left as they are (spec 027 redoes the VLM's).

**Rationale**: smallest change that meets FR-009 on today's pages; spec 029 later moves it into the
new run setup.

## R10. Generated pipeline doc

**Decision**: `scripts/generate_pipeline_specs.py` adds a per-pipeline section (title, summary,
behavioural settings, homepage, benchmarks, citations) to `docs/pipelines_spec.md`; the existing
drift test covers it.

**Rationale**: the docs and the app say the same (v1.7.0 Phase 3's "which pipeline for which
question" starts here).
