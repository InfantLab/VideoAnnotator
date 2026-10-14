<!--
SYNC IMPACT REPORT
==================
Version change: 1.0.1 → 1.1.0 (2026-10-02)
Bump rationale: MINOR. Folds in the Video Annotation Viewer's constitution
(viewer/.specify/memory/constitution.md v1.0.0, ratified 2026-05-06), now that
the viewer lives in viewer/ and shares VideoAnnotator's version. One principle
added; four expanded to cover the viewer; nothing removed or redefined, so no
existing spec or plan is invalidated. Approved by the maintainer 2026-10-02.

Modified principles:
  - I. Local-First Execution → I. Local-First Execution, No Telemetry
    (viewer I: rendering in the browser, no analytics/beacons/CDN assets;
    no telemetry product-wide, dependencies' telemetry off by default)
  - II. Stable Pipeline Contract and Open Formats (viewer II: the viewer
    reads the same formats, tolerates missing optional fields, invents
    nothing; format and viewer change ship together)
  - IV. Modular by Construction (viewer IV: composable view layers)
  - V. Backward Compatibility by Default (viewer V: the viewer's user-facing
    surface is covered)

Added sections:
  - VI. Faithful Annotation Display (NON-NEGOTIABLE), from viewer III, with
    one clarification: holding the latest sample until the next sampling
    instant creates no value and is allowed (viewer since 98a0cf1).
  - Engineering Standards: "Viewer" bullet.
  - Plan Gating: a check for Principle VI.

Brought into compliance with the expanded Principle I in the same change:
person tracking switched Ultralytics' analytics on (a Google Analytics event per
predict, its default); now off for the process. HF_HUB_DISABLE_TELEMETRY is set
by default alongside PYANNOTE_METRICS_ENABLED=0 (tests/unit/test_no_telemetry.py).

Removed sections: none. The viewer's separate constitution and spec-kit setup
(viewer/.specify, viewer/.claude/skills, viewer/.github/prompts) are retired;
viewer/specs/ stays as history.

Templates requiring updates:
  ✅ .specify/templates/plan-template.md: gates come from Plan Gating, which
     now covers Principle VI; no edit needed.
  ✅ .specify/templates/spec-template.md, tasks-template.md: compatible.
  ✅ AGENTS.md §23: no longer points to the viewer's constitution.
  ⚠ .specify/templates/checklist-template.md: still not reviewed (carried
     over from 1.0.0).

Follow-up TODOs (the viewer does not yet meet these; each is on
docs/development/roadmap_v1.6.0.md):
  - Viewer type check: tsconfig.app.json has "strict": false and `tsc
    --noEmit -p tsconfig.app.json` reports 24 errors; tsc is not in CI.
  - Viewer bundle: 304 KB gzipped initial bundle against the 300 KB budget.
  - Principle VI attribution: overlays do not yet show the pipeline name and
    version that produced them.

Previous report (1.0.1):
Version change: 1.0.0 → 1.0.1 (2026-10-01)
Bump rationale: PATCH. Engineering Standards' CI line now names Python 3.12 and
3.13 as the supported versions, as the line itself anticipated ("3.13 added when
upstream deps allow"); spec 012-python-313-support. No principle changed.
Templates requiring updates: none.

Previous report (1.0.0):
Version change: (none) → 1.0.0
Bump rationale: First ratification. Replaces the unfilled boilerplate template
with concrete principles derived from the v1.4.2 (JOSS) project state and the
v1.5–v2.0 modularity roadmap.

Modified principles:
  - All five principles are new; no prior versions to rename.

Added sections:
  - Core Principles (5)
  - Engineering Standards
  - Development Workflow
  - Governance

Removed sections:
  - None (placeholder template only).

Templates requiring updates:
  ✅ .specify/templates/plan-template.md — "Constitution Check" gate now has
     concrete principles to evaluate against (see Governance §Plan Gating).
     The template's structure is compatible; no edits required at ratification.
  ✅ .specify/templates/spec-template.md — User-story / acceptance-scenario
     structure is compatible with Principle II (Stable Pipeline Contract);
     no edits required.
  ✅ .specify/templates/tasks-template.md — Phase organisation (Setup →
     Foundational → User Stories) is compatible with Principle V (Backward
     Compatibility by Default) and the v1.5 phased roadmap; no edits required.
  ⚠ .specify/templates/checklist-template.md — Not reviewed at ratification;
     verify alignment when first /speckit-checklist is run.

Follow-up TODOs:
  - None at ratification. Future amendments should re-run the propagation
    checklist before bumping version.
-->

# VideoAnnotator Constitution

VideoAnnotator is an open-source toolkit for automated video annotation in
behavioural, social, and health research: a Python pipeline and API server, and
the Video Annotation Viewer (`viewer/`) that displays its output. This constitution captures the
non-negotiable commitments that bind every specification, plan, and pull
request. Where principles conflict with convenience, the principle wins.

## Core Principles

### I. Local-First Execution, No Telemetry (NON-NEGOTIABLE)

All annotation processing MUST be runnable on the user's own hardware without
required network access to a cloud or third-party inference service. Pipelines
MAY offer optional remote backends (e.g. an in-house GPU server, an HPC
cluster, a self-hosted Ollama instance) but the default install MUST process
video locally end-to-end.

The viewer renders annotations and computes timelines in the user's browser.
The only network traffic it initiates is to the VideoAnnotator server the user
configured (by default, the one that served it). It loads no runtime assets
from third-party CDNs.

VideoAnnotator sends no telemetry, analytics, or error reports, from the server
or the viewer. A dependency that sends telemetry by default (e.g.
pyannote.audio's metrics) MUST be switched off unless the user turns it on.

**Rationale.** VideoAnnotator's primary users are research teams working with
recordings of children, patients, and other vulnerable populations under IRB
approval. Sending such data to an external service is an ethical and
legal non-starter for a substantial fraction of the user base. Local-first is
not a feature; it is the precondition for the toolkit being usable at all in
its intended context. A viewer or dependency that silently phones home would
breach the same protocols as a cloud inference call.

### II. Stable Pipeline Contract and Open Formats (NON-NEGOTIABLE)

The `BasePipeline` abstract base class — `initialize()`, `process()`,
`cleanup()`, `get_schema()`, and the `list[dict[str, Any]]` return shape — is
a stable interface within a major version. The pipeline registry's YAML
metadata schema is similarly stable. All pipeline outputs MUST map to
established open standards: COCO JSON for spatial annotations, RTTM for
speaker diarisation, WebVTT for timed text, JSON for everything else.

Adding new keys to the metadata schema or new optional config keys is allowed
within a minor version; renaming or removing existing keys requires a major
version bump and a migration guide.

The viewer reads exactly these formats. It accepts output from any compatible
version within the major version, degrades gracefully when optional fields are
missing, and never inserts inferred values to fill gaps. A change to what a
pipeline writes ships with the matching viewer change in the same release; the
viewer contract test (`viewer/src/test/contract/`, run in CI against real
outputs in `tests/fixtures/viewer_contract/`) is the guard.

**Rationale.** A research toolkit's value is its glue role between many
upstream models and many downstream analyses. If the glue moves, every
analysis pipeline downstream of it breaks. Open formats keep VideoAnnotator
interoperable with the wider behavioural-research ecosystem (FiftyOne, Label
Studio, ELAN, Datavyu, custom Pandas analysis) rather than forcing users into
a proprietary cul-de-sac.

### III. Provenance and Reproducibility (NON-NEGOTIABLE)

Every annotation produced by VideoAnnotator MUST carry sufficient provenance
metadata to be reproduced from the same input video on different hardware:
pipeline name, pipeline version, configuration parameters, processing
timestamps, and (where relevant for ML pipelines) model identifier, model
revision SHA, quantisation level, and sampling parameters. Pipelines are
configured **declaratively** via YAML; ad-hoc imperative configuration is not
acceptable for outputs that may end up in a published paper.

**Rationale.** VideoAnnotator output is intended to be cited in scientific
publications. "I ran VideoAnnotator on these videos" is not a reproducible
methods section. "I ran VideoAnnotator v1.5.2 with config X.yaml using
faster-whisper-large-v3-turbo at HF revision <sha>, temperature 0.0, on a
single RTX 4090" is. The toolkit MUST make the latter the default and the
former impossible.

### IV. Modular by Construction

The default `pip install videoannotator` MUST yield a slim core install that
runs the FastAPI service and CLI without any heavy ML dependency. Each
pipeline family MUST be installable as a named extra (e.g.
`pip install videoannotator[face]`) or as a separately published plugin
package. Heavy ML libraries (torch, ultralytics, pyannote, whisper, deepface,
etc.) MUST NOT be unconditional core dependencies.

The core MUST NOT import from any specific pipeline package; pipeline
discovery MUST go through the registry layer (YAML metadata + entry-point
discovery via `importlib.metadata.entry_points`).

The viewer mirrors this: each annotation modality (pose, face, scene, speech,
speaker, VLM labels, custom tracks) is a self-contained layer the user can
switch off at runtime without breaking the others. A new annotation type ships
as a new layer without modifying existing ones; layers talk to the timeline and
player through defined interfaces, never through each other's state.

**Rationale.** A 30 GB Docker image is a barrier to adoption for the very
research groups VideoAnnotator targets. Modularity is what lets a researcher
who only needs scene labelling install scene labelling, run it on a laptop,
and cite a slim install in their paper. New annotation types arrive every
release; independent layers keep each one additive, and let a reviewer isolate
what they are auditing.

### V. Backward Compatibility by Default

Within a major version (e.g. v1.x → v1.y), existing config files MUST
continue to work, existing CLI invocations MUST produce equivalent output,
and existing pipeline output schemas MUST NOT have keys renamed or removed.
Default checkpoints and model identifiers MAY be upgraded within a minor
version provided the output schema is preserved and the upgrade is documented
in release notes; users MUST be able to pin a previous checkpoint via config
if they need bit-identical behaviour for an in-flight study.

The viewer's user-facing surface is covered the same way: URL routes and
parameters, keyboard shortcuts, layer-toggle configuration keys, and the file
formats it accepts by drag-and-drop stay stable within a major version.

Breaking changes require a major version bump, a migration guide, and at
least one minor release of advance notice via deprecation warnings.

**Rationale.** Studies run for months or years. A toolkit that breaks running
analyses mid-study is a toolkit researchers can't trust. Researchers also cite
versions in methods sections and keep viewer links in shared analysis notes;
routes and shortcuts that drift break those references silently. Discipline at this
boundary is what distinguishes a research tool from a hobby project.

### VI. Faithful Annotation Display (NON-NEGOTIABLE)

What the viewer shows MUST reflect pipeline output exactly. No client-side
smoothing, interpolation, gap-filling, or confidence thresholding may create
values the pipeline did not produce, or hide values it did. Pipelines sample
frames; drawing a sample's detections until the next sampling instant (capped
by the sampling interval) creates no value and is allowed, provided the
samples' own timestamps remain inspectable. Every overlay MUST be attributable
to the pipeline name and version that produced it. Display-only conveniences
are permitted only while the raw values remain inspectable.

**Rationale.** The viewer is an audit tool for the pipeline. A viewer that
hides outliers or smooths over confidence drops makes the pipeline look more
reliable than it is, and undermines the reproducibility that Principle III
exists to guarantee.

## Engineering Standards

These are the engineering hygiene gates that every change MUST clear before
landing on `master`.

- **Testing.** Pytest suite covers unit, integration, and (where relevant)
  performance tests. Coverage MUST stay ≥ 80%. New pipelines and new public
  CLI/API surface MUST ship with tests.
- **Continuous integration.** GitHub Actions runs the full test suite on
  Ubuntu, Windows, and macOS against the supported Python versions (currently
  3.12 and 3.13). Ruff, mypy, and Trivy MUST pass
  on `master`.
- **Type safety.** Public APIs (`BasePipeline` subclasses, FastAPI handlers,
  CLI command signatures, registry helpers) are fully type-annotated and pass
  mypy without `# type: ignore` on the public boundary.
- **Security and licensing.** No proprietary or research-only weights ship as
  default model identifiers. Each upstream dependency's licence is recorded
  in the per-plugin pyproject metadata. AGPL-3.0 dependencies (currently
  Ultralytics) are isolated to dedicated plugins so the rest of the toolkit
  stays MIT-clean.
- **Documentation.** Every public-facing change includes corresponding doc
  updates. The README install matrix, the JOSS paper's claims, and the
  per-version roadmap docs MUST stay in sync with the released code.
- **Viewer.** TypeScript type-checking (`tsc --noEmit -p tsconfig.app.json`),
  ESLint, the Vitest suite (including the contract test), and the
  `scripts/build_viewer.sh --check` bundle check MUST pass in CI; new code
  avoids `any` without an inline justification. Pipeline-supplied strings are
  never rendered with `dangerouslySetInnerHTML`. The initial bundle stays
  within 300 KB gzipped; a feature pulling in a dependency over 50 KB gzipped
  is code-split. Supported browsers: the latest two stable Chrome, Firefox and
  Safari; mobile is best-effort. High-severity advisories from `bun audit`
  block a release.
- **Reproducibility plumbing.** Annotation metadata writers (see Principle
  III) are part of the public test surface; regressions in provenance fields
  fail CI, not silently degrade.

## Development Workflow

VideoAnnotator is a **sole-maintainer** project. The workflow trades formal
PR ceremony for atomic commits, clear messages, and disciplined branching.

- **Feature work via Spec Kit.** Substantial features (anything that touches
  the pipeline registry, the dispatcher abstraction, the FastAPI surface, or
  introduces a new pipeline family) MUST be developed via the speckit
  workflow: `/speckit-specify` → `/speckit-clarify` (if needed) →
  `/speckit-plan` → `/speckit-tasks` → `/speckit-implement`. Specs live under
  `specs/<NNN>-<slug>/`.
- **Branching.** One long-lived feature branch per release theme (e.g.
  `v1.5-modularity`, `v1.6-ux`); commit atomically and frequently directly
  on the release branch. Sub-branches per phase or per task are NOT required
  and SHOULD be avoided. Sub-branches are appropriate only for genuine
  experiments that may be discarded.
- **Commit messages.** Conventional structure: `<area>: <imperative summary>`
  (e.g. `paper: fix ORCID format`, `pipelines: bump default whisper backend`).
  Body explains *why*, not *what* the diff already shows. Co-author trailers
  are used for AI-assisted commits.
- **One product, one version.** The viewer shares VideoAnnotator's version
  number and changelog (`tests/unit/test_versions_match.py` checks). A change
  to an output format and the viewer's handling of it land together, in the
  same commit where practical and always in the same release.
- **Release versioning.** Semantic versioning. Patch = bug fixes only. Minor
  = new pipelines, new model defaults (with output-schema preserved),
  internal refactors. Major = breaking changes to `BasePipeline`, the
  metadata schema, or output formats.
- **JOSS-stable master.** While VideoAnnotator is under JOSS review, `master`
  MUST remain installable and runnable matching the JOSS-submitted version.
  Speculative or incomplete refactors live on feature branches until ready
  to merge with passing CI.
- **Skipping hooks.** `--no-verify` on git operations is forbidden except in
  documented emergencies; pre-commit hook failures are fixed at the source,
  not bypassed.

## Governance

This constitution supersedes all other practices. When the roadmap, the
README, or any other doc conflicts with the constitution, the constitution
wins and the conflicting doc gets updated.

### Amendment procedure

1. Open a discussion (issue or in-line comment in `.specify/memory/`) naming
   the principle to amend and the motivating evidence.
2. Update `.specify/memory/constitution.md` via the `/speckit-constitution`
   command. The command produces the Sync Impact Report (HTML comment at the
   top of this file) automatically.
3. Bump the constitution version per the rules below.
4. Run the propagation checklist: re-read the four `.specify/templates/*.md`
   files and any agent guidance docs (`CLAUDE.md`, `AGENTS.md`); update them
   if a principle was added, removed, or materially redefined.
5. Commit the amendment under a `docs(constitution)` subject line.

### Versioning policy for the constitution itself

- **MAJOR** — A principle is removed or its meaning is materially redefined
  in a way that invalidates existing specs/plans.
- **MINOR** — A new principle or section is added, or an existing principle
  is materially expanded with new requirements.
- **PATCH** — Wording clarifications, typo fixes, non-semantic edits.

### Plan Gating

The `Constitution Check` gate in `.specify/templates/plan-template.md` MUST
verify, at minimum, that the proposed plan:

- Preserves local-first default execution (Principle I).
- Does not break the `BasePipeline` contract or rename existing output keys
  within a minor version (Principles II, V).
- Adds provenance metadata for any new annotation type (Principle III).
- Displays annotations without creating or hiding values, and attributes each
  overlay to its pipeline (Principle VI).
- Routes any new heavy ML dependency through an extras group or plugin
  package (Principle IV).
- Includes tests, CI hygiene, and documentation updates per Engineering
  Standards.

A plan that violates any of the above MUST either revise the design until
the gate passes, or document the violation under "Complexity Tracking" with
a justification reviewable against this constitution at the next amendment
cycle.

### Compliance review

The maintainer reviews the constitution against on-the-ground reality at the
start of each minor-version planning cycle (e.g. when drafting
`docs/development/roadmap_v1.X.0.md`). If a principle has drifted from
practice, either the practice is corrected or the principle is amended; the
gap is not allowed to persist.

**Version**: 1.1.0 | **Ratified**: 2026-05-06 | **Last Amended**: 2026-10-02
