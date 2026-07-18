<!--
SYNC IMPACT REPORT
==================
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

VideoAnnotator is an open-source Python toolkit for automated video annotation
in behavioural, social, and health research. This constitution captures the
non-negotiable commitments that bind every specification, plan, and pull
request. Where principles conflict with convenience, the principle wins.

## Core Principles

### I. Local-First Execution (NON-NEGOTIABLE)

All annotation processing MUST be runnable on the user's own hardware without
required network access to a cloud or third-party inference service. Pipelines
MAY offer optional remote backends (e.g. an in-house GPU server, an HPC
cluster, a self-hosted Ollama instance) but the default install MUST process
video locally end-to-end.

**Rationale.** VideoAnnotator's primary users are research teams working with
recordings of children, patients, and other vulnerable populations under IRB
approval. Sending such data to an external service is an ethical and
legal non-starter for a substantial fraction of the user base. Local-first is
not a feature; it is the precondition for the toolkit being usable at all in
its intended context.

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

**Rationale.** A 30 GB Docker image is a barrier to adoption for the very
research groups VideoAnnotator targets. Modularity is what lets a researcher
who only needs scene labelling install scene labelling, run it on a laptop,
and cite a slim install in their paper.

### V. Backward Compatibility by Default

Within a major version (e.g. v1.x → v1.y), existing config files MUST
continue to work, existing CLI invocations MUST produce equivalent output,
and existing pipeline output schemas MUST NOT have keys renamed or removed.
Default checkpoints and model identifiers MAY be upgraded within a minor
version provided the output schema is preserved and the upgrade is documented
in release notes; users MUST be able to pin a previous checkpoint via config
if they need bit-identical behaviour for an in-flight study.

Breaking changes require a major version bump, a migration guide, and at
least one minor release of advance notice via deprecation warnings.

**Rationale.** Studies run for months or years. A toolkit that breaks running
analyses mid-study is a toolkit researchers can't trust. Discipline at this
boundary is what distinguishes a research tool from a hobby project.

## Engineering Standards

These are the engineering hygiene gates that every change MUST clear before
landing on `master`.

- **Testing.** Pytest suite covers unit, integration, and (where relevant)
  performance tests. Coverage MUST stay ≥ 80%. New pipelines and new public
  CLI/API surface MUST ship with tests.
- **Continuous integration.** GitHub Actions runs the full test suite on
  Ubuntu, Windows, and macOS against the supported Python versions (currently
  3.12; 3.13 added when upstream deps allow). Ruff, mypy, and Trivy MUST pass
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

**Version**: 1.0.0 | **Ratified**: 2026-05-06 | **Last Amended**: 2026-05-06
