# Video Annotation Viewer Constitution

## Core Principles

### I. Local-First Rendering & No Telemetry (NON-NEGOTIABLE)

All annotation rendering and timeline computation happens in the user's
browser. The viewer ships no analytics, no telemetry, no error-reporting
beacons, and no third-party CDN dependencies for runtime assets. The only
network traffic the application initiates is to a user-controlled
VideoAnnotator API endpoint that the user has explicitly configured, plus
the static asset host serving the build itself.

**Rationale:** Behavioral and developmental research videos frequently
contain identifiable participants (often children) under data-protection
regimes (GDPR, POPIA, IRB protocols) that forbid third-party processing.
A viewer that silently phones home would be unusable in the contexts the
companion VideoAnnotator pipeline targets.

### II. Stable API Consumption Contract (NON-NEGOTIABLE)

The viewer pins to documented VideoAnnotator API endpoints and to the
documented open formats (COCO keypoints, RTTM speaker segments, WebVTT
captions, the JSON envelope). Within a major version it accepts
annotations from any compatible VideoAnnotator API version, degrades
gracefully on missing optional fields, and never silently inserts
inferred values to fill gaps. Schema bumps in the upstream pipeline
require a viewer update before either side ships in a paper-cited
release.

**Rationale:** The viewer and VideoAnnotator are cited as a pair in JOSS.
A viewer that quietly diverges from the documented schemas — or invents
data to paper over upstream changes — would defeat the auditing workflow
the paper claims it supports.

### III. Faithful Annotation Display (NON-NEGOTIABLE)

What the viewer shows must reflect upstream pipeline output exactly. No
client-side smoothing, interpolation, gap-filling, or confidence
thresholding may hide upstream values from the user. Each overlay must
carry visible attribution (pipeline name + version) so a reviewer
auditing a frame can identify which model produced what they see.
Display-only conveniences (e.g. a smoothed playback indicator) are
permitted only when the underlying raw values remain inspectable.

**Rationale:** The whole point of an audit tool is to surface what the
upstream pipeline actually did. A "helpful" viewer that hides outliers
or smooths over confidence drops makes the pipeline look more reliable
than it is and undermines the reproducibility claim of the paper.

### IV. Composable View Layers

Each annotation modality (pose, face, scene, speech, speaker, custom
JSON tracks) is implemented as a self-contained, independently
toggleable layer. Adding a new annotation type ships as a new layer
without modifying existing ones. Layers communicate with the timeline
and player through well-defined interfaces, never by reaching into each
other's state. A user must be able to disable any layer at runtime
without breaking the rest.

**Rationale:** New annotation types arrive on every VideoAnnotator
release. Coupling layers makes each new format a refactor; keeping them
independent makes additions additive and lets reviewers isolate what
they're auditing.

### V. API & UI Stability

Within a major version, the viewer accepts annotations from any
compatible VideoAnnotator API version, and its user-facing surface — URL
parameters, keyboard shortcuts, layer-toggle config keys, file-format
expectations on drag-and-drop — remains stable. Breaking changes to any
of these require a major-version bump in lockstep with the companion
pipeline, with coordinated release notes.

**Rationale:** Researchers cite specific viewer versions in
methods sections and embed viewer URLs in shared analysis notes. URL
schemes and keyboard shortcuts that drift between minor versions break
those external references silently.

## Engineering Standards

**Type safety.** TypeScript with `strict: true` in `tsconfig.json`;
`tsc --noEmit` is part of CI and must pass before merge. No `any` in
new code without an inline justification comment.

**Tests.** Vitest for unit tests (parsers, utilities, hooks); Playwright
for end-to-end flows that exercise overlay rendering against fixture
videos. Parser regressions and overlay correctness are the priority
coverage areas. New annotation layers ship with parser tests for at
least one happy-path and one degraded-input fixture.

**Lint & format.** ESLint with the project's flat config, plus
Prettier-equivalent formatting. CI fails on lint errors. Warnings are
allowed but tracked.

**Security.** `npm audit` (or equivalent via `bun audit` / Snyk) is run
on every release branch; high-severity advisories block release. The
viewer takes no untrusted code execution paths and must not introduce
`dangerouslySetInnerHTML` against pipeline-supplied strings.

**Bundle size.** Initial gzipped bundle budget is 300 KB; route-level
code-splitting is required for any feature that pulls in a transitive
dependency above 50 KB gzipped. Bundle size is tracked per release.

**Browser support.** Latest two stable versions of Chrome, Firefox, and
Safari. Mobile Safari and Android Chrome are best-effort; desktop is the
supported audit surface.

## Development Workflow

**Spec Kit for substantial work.** Anything beyond a one-file fix runs
through `/speckit-specify` → `/speckit-plan` → `/speckit-tasks` →
`/speckit-implement`. Optional `/speckit-clarify`, `/speckit-analyze`,
and `/speckit-checklist` are used when ambiguity or cross-artifact
consistency warrants it.

**Branching.** One feature branch per release theme (e.g.
`v0.7-modular-layers`), atomic commits directly on it, no per-phase
sub-branches. Master remains JOSS-stable: the version cited in the
paper must always be reachable and runnable from `main`.

**Commits.** `<area>(scope?): <imperative summary>` style. Body explains
*why*. AI-assisted commits include a `Co-Authored-By` trailer. Hooks
are never bypassed (`--no-verify` is forbidden); if a hook fails, the
underlying issue is fixed.

**Versioning.** Semver from v1.0.0 onward. Pre-1.0 (the current state
at v0.6.x) follows semver-spirit: minor bumps may include breaking
changes but are documented as such in `CHANGELOG.md`. The first release
after the JOSS paper is accepted is v1.0.0.

**Release coordination.** A breaking change in either repo's contract
drives a major bump in both, with coordinated release notes. The
viewer's `README.md` documents the VideoAnnotator API versions each
release supports.

## Governance

This constitution supersedes ad-hoc practices. All PRs and reviews must
verify compliance with the principles above; complexity that appears to
violate a principle must be justified inline (in the PR description or
a code comment) or the PR must include a constitutional amendment.

Amendments require:

1. A PR modifying this file with a `**Why:**` section explaining the
   driver.
2. A version bump per the rules below.
3. An update to the `**Last Amended**` date.

Amendment versioning follows semver applied to the constitution itself:

- **MAJOR** — a NON-NEGOTIABLE principle is removed, redefined in a way
  that changes its scope, or a new NON-NEGOTIABLE principle is added.
- **MINOR** — a new non-mandatory principle or section is added, or
  existing guidance is materially expanded.
- **PATCH** — clarifications, wording fixes, typo corrections, or
  refinements that do not change intent.

The companion VideoAnnotator constitution at
`InfantLab/VideoAnnotator:.specify/memory/constitution.md` is the
authoritative source for principles that govern the upstream pipeline.
This document governs the viewer only; cross-repo coordination rules
live in `docs/development/handover_to_viewer.md`.

**Version**: 1.0.0 | **Ratified**: 2026-05-06 | **Last Amended**: 2026-05-06
