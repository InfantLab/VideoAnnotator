# Handover: Applying VideoAnnotator Conventions in video-annotation-viewer

**Status:** Active reference
**Created:** 2026-05-06
**Audience:** Future Claude Code session (or a human) working in [video-annotation-viewer](https://github.com/InfantLab/video-annotation-viewer), the companion frontend to VideoAnnotator.

## Context

VideoAnnotator (this repo) and video-annotation-viewer are cited together in the JOSS submission and are designed to be used as a pair. The conventions established here in May 2026 — Spec Kit setup, project constitution, branching model, doc layout — should be mirrored in the viewer repo so future work, specs, and roadmap docs stay consistent across both repositories.

This document is a handover so a fresh Claude Code session in the viewer repo (or you working solo) can replicate the setup. It is deliberately concrete: commands, file paths, and a translated constitution sketch.

## What's been set up in VideoAnnotator (May 2026)

- **Spec Kit 0.8.6.dev0** installed via:
  ```
  uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
  specify init --here --integration claude --force --ignore-agent-tools
  ```
  Skills now live at `.claude/skills/speckit-*/`. Slash commands use hyphens: `/speckit-constitution`, `/speckit-specify`, `/speckit-plan`, `/speckit-tasks`, `/speckit-implement`, plus optional `/speckit-clarify`, `/speckit-analyze`, `/speckit-checklist`, and the `/speckit-git-*` family added with the v0.8 git extension.

- **Constitution v1.0.0** at `.specify/memory/constitution.md`. Five core principles (Local-First Execution, Stable Pipeline Contract, Provenance & Reproducibility, Modular by Construction, Backward Compatibility) plus Engineering Standards, Development Workflow, and Governance sections. First three principles are NON-NEGOTIABLE.

- **Roadmap docs** at `docs/development/roadmap_v{1.5,1.6,1.7_to_v2.0}.md`; archived predecessor at `docs/archive/development/roadmap_v1.5.0_original.md`. Pattern: one current per active version, archive the rest.

- **Branching model**: one feature branch per release theme (e.g. `v1.5-modularity`), atomic commits directly on it, no per-phase sub-branches, JOSS-stable master.

- **Commit message style**: `<area>(scope?): <imperative summary>`. Examples: `paper: fix ORCID format`, `chore(speckit): refresh to 0.8.6.dev0`, `docs(constitution): ratify v1.0.0`. Body explains *why*. Co-author trailer for AI-assisted commits.

- **Gitignore additions**: `.claude/settings.local.json`, `.claude/.credentials.json`, `.claude/state/` (keep `.claude/skills/` and project `.claude/settings.json` shared).

## First session in video-annotation-viewer (recommended order)

```bash
# 1. Install speckit
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git
specify init --here --integration claude --force --ignore-agent-tools

# 2. (LINUX/macOS) Set executable bits on shipped scripts before committing.
#    Spec-kit ships scripts without +x, which collides with most pre-commit
#    configs that include check-shebang-scripts-are-executable.
chmod +x .specify/extensions/git/scripts/bash/*.sh \
         .specify/extensions/git/scripts/powershell/*.ps1 \
         .specify/scripts/bash/setup-tasks.sh
git add -A
git update-index --chmod=+x \
  .specify/extensions/git/scripts/bash/*.sh \
  .specify/extensions/git/scripts/powershell/*.ps1 \
  .specify/scripts/bash/setup-tasks.sh

# 3. Add to .gitignore
cat >> .gitignore <<'EOF'

# Claude Code (per spec-kit init recommendation; keep skills/ and project settings.json shared)
.claude/settings.local.json
.claude/.credentials.json
.claude/state/
EOF

# 4. Commit the speckit setup
git commit -m "chore(speckit): bootstrap to 0.8.6.dev0 with claude integration

(see VideoAnnotator commit c38ff05 for the parallel setup)"

# 5. Run /speckit-constitution to ratify v1.0.0 (drafting suggestions below).
```

## Translated constitution principles for the viewer

The Python-server principles need adapting for a React/TypeScript frontend. Suggested translations — the actual constitution should be ratified via `/speckit-constitution` so the Sync Impact Report machinery runs:

| VideoAnnotator (server) | video-annotation-viewer (frontend) |
|---|---|
| **I. Local-First Execution** (NON-NEGOTIABLE) — annotation processing runs on user hardware. | **I. Local-First Rendering & No Telemetry** (NON-NEGOTIABLE) — all rendering happens client-side; no analytics, no telemetry, no third-party CDN for runtime; the viewer connects only to a user-controlled VideoAnnotator API endpoint. |
| **II. Stable Pipeline Contract & Open Formats** (NON-NEGOTIABLE) — `BasePipeline` ABC + COCO/RTTM/WebVTT/JSON. | **II. Stable API Consumption Contract** (NON-NEGOTIABLE) — viewer pins to documented VideoAnnotator API endpoints; gracefully degrades on missing optional fields; never silently inserts inferred values; same open formats. |
| **III. Provenance & Reproducibility** (NON-NEGOTIABLE) — every annotation carries pipeline version, config, model SHA. | **III. Faithful Annotation Display** (NON-NEGOTIABLE) — display reflects upstream pipeline output exactly; per-overlay attribution (pipeline name + version) visible in UI; no client-side smoothing or interpolation that hides upstream data. |
| **IV. Modular by Construction** — slim core install, pipelines via extras. | **IV. Composable View Layers** — each annotation type (pose, faces, scenes, captions, audio) is a self-contained, independently toggleable layer; new types ship as new layers without modifying existing ones. |
| **V. Backward Compatibility by Default** — config files and output schemas stable within major versions. | **V. API & UI Stability** — within a major, viewer accepts annotations from any compatible VideoAnnotator API version; URL params, keyboard shortcuts, layer-toggle config remain stable. |

**Engineering Standards** translated: Vitest/Playwright instead of pytest; ESLint + tsc strict instead of ruff + mypy; npm-audit / Snyk instead of Trivy; bundle-size budget (e.g. ≤300 KB gzipped initial) instead of Docker-image size budget.

**Development Workflow** stays nearly identical — speckit for substantial features, one branch per release theme, atomic commits, semver, no `--no-verify`. The only change: "JOSS-stable master" applies the same way (the viewer is also cited in the paper).

## Cross-repo coordination rules

Once both repos have constitutions, a few things need consistent handling:

- **API version compatibility matrix.** Each viewer release documents the VideoAnnotator API versions it supports (in viewer's `README.md` or `docs/compatibility.md`), and each VideoAnnotator release documents the viewer versions it has been tested against. Tests live in whichever repo introduced the change.
- **Shared schema source of truth.** The annotation output schemas (COCO, RTTM, WebVTT, the JSON envelope) are defined in VideoAnnotator. The viewer consumes them. Schema bumps in VideoAnnotator require a viewer update before either ships in a paper-cited release.
- **Coupled major-version cuts.** A breaking change in either repo's contract should drive a major bump in both, with coordinated release notes. Avoid cases where viewer 2.0 and VideoAnnotator 2.0 silently disagree on what an annotation looks like.
- **JOSS paper updates.** Cite-pair: any future revision to the paper should bump both versions together if either has a non-trivial release between rounds.

## Files worth carrying over verbatim

These VideoAnnotator files are reasonable starting templates for the viewer (with the translations above applied):

- `.specify/memory/constitution.md` — adapt principles, keep Engineering Standards / Development Workflow / Governance structure unchanged.
- `docs/development/roadmap_v1.5.0.md` structure — phase plan, effort summary table, success criteria, what-stays-unchanged section. The viewer is at v0.6.2 per the JOSS bib, so its first roadmap doc would be `roadmap_v0.7.0.md` or `roadmap_v1.0.0.md` depending on how you want to version (semver-strict from now is the usual call once a JOSS paper cites a version).
- The two-commit pattern — speckit setup as `chore(speckit)`, constitution ratification as `docs(constitution)` — is worth keeping for symmetry.

## Known gotchas

- **Pre-commit + spec-kit shipped scripts.** The `check-shebang-scripts-are-executable` hook fails on freshly-init'd spec-kit scripts because upstream ships them without +x. Either chmod and `git update-index --chmod=+x` (see step 2 above), or exclude `.specify/extensions/` from that hook in `.pre-commit-config.yaml`. The latter is the cleaner long-term fix and means future `specify init` refreshes are painless.
- **`chmod +x` filesystem ≠ git stored mode.** `git update-index --chmod=+x` is required to update what git records, regardless of filesystem permissions. The pre-commit hook checks the git index, not the filesystem.
- **Pre-commit `[INFO] Restored changes from patch...` looks like success but is not.** It just means pre-commit restored unstaged changes from its stash after running. Always confirm the commit actually landed via `git log --oneline -1`.
- **Slash command naming changed in spec-kit 0.8.** Old form `/speckit.specify` (dot) is gone; new form `/speckit-specify` (hyphen). If older docs reference the dot form, update them.

## What this handover is NOT

- It's not a viewer roadmap. That should be drafted in-situ once the viewer codebase is reviewed (use the same Plan-agent architecture-review approach that produced this repo's v1.5 roadmap).
- It's not a binding constitution for the viewer. That gets ratified by `/speckit-constitution` in the viewer repo with whatever content the maintainer decides; the table above is a sketch.
- It's not an API contract spec. The shared schemas should be documented separately, probably in VideoAnnotator's `docs/development/api-contract.md` or a new `docs/contracts/` directory cited from both repos.

## Reference commits in VideoAnnotator

- `c38ff05` — `chore(speckit): refresh to 0.8.6.dev0 and migrate prompts to skills`
- `7cbd4ed` — `docs(constitution): ratify v1.0.0 with five core principles and governance`
- `0bfbea2` — `docs: rework v1.5+ roadmap around modularity and model currency`

These three together capture the conventions to mirror.
