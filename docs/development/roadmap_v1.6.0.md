# 🚀 VideoAnnotator v1.6.0 Development Roadmap

## Release Overview

VideoAnnotator v1.6.0 is the **Public Release**: the first version we announce to the developmental
research community, with a tutorial, mailing-list posts and a webinar.

v1.5.0 made the tool usable by us without a terminal: in-app pipeline installs, batches, folder
ingest, the VLM pipeline, and the viewer bundled at `/viewer` (see
[`roadmap_v1.5.0.md`](roadmap_v1.5.0.md)). v1.6.0 is what it takes for **a lab we have never met**
to install it, run it on their own videos, and get the results into their own analysis without
asking us.

**Why this shape**: on 2026-09-24 the JOSS submission
([#10182](https://github.com/openjournals/joss-reviews/issues/10182)) was sent for an editorial
scope check because no one outside the author team has used or contributed to it yet. JOSS accepts
publications, outside use, integration with other research tools, or benchmarks with reproducible
materials as evidence. v1.6.0 is built to produce all four: pilot labs start on release candidates,
results export to ELAN, and every default model has a published benchmark score.

> Until 2026-09-24 this file held the plugin-ecosystem plan. That plan is now
> [`roadmap_v1.7.0.md`](roadmap_v1.7.0.md), with its git history.

**Target Release**: in time for the early-2027 conference season (BCCCD, 7–9 Jan 2027)
**Current Status**: Planning Phase
**Main Goal**: an outside lab can go from install to results in R or Python, on its own videos,
without help
**Prerequisites**: v1.5.0 tagged, after spec 011's SC-001 manual run
(`tests/manual/pipeline_readiness_e2e.md`)

---

## 🎯 Core Principles

- ✅ **A stranger can do it.** Each feature is judged by whether a researcher outside the team can
  use it on their own machine with only the docs.
- ✅ **Local by default.** No video or frame leaves the user's machine unless they switch on a hosted
  backend, and switching one on says so plainly.
- ✅ **Results leave in formats researchers already use**: CSV/Parquet for R and pandas, `.eaf` for
  ELAN.
- ✅ **Agents are users too.** Anything a person can do in the viewer, an agent can do through the
  CLI or MCP.
- ✅ **No model change without a score.** A pipeline default changes only when the benchmark says the
  new one is better.
- ✅ **Pilot on release candidates.** Outside labs start at rc1, not at the final release.

---

## 📋 v1.6.0 Deliverables

Phases are in order. Phases 0–3 make release candidate 1; the pilot runs from there to release and
reorders Phases 4–7.

### Phase 0: One Repository

**Problem**: VideoAnnotator and Video Annotation Viewer are one product in two repositories. The
viewer is built in one repo and hand-copied into `src/videoannotator/viewer_static/` in the other.
That copy went stale twice (July to August, then August to September) and hid finished features
from users both times. Cross-cutting work needs paired commits, handoff documents that went
unactioned (spec 008's `viewer-handoff.md`), two changelogs and two CI setups, and debugging
happens from outside either repo.

**Solution**:
- [ ] Import `video-annotation-viewer` into `viewer/` with its full history (rewrite paths with
      `git filter-repo --to-subdirectory-filter viewer`, then merge with
      `--allow-unrelated-histories`), taken from `feature/vlm-annotation-support`.
- [ ] One build step produces `viewer_static/` from `viewer/`. CI fails when the committed bundle
      doesn't match the source, so the copy can't be forgotten.
- [ ] One dev command runs the API server and the Vite dev server (API proxied, hot reload), plus
      VS Code tasks for it.
- [ ] The viewer takes VideoAnnotator's version number from v1.6.0: one product, one version, one
      changelog. Standalone mode (drop in files, no server) stays, and is still published.
- [ ] The TypeScript half of the viewer contract test, deferred in v1.5.0, runs in CI.
- [ ] One JS package manager (the viewer has both `bun.lock` and `package-lock.json`).
- [ ] Merge the two `AGENTS.md`/`CLAUDE.md` files, move the viewer's open issues, archive the old
      repo with a pointer README, and update links in `README.md`, `paper/paper.md` and
      `CITATION.cff`.

**Why first**: every later phase writes docs and tutorials full of repository links. Moving after
that means rewriting them, and breaking links pilot labs have already saved.

---

### Phase 1: Housekeeping — Pipelines and Dependencies

**Problem**: both the set of pipelines and most version pins are historical, not decisions. The
pipelines are whatever got added along the way, and no one has asked which tools a developmental
researcher actually needs. Most pins are the same: `torch==2.6.0` is pinned exactly in
five extras groups; `pyannote.audio<4.0`, `transformers<5.0` and `opencv-python<5.0` were
defensive bounds against breakage we never tested for; Python is locked to 3.12
(`>=3.12,<3.13`), which has had security fixes only since 2025 and reaches end of life in October
2028. Every one of these gets more expensive after release: once outside labs have results, an
upgrade that changes outputs breaks their comparisons. With no users yet, now is the cheapest time
it will ever be.

**Already known about dependencies** (uv resolution of every extras group with the historical pins
lifted, 2026-09-26):
- Python 3.13 resolves cleanly for every group, with current releases (torch 2.14, pyannote.audio
  4, transformers 5, TensorFlow 2.21).
- Python 3.14 resolves for every group except `face`: deepface needs TensorFlow, which has only a
  release candidate for 3.14.
- `openface-test` must stay at `==0.1.13`: every later release pins Pillow 9.4, numpy 1.26 and
  scipy 1.13, which have to be built from source on 3.13+.
- pyannote.audio 4 changes the pipeline API, reads audio through torchcodec (FFmpeg shared
  libraries at runtime) and brings `speaker-diarization-community-1`, the free open-weights
  successor to `speaker-diarization-3.1` (the paid "precision" models are pyannoteAI's hosted
  service and stay out: local by default).
- A newer torch leaves the cu124 wheel index in `[tool.uv.sources]`, which raises the minimum GPU
  driver.

**Also known about the pipelines** (Hugging Face, 2026-09-26):
- LAION Empathic-Insight Voice (`laion_voice`): the model repositories are 16 GB (Small) and
  32 GB (Large), last updated in May 2025, with 1 and 4 downloads in the last 30 days. The face
  models (`face_laion_clip`) are small (30 MB / 300 MB) but had no downloads in that period. Both
  look abandoned upstream.
- deepface (`face_analysis`) is the only reason TensorFlow is in the dependency tree, and so the
  only thing holding back Python 3.14. Its age and gender outputs say nothing useful about infants.
- `face_openface3_embedding` installs OpenFace 3 through `openface-test`, a third-party
  repackaging whose releases after 0.1.13 pin old Pillow, numpy and scipy.

**Solution**, in four steps:
- [ ] **Pipeline review** (`docs/development/pipeline_review_v1.6.0.md`). Start from what the
      field asks, not from what we have: caregiver speech, infant vocalisations, who is speaking,
      faces and expressions, movement, gaze and joint attention, touch. For each question, which
      tool answers it best today. For each existing pipeline: which question it answers, whether
      its upstream is maintained, download and install cost, licence, how it does on infants and
      young children, and overlap with other pipelines. Decide for each: keep, replace, mark
      experimental, or drop (a dropped pipeline can come back as a v1.7.0 plugin). The review sets
      the scope of the audit and of Phase 5's model work, and gives Phase 7 its "which pipeline
      for which question" page. With no users yet, dropping a pipeline breaks no one.
- [ ] **Dependency audit** (`docs/development/dependency_audit_v1.6.0.md`), for the pipelines the
      review keeps. For every direct dependency of the core package, each extras group and the
      viewer (merged in Phase 0): current constraint,
      latest release, why it is pinned (git history, CHANGELOG), what upgrading changes (API,
      outputs, install: wheels, CUDA, system libraries), and a decision: upgrade, keep with a
      stated reason, replace, or drop. Also covers the Python version, the CUDA index and driver
      floor, Docker base images, Node and the JS package manager, pre-commit hooks and GitHub
      Actions versions.
- [ ] **Specs**: one spec-kit spec per coherent change the audit calls for (`/speckit-specify`).
      Expected, subject to the review and audit: removing dropped pipelines; Python 3.13 (with a
      3.14 CI job, made required once TensorFlow ships for it, or once deepface is dropped) and the
      core dependencies; the torch stack and CUDA index; the pyannote.audio 4 migration; the
      viewer's dependencies.
- [ ] **Implement** the specs before rc1. Re-baseline the v1.4.x acceptance fixtures once, on
      purpose, and record the before/after differences on the demo video in the CHANGELOG.

**Library versions vs default models**: this phase upgrades libraries. Switching a pipeline's
*default model* (for example to `speaker-diarization-community-1`) still waits for the benchmark in
Phase 5. If a library upgrade can't keep the current default model runnable, the audit says so and
the spec decides.

**Why second**: the install instructions, tutorials and docs written from Phase 2 onwards name the
pipelines, the Python version, the extras and the system requirements. Changing those after writing
the docs means rewriting them.

---

### Phase 2: Clean First Contact

**Problem**: the README and docs are written for developers and have drifted. The README's
quickstart still installs pipelines with `curl` and says every install needs a restart, both
replaced by spec 011. Internal team material sits in the public docs. The viewer shows a
half-built page.

**Solution**:
- [ ] README rewritten for researchers: what it does, one screenshot, install in three steps per
      OS, open the viewer.
- [ ] Internal material out of the public docs: `docs/testing/` team handoffs (`Jerry-issues.md`,
      `Issues for Server Team.md`, `TEAM_HANDOFF_PACKAGE.md` and others) and `docs/Figure 1.docx`
      move to `docs/archive/` or go.
- [ ] Docs site (mkdocs-material on GitHub Pages) built from `docs/`, archive excluded.
- [ ] No placeholder pages. The Datasets page says "Coming Soon" behind a disabled button, although
      spec 007's backend shipped in v1.5.0: wire it up or hide it.
      (Preset load/save in the job wizard landed in v1.5.0; the Datasets page, saved datasets and
      import/export from spec 007's viewer handoff are what's left.)
- [ ] Viewer: zero `tsc --noEmit` errors (24 on 2026-09-26), with typechecking in CI.
- [ ] Viewer: one function decides which pipeline produced a file. Today there are four
      (`merger.ts`, `fileUtils.ts`, and two arrays in `FileUploader.tsx`) and they disagree.
- [ ] A structural pass along the one path users take: install → add videos → run → review →
      export. Anything off that path moves or goes. A Playwright "first-time user" run on a clean
      machine files each point of friction as an issue.

**Not in this phase**: a visual redesign. Decide on one after the pilot, from what outside users
say.

---

### Phase 3: Results Out

**Problem**: each pipeline writes its own native format (COCO, WebVTT, RTTM, scene JSON). That is
right for provenance and wrong for analysis: to get one table in R, a researcher has to write a
parser per pipeline.

**Solution**:
- [ ] **Tidy export**: one row per event (video, pipeline, track or person, start, end, label,
      value, confidence, model), as CSV and Parquet, per job and per batch or dataset, from the
      API, the CLI and a viewer button. This table also feeds the corpus view (Phase 6) and agents
      (Phase 4).
- [ ] **ELAN export**: `.eaf` with one tier per pipeline track. The viewer already parses `.eaf`
      (`src/lib/parsers/elan.ts`), so this closes the loop with the tool many labs already code in.
- [ ] **Methods paragraph**: `GET /api/v1/jobs/{id}/methods`, a CLI command and a viewer button.
      Returns the models, versions, weight revisions, parameters and citations as prose plus
      BibTeX, ready for a methods section. Needs the provenance fields from the reproducibility
      item in [`roadmap_v1.7_to_v2.0.md`](roadmap_v1.7_to_v2.0.md), pulled forward: model revision
      SHA, prompt SHA-256, quantisation.
- [ ] Two worked analysis notebooks, one R and one Python, reading the tidy export of the demo
      video.

**→ Release candidate 1 (`v1.6.0rc1`)**: Phases 0–3 done. The pilot starts.

---

### Pilot (from rc1 to release)

- [ ] 3–5 labs outside the team (conference contacts first) run it on about ten of their own
      videos, with a concrete ask: tell us where it broke.
- [ ] Their reports go in the public issue tracker. Label starter tasks `good first issue`.
- [ ] Pilot feedback reorders Phases 4–7.
- [ ] With permission, record who used it and for what, for the JOSS research impact statement.

---

### Phase 4: Agents

**Problem**: researchers increasingly work alongside an agent of their own (Claude Code, Codex,
Gemini CLI and others). The REST API is complete but an agent has to discover it from scratch, and
parts of the CLI still prompt interactively.

**Solution**:
- [ ] **MCP server**: `videoannotator mcp` (stdio), and the same tools over HTTP on a running server.
      Tools: pipeline readiness, submit a folder or batch, job and batch status, annotations for a
      time window (tidy rows), the frame at time *t* as an image, comparison with an `.eaf` file,
      and the methods paragraph. With frames plus annotations an agent can audit ("look at the ten
      moments where the speaker label and the face on screen disagree").
- [ ] **Agent Skill** shipped in the repo (`SKILL.md`): installing, choosing pipelines for a
      research question, reading outputs, auditing disagreements.
- [ ] **A CLI that agents can drive**: `--json` on every command, no prompts when the flags are given
      (`generate-token` still prompts), documented exit codes.
- [ ] `llms.txt` and a "Using VideoAnnotator with your agent" docs page. It says plainly that frames
      sent to a cloud-hosted agent leave the machine.

---

### Phase 5: Models

#### 5a. Connectors

- [ ] **OpenAI-compatible connector** alongside the Ollama one (`backends: [ollama,
      openai_compatible]`), moved here from v1.7.0. One client covers Ollama, llama.cpp, vLLM,
      LM Studio, SGLang and hosted APIs.
- [ ] **Structured outputs**: the prompt carries a JSON schema, so VLM labels come back parseable
      rather than as free text.
- [ ] **Video input** through vLLM (Qwen-VL models) as a third sampling mode beside single frame and
      burst. Ollama's Qwen3-VL has no video input as of September 2026.
- [ ] Hosted endpoints are off by default. Enabling one shows a plain warning that frames leave the
      machine; for infant video that is an ethics-approval question.

#### 5b. Benchmark first

- [ ] `videoannotator benchmark`: runs chosen pipelines on a benchmark set that has human codes, and
      writes a score table per pipeline and model version.
- [ ] The benchmark set: 10–20 clips we have consent to share, with human codes. The Peekaboo demo
      clip is one. The limit here is data, not code.
- [ ] A published score for every default model, in the docs.

#### 5c. Refresh defaults (each only if the benchmark agrees)

- [ ] **Face** (if the Phase 1 review keeps a deepface-based pipeline): replace the default OpenCV
      Haar-cascade detector (`face_pipeline.py`, `detector_backend: opencv`) with YuNet or
      RetinaFace. Turn age and gender off by default: the models were trained on adults and say
      nothing useful about infants. Choose the emotion model using Uwerikowe et al.'s comparison of
      facial-emotion models on caregiver–child video.
- [ ] **Speech**: `openai-whisper` (built from source at install) → faster-whisper with
      large-v3-turbo.
- [ ] **Diarization**: pyannote `speaker-diarization-3.1` → `speaker-diarization-community-1`,
      which mainly improves speaker counting and keeps speaker identity consistent across a
      recording. The pyannote.audio 4 library upgrade itself happens in Phase 1.
- [ ] **Person**: `yolo11n-pose` → `yolo26n-pose`.
- [ ] **OpenFace 3**: confirm `openface-test` is the maintained distribution (the Phase 1
      audit keeps it at `==0.1.13`).

#### 5d. New pipelines: whichever the pilot labs ask for

Candidates, none committed (the Phase 1 pipeline review may promote some before the pilot):
- **Voice Type Classifier**: key child, other child, female adult, male adult. Already standard in
  child-language research.
- **Gaze target** (Gaze-LLE), for joint attention.
- **Text-prompted segmentation and tracking** (SAM 3), e.g. "infant", "adult". Licence to check.

---

### Phase 6: Corpus Overview

**Problem**: the viewer shows one video at a time. Researchers think in corpora: which videos
failed, where the pipelines disagree, what the whole dataset looks like.

**Solution**:
- [ ] Implement [`specs/010-corpus-analysis-foundations`](../../specs/010-corpus-analysis-foundations/spec.md)
      (dataset summary endpoint) on top of the tidy export.
- [ ] A corpus page in the viewer: one row per video, one small track per pipeline, sortable (by
      speaker/face disagreement, coverage, failures). A click opens the existing single-video
      timeline. This is the paper's cross-modal audit, at corpus scale.
- [ ] Previous/next between a batch's videos in the results viewer (item 4 of spec 008's viewer
      handoff).

---

### Phase 7: Docs and Tutorials

- [ ] **Tutorial**: the Peekaboo demo video from start to finish (install, run, review, export,
      analyse in R and Python).
- [ ] **Which pipeline for which question**, in the field's terms: caregiver speech, infant
      vocalisations, movement, faces, joint attention, touch.
- [ ] **A model card per pipeline**: the model, what it was trained on (few include infants), known
      failure modes, licence (Ultralytics is AGPL-3.0; pyannote models are gated), what to cite.
- [ ] **Data protection**: what runs where, what never leaves the machine, what changes with a
      hosted backend or a cloud agent.
- [ ] **Contributing a pipeline**: a guide and template, with `vlm_annotation` as the worked
      example. Contributions go into core by pull request; entry-point plugins wait for v1.7.0.
- [ ] **A 10-minute screencast**, reused for the webinar and conference tutorial.

---

### Launch

- [ ] Tag `v1.6.0`, GitHub release, Zenodo DOI, `CITATION.cff` updated.
- [ ] Update `paper/paper.md` and its research impact statement with the pilot's use, and tell the
      editors on #10182.
- [ ] Announce: mailing lists, webinar, conference tutorial.

---

## ❌ Out of Scope for v1.6.0

- Plugin ecosystem: entry-point discovery, `Dispatcher` ABC, `videoannotator-utils`
  ([`roadmap_v1.7.0.md`](roadmap_v1.7.0.md)).
- A visual redesign of the viewer: decided after the pilot.
- Remote and HPC dispatch ([`roadmap_v1.7_to_v2.0.md`](roadmap_v1.7_to_v2.0.md)).
- Pose, hand and motion specialist pipelines, unless pilot labs ask.

---

## ✅ Success Criteria

- [ ] A researcher outside the team installs it, works through the tutorial on their own machine,
      and gets the tidy export into R or Python without help.
- [ ] At least three labs outside the team have run it on their own videos (documented, with
      permission).
- [ ] At least one issue or pull request from outside the team has been resolved or merged.
- [ ] Changing viewer code cannot ship a stale bundle: CI fails on a mismatch.
- [ ] Every version constraint in `pyproject.toml` and the viewer's `package.json` has a recorded
      reason in the dependency audit, and the release supports Python 3.13 (3.14 as soon as
      TensorFlow does).
- [ ] Every default model has a published benchmark score, and none changed without one.
- [ ] Everything the viewer does is available from the CLI with `--json` and through MCP.
- [ ] The viewer has zero TypeScript errors and no placeholder pages.
- [ ] Nothing leaves the user's machine unless they switched on a hosted backend.

---

**Last Updated**: 2026-09-26
**Target Release**: Early 2027, ahead of BCCCD (7–9 Jan 2027)
**Status**: Planning Phase — Public Release
