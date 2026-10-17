# 🚀 VideoAnnotator v1.7.0 Development Roadmap

> **New 2026-10-17.** The public release moved here from v1.6.0 (Caspar): the viewer is redesigned
> first, so what is announced and piloted is what stays. v1.6.0 is now a quiet consolidation
> release ([`roadmap_v1.6.0.md`](roadmap_v1.6.0.md)). The plugin plan that was v1.7.0 is now
> [`roadmap_v1.8.0.md`](roadmap_v1.8.0.md). Phases moved from v1.6.0 keep their text and history.

## Release Overview

VideoAnnotator v1.7.0 is the **Public Release**: the first version we announce to the developmental
research community, with a tutorial, mailing-list posts and a webinar, and the first with the
redesigned viewer.

**Main Goal**: an outside lab can go from install to results in R or Python, on its own videos,
without help, and a researcher new to AI video analysis can ask questions of a video and get
systematic answers without help.
**Target Release**: when the pilot's fixes are in. No deadline.
**Current Status**: Planning. Viewer design done (spec 023); specs 026–029 specified 2026-10-16.
**Prerequisites**: v1.6.0 tagged.

---

## 🎯 Core Principles

v1.6.0's principles, plus:

- ✅ **Call things what they are** (spec 023 Decision 1): a video is a video, a folder a folder; the
  viewer doesn't show the API's words.
- ✅ **Nothing to set up before the first result.** Projects and workflows are offered after the
  fact, never required (spec 023 Decisions 2 and 5).
- ✅ **Pilot on what will be announced.**

---

## 📋 v1.7.0 Deliverables

### Phase 1: Viewer Overhaul

Design: [`specs/023-viewer-overhaul`](../../specs/023-viewer-overhaul/spec.md) (inventory, comparable
tools, decisions, journeys). Built by, in order:

- [ ] [`specs/027-ask-ai`](../../specs/027-ask-ai/spec.md): the VLM window. First answer in under a
      minute, prompt history never lost, sampling shown on the timeline, a live local/remote
      badge; the pipeline gains frame spacing, ranges and answers as they come.
- [ ] [`specs/028-video-page`](../../specs/028-video-page/spec.md): lanes as controls, one row per
      person or speaker, zoom, the data table beside the video, the run per lane, notes;
      streamed, with no folder question.
- [ ] [`specs/029-videos-and-projects`](../../specs/029-videos-and-projects/spec.md): one Videos page
      with folders and projects (from saved datasets), settings projects remember, workflows
      (from presets), the new nav, help in context. Takes over v1.6.0's Phase 6 (corpus
      overview); [`specs/010-corpus-analysis-foundations`](../../specs/010-corpus-analysis-foundations/spec.md)
      to be reconciled with it.
- [ ] Try the sketches on one or two researchers before 027's plan (spec 023 Process step 5).

---

### Phase 2: Connectors (was v1.6.0 Phase 5a)

- [ ] **OpenAI-compatible connector** alongside the Ollama one (`backends: [ollama,
      openai_compatible]`), moved here from the plugin plan (now v1.8.0). One client covers Ollama, llama.cpp, vLLM,
      LM Studio, SGLang and hosted APIs.
- [ ] **Structured outputs**: the prompt carries a JSON schema, so VLM labels come back parseable
      rather than as free text.
- [ ] **Video input** through vLLM (Qwen-VL models) as a third sampling mode beside single frame and
      burst. Ollama's Qwen3-VL has no video input as of September 2026.
- [ ] Hosted endpoints are off by default. Enabling one shows a plain warning that frames leave the
      machine; for infant video that is an ethics-approval question.
- [ ] The connector ships together with spec 027's local/remote badge and its confirmation before
      a remote send, never before them.
- [ ] Structured outputs fit Ask AI's lesson that the question defines the answer (journey 1): the
      window helps build the schema.

---

### Phase 3: Docs and Tutorials, Written Against the New Viewer

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
- [ ] Model cards and "which pipeline for which question" start from spec 026's pipeline guide
      (v1.6.0 Phase 3b), so the docs and the app say the same.

---

### Phase 4: Pilot (on the release candidates)

Outside testing runs on what will be announced (Caspar, 2026-10-17; before that it followed the
v1.6.0 tag, 2026-10-14). Spec 024's T083 (a pilot researcher on macOS) runs here too.

- [ ] 3–5 labs outside the team (conference contacts first) run it on about ten of their own
      videos, with a concrete ask: tell us where it broke.
- [ ] Their reports go in the public issue tracker. Label starter tasks `good first issue`.
- [ ] Pilot feedback sets the order of v1.6.x and v1.7 work.
- [ ] With permission, record who used it and for what.

---

---

### Phase 5: Agents (not a release gate)

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

---

### Phase 6: New Pipelines, Whichever the Pilot Labs Ask For

Candidates, none committed (the Phase 1 pipeline review may promote some before the pilot):
- **Voice Type Classifier**: key child, other child, female adult, male adult. Already standard in
  child-language research. VTC 2 depends on pyannote.audio and has no stated licence (audit §8).
- **Diarization on Sortformer** (moved from v1.6.0 Phase 5c, 2026-10-17, spec 030 Q2): first a
  half-day spike. Does NVIDIA Streaming Sortformer install and run on a torch newer than 2.11
  without torchaudio? What does NeMo send over the network by default, and can it be switched off?
  Install size? Output on the demo clip next to pyannote community-1's? Findings and the decision
  go into the dependency audit §8 (spec 030 FR-028). pyannote is what holds torch at 2.11.
- **Gaze target** (Gaze-LLE), for joint attention.
- **Text-prompted segmentation and tracking** (SAM 3), e.g. "infant", "adult". Licence to check.

---

---

### Launch

- [ ] Tag `v1.7.0`, GitHub release, Zenodo DOI, `CITATION.cff` updated.
- [ ] Update `paper/paper.md` and its research impact statement with the pilot's use, and tell the
      editors on #10182.
- [ ] Announce: mailing lists, webinar, conference tutorial.

---

---

## ❌ Out of Scope for v1.7.0

- Plugin ecosystem ([`roadmap_v1.8.0.md`](roadmap_v1.8.0.md)).
- Remote and HPC dispatch ([`roadmap_v1.8_to_v2.0.md`](roadmap_v1.8_to_v2.0.md)).
- A benchmark of our own (see v1.6.0, Phase 5c).
- Human review or correction of pipeline output (shelved, spec 023).
- A movement measure per person (v1.8.0 Phase 3, server-side post-processing).

---

## ✅ Success Criteria

- [ ] Every default model has a model card citing its upstream evaluation.
- [ ] The viewer shows none of the words job, batch, dataset or preset (spec 029 SC-002).
- [ ] A first-time user gets a first Ask AI answer on their own video in under a minute (spec 027
      SC-001).
- [ ] Nothing leaves the user's machine unless they switched on a hosted backend, and the viewer
      says where the model runs.
- [ ] Everything the viewer does is available from the CLI with `--json` (MCP if Phase 5 lands).

**The pilot's measures** (before launch):

- [ ] A researcher outside the team installs it, works through the tutorial on their own machine,
      and gets the tidy export into R or Python without help.
- [ ] At least three labs outside the team have run it on their own videos (documented, with
      permission).
- [ ] At least one issue or pull request from outside the team has been resolved or merged.

---

**Last Updated**: 2026-10-17
**Target Release**: when the pilot's fixes are in
**Status**: Planning Phase — Public Release
