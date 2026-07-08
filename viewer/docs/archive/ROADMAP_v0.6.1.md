# Video Annotation Viewer v0.6.1 Roadmap (JOSS Submission Release)

**Theme:** JOSS Paper + Release Readiness (No New Features)

**Status:** 📋 PLANNED

**Target Window:** As soon as VideoAnnotator **v1.4.2** is tagged/released (Dec 2025–Jan 2026)

**Previous Version:** v0.6.0 (Job Results Viewer + Artifacts ZIP support — 2025-12-15)

---

## 🎯 Overview

v0.6.1 is a **documentation + release process** release to support a **JOSS submission** for Video Annotation Viewer (vav), coordinated with the VideoAnnotator server **v1.4.2** release.

This roadmap is a **full implementation plan**: it lists concrete deliverables, owners, acceptance criteria, and the release steps required to submit to JOSS.

### Scope constraints (non-negotiable)

- **No new user-facing features** unless they are required to make existing claims verifiable for reviewers.
- **CI improvements only**: improve stability/“scores” of existing CI checks; do not add new mandatory quality gates.
- **Server coordination first**: we will not cut v0.6.1 until the server’s v1.4.2 tag/contract is available.

---

## ✅ Current State (Dec 19, 2025)

- App is already at **v0.6.0** (see `package.json` + `CHANGELOG.md`).
- Lint/typecheck are currently clean (`npm run lint:strict`, `npx tsc --noEmit`).
- JOSS draft exists at `docs/joss.md`, but **JOSS submission requires `paper.md` + bibliography** and the draft currently:
  - references `paper.bib` that does not exist
  - contains a stray `:contentReference[...]` token that likely breaks compilation

---

## 📦 Workstreams

### 1) JOSS paper & submission artifacts 📄 (Owner: Docs)

**Goal:** Have a paper that compiles in the Open Journals toolchain and a repository that satisfies the JOSS reviewer checklist.

#### 1.1 Paper files (blocking)
- [x] Create `paper/` folder
- [x] Move/rename `docs/joss.md` → `paper/paper.md`
- [x] Create `paper/paper.bib`
- [x] Remove compilation-breaking tokens (e.g., `:contentReference[...]`)
- [x] Replace placeholder references with real citations and in-text citekeys (e.g., `[@key]`)
- [x] Validate paper length stays within JOSS guidance (250–1000 words)
- [x] Document and run the local paper compilation step (see `docs/JOSS_BUILD.md`)

**Acceptance criteria:** `paper/paper.md` + `paper/paper.bib` compile successfully via the local JOSS build procedure in `docs/JOSS_BUILD.md`.

#### 1.2 Repo metadata (blocking)
- [x] Update `CITATION.cff` (version, release date, title, author ORCIDs if available)
- [x] Ensure `LICENSE` is present and OSI-approved (already present; verify no contradictions)
- [x] Ensure `README.md` reflects current version and reviewer-facing quickstart (see Workstream 2)

**Acceptance criteria:** JOSS reviewer can find how to cite, how to install/run, and what the software does within 5 minutes.

#### 1.3 JOSS submission checklist mapping
Add a short checklist mapping in this repo (either in this roadmap or a dedicated `docs/JOSS_CHECKLIST.md`) covering:
- [x] Paper present (`paper/paper.md`, `paper/paper.bib`)
- [x] License present (`LICENSE`)
- [x] Citation metadata present (`CITATION.cff`)
- [x] Installation instructions present (README)
- [x] Tests runnable locally (README + `docs/TESTING_GUIDE.md`)
- [ ] Archive DOI minted for the tagged release (Zenodo)

---

### 2) Documentation: “Reviewer journey” 📚 (Owner: Docs + Client)

**Goal:** A reviewer can clone, run, and verify core claims quickly.

- [x] Update `README.md` header/version references (currently stale)
- [x] Add a short **Local Run** section:
  - install deps
  - run dev server
  - open the app
- [x] Add a short **Connect to VideoAnnotator** section:
  - required env vars (`VITE_API_BASE_URL`, `VITE_API_TOKEN`) and localStorage fallbacks
  - expected default server URL
- [x] Update `docs/CLIENT_SERVER_COLLABORATION_GUIDE.md` to the v1.4.2 contract (endpoints, auth, artifacts)
- [x] Update `docs/TESTING_GUIDE.md` to current scripts and CI expectations

**Acceptance criteria:** Following docs, a reviewer can (a) run the UI locally, and (b) either load demo artifacts or connect to a running v1.4.2 server.

---

### 3) CI health (“scores”) ✅ (Owner: Client)

**Goal:** Improve reliability of existing CI checks without changing which checks are required.

Current CI behavior (reference):
- Required: lint + unit tests
- Optional (non-blocking): coverage, e2e, lighthouse

Planned improvements:
- [ ] Reduce flakiness in existing Playwright smoke tests (no new tests required)
- [ ] Stabilize Lighthouse CI runs (reduce transient failures)
- [ ] Document a simple “CI score” metric (e.g., % green over last N runs)

**Acceptance criteria (CI score):** Over the **last 10 runs on `main`**, optional jobs (`e2e`, `lighthouse`, `coverage`) are green in **≥80%** of runs, with failures triaged and documented.

---

### 4) Server-team coordination (VideoAnnotator v1.4.2) 🔁 (Owner: Server + Client)

This section is intended to be shareable with the VideoAnnotator server team.

**Blocking asks for the server team:**
- [ ] Publish/tag **VideoAnnotator v1.4.2** (or confirm final tag name/date)
- [ ] Provide a versioned **OpenAPI spec** for that tag (or confirm where to fetch it)
- [ ] Confirm artifacts/results bundle expectations:
  - ZIP includes source video
  - stable directory structure and filenames (or document variability)
  - stable metadata keys for pipeline ID, parameters, and versions
- [ ] Confirm auth + CORS expectations for local reviewer runs (token format, headers, allowed origins)

**Client-side follow-ups (once server tag exists):**
- [ ] Validate the viewer against v1.4.2 (manual smoke + documented steps)
- [ ] Update any pinned OpenAPI snapshots and compatibility notes

**Acceptance criteria:** We can state a clear compatibility line in release notes: “v0.6.1 supports VideoAnnotator v1.4.2”.

---

## 📋 Development Phases

### Phase 1: Paper packaging + server contract (Week 1)
- [x] Create `paper/` structure and bibliography
- [x] Remove paper compilation blockers
- [x] Draft server coordination asks and confirm v1.4.2 tag timeline

**Milestone:** Paper compiles locally; server tag plan confirmed.

### Phase 2: Documentation refresh + CI stabilization (Weeks 2–3)
- [x] Refresh README + key docs for reviewer journey
- [ ] Improve CI score for optional jobs (stability only)

**Milestone:** Docs are reviewer-ready; CI optional jobs mostly green.

### Phase 3: Release + archive + submit (Week 4)
- [x] Bump version to v0.6.1 + update changelog
- [ ] Tag release + GitHub release notes
- [ ] Archive tagged release via Zenodo and obtain archive DOI
- [ ] Submit to JOSS and start review

**Milestone:** v0.6.1 is tagged and archived; JOSS submission created.

---

## 🎯 Success Criteria

### Must-have (blocking v0.6.1 release)
- [ ] `paper/paper.md` + `paper/paper.bib` compile successfully
- [ ] `README.md` provides a reviewer-friendly run path
- [ ] `CITATION.cff` is accurate for v0.6.1
- [ ] VideoAnnotator server **v1.4.2** is tagged/released and compatibility statement is documented

### Must-have (blocking JOSS submission)
- [ ] v0.6.1 is tagged and archived (Zenodo DOI)
- [ ] Paper references the archived DOI (once available)

### CI score goals (non-blocking but tracked)
- [ ] Optional CI jobs are green ≥80% over last 10 runs on `main`

---

## 🚫 Out of scope

- New viewer features
- Major UI redesigns
- New mandatory test gates
- Major performance/architecture work
- Full accessibility compliance (track separately)

---

## 🔧 Release checklist (operator steps)

- [ ] Update version to `0.6.1` and add release notes to `CHANGELOG.md`
- [ ] Run locally: `npm run lint:strict`, `npx tsc --noEmit`, `npm run test:run`
- [ ] Build: `npm run build`
- [ ] Tag `v0.6.1` and create GitHub release
- [ ] Confirm Zenodo archive DOI minted from the `v0.6.1` tag
- [ ] Submit to JOSS (repository URL, tagged release URL, archive DOI, paper)

---

**Document Version:** 1.0

**Created:** 2025-12-19

**Status:** 📋 Planning — JOSS submission prep for v0.6.1
