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
**Current Status**: Phase 0 done; Phase 1 nearly done (open items below); Phase 2 next (2026-10-02)
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
- [x] Import `video-annotation-viewer` into `viewer/` with its full history (rewrite paths with
      `git filter-repo --to-subdirectory-filter viewer`, then merge with
      `--allow-unrelated-histories`), taken from `feature/vlm-annotation-support`.
      Done 2026-09-30 from the viewer's `main` at its v0.7.0 release, which includes that branch;
      its tags are kept as `viewer-v0.x.x`.
- [x] One build step produces `viewer_static/` from `viewer/`. CI fails when the committed bundle
      doesn't match the source, so the copy can't be forgotten.
      `scripts/build_viewer.sh` (`--check` in CI's `viewer` job, which also runs the viewer's lint
      and unit tests; releases wait for it). The viewer's Playwright and Lighthouse jobs, both
      informational, aren't in CI yet.
- [x] One dev command runs the API server and the Vite dev server (API proxied, hot reload), plus
      VS Code tasks for it.
- [x] The viewer takes VideoAnnotator's version number from v1.6.0: one product, one version, one
      changelog. Standalone mode (drop in files, no server) stays, and is still published.
- [x] The TypeScript half of the viewer contract test, deferred in v1.5.0, runs in CI.
      `viewer/src/test/contract/videoannotator-outputs.test.ts` loads real job outputs
      (`tests/fixtures/viewer_contract/`) the way the artifacts zip is loaded. It found that
      `*_openface3_analysis.json` was detected as person tracking and, sorting first in the zip,
      replaced the real person tracks with OpenFace's face boxes, while `*_openface3_detailed.json`
      was detected as face analysis. Cause: every detector `JSON.parse`d a 5–15 KB prefix, which
      throws on any larger file, so a looser substring check claimed it. Detection now classifies
      the whole parsed file by its annotation fields first.
- [x] One JS package manager (the viewer has both `bun.lock` and `package-lock.json`). Bun; `package-lock.json` removed.
- [x] Merge the two `AGENTS.md`/`CLAUDE.md` files. The viewer's guidance is `AGENTS.md` §23
      (corrected for Bun, `127.0.0.1`, the shared version and the contract fixtures); the viewer's
      `AGENTS.md`, `CLAUDE.md`, Copilot instructions and Claude settings are removed.
      The viewer's constitution is folded into `.specify/memory/constitution.md` (v1.1.0,
      2026-10-02: new Principle VI, Faithful Annotation Display; viewer clauses in I, II, IV, V and
      Engineering Standards), and the viewer's own spec-kit setup is removed. Making "no
      telemetry" product-wide found Ultralytics sending a Google Analytics event per predict; off.
- [x] Move the viewer's open issues, archive the old repo with a pointer README, and update links
      in `README.md`, `paper/paper.md` and `CITATION.cff`.
      Links done (2026-10-01): `README.md`, `docs/usage/GETTING_STARTED.md`, and the viewer's README,
      docs, `CONTRIBUTING.md` and `package.json` point here; `viewer/CITATION.cff` (a separate
      v0.7.0 that no longer exists) removed. `paper/paper.md` and `CITATION.cff` had none. Archives,
      the frozen viewer changelog and the sent cover letter keep the old links. The old repo has
      no open issues, so nothing to move. Pointer README pushed and the repo archived 2026-10-02.

**Viewer bugs found in the v1.5.0 end-to-end run** (2026-09-26), to fix once `viewer/` is in
this repo:
- [x] **VLM prompt sent as `base_url`**: submitting a job with `vlm_annotation` stored the prompt
      text in both `prompt` and `base_url`, so the job failed with "Port could not be cast to
      integer value as ' 2, child: True'". Likely the wizard's config form binding the prompt
      textarea's value to the `base_url` field too. Since `610aea1` the server rejects a non-URL
      `base_url` at submission (400 `INVALID_URL`), so it now fails fast, but the form is still
      wrong.
      Fixed early in the viewer (`4ec8d0b`): the stored job's `base_url` is the prompt with its
      newlines stripped, i.e. pasted into the single-line Base Url box, not a binding bug. URL
      fields now show an inline error and block Next/Submit.
- [x] **Misleading error on a rejected job**: when `POST /api/v1/jobs` returns 400 with a clear
      message (e.g. "Unknown pipeline 'speaker_diarization'"), the viewer shows "All job
      submissions failed" with a generic tip that the server may not be running or the token may be
      invalid. Show the server's `message`/`hint` instead; keep connection tips for actual
      connection failures.
      Fixed early in the viewer (`4ec8d0b`).
- [x] **No library folder selected → flickering dialog on "View jobs"**: the dialog flickers and
      never explains what a library folder is or how to choose one. Show a steady prompt with a
      "Choose folder" action (and why it's needed), or let job viewing work without one.
      Fixed: cause was a loop. The results page starts a download whenever its state is `idle`; a
      folder picker that was cancelled, or refused because no click preceded it, set `idle`
      again, so it restarted at once. Now it waits in `needs_folder`: a card says what the
      library folder is for, with "Choose folder" (a real click, so the picker is allowed) and
      "View without saving". A folder already granted is reused without asking.
      Found alongside: the exported `apiClient` proxy didn't bind methods, so
      `apiClient.updateConfig()` changed nothing and Settings' "Test Connection" always tested
      the saved configuration, not the one on screen.
- [x] **Scene detection always shows "(No data)"**: `parseSceneDetection`
      (`src/lib/parsers/scene.ts`) accepts a bare array, `results` or `scenes`, but not COCO's
      `annotations`, which is what the backend writes. It throws, and the scenes panel is empty
      for every job, however many scenes there are. Found 2026-09-28 on a clip with one scene
      (0–7.32 s, "nursery") the backend had detected correctly.
      Fixed early in the viewer (`4ec8d0b`).
- [x] **Failed pipelines' reasons are hard to find**: a batch says "N with errors … the reason is
      on each video's row", but the reason is only a hover tooltip on the status badge, and the
      job page doesn't list failed pipelines with their `error_message` from
      `GET /jobs/{id}/results`. Show per-pipeline errors on the job page and inline in the batch
      row. (Server side: since 2026-09-28 the job-level `error_message` includes each failed
      pipeline's error, not only its name.)
      Fixed: each batch/jobs row shows the reasons under its status badge (two lines, full text
      on hover); a job that failed outright lists each pipeline's error on its page, as a job
      completed with errors already did.
- [x] **`localhost` vs `127.0.0.1`: viewer can't connect, or connects without its token**
      (recurring). The server side (`start_server.sh`, the `setup-db`/`generate-token` login
      links, `CORS_AND_AUTH_PROTOCOL.md`) says `localhost`. The viewer defaults its API URL to
      `127.0.0.1` and rewrites a saved `localhost` to `127.0.0.1`. The browser treats them as two
      sites with separate storage, so after logging in via a `localhost` link the viewer either
      can't reach the API ("Cannot Connect", seen 2026-09-28 in the devcontainer on Windows) or
      reaches it with no token (401 on every call). **Server side fixed in v1.5.0** (2026-09-28):
      the server redirects `localhost/viewer…` and `/viewer-connect` to `127.0.0.1`, and every
      link it prints uses `127.0.0.1`. **Left for the viewer**: when the server serves it, use the
      page's own origin as the API URL, with no rewriting, and drop the redirect.
      Fixed (`viewer/src/lib/apiConnection.ts`): an empty API URL, or one naming the page's own
      origin, stays relative, so the served viewer always calls its own server; Settings no longer
      pre-fills `http://127.0.0.1:18011` there (saving that made the call cross-origin). Another
      server at `localhost` is still sent to `127.0.0.1` (the server binds IPv4). The redirect
      stays: the printed links save the key under `127.0.0.1`, and the redirect is what makes a
      viewer opened at `localhost` find it. Also fixed: "Test Connection" with an empty key tested
      the previously saved key, because the client treated `''` as "keep".
- [x] **Settings page doesn't say how to get a token** (2026-09-30): the welcome box says "Get your
      API token from the server console or administrator", and its default URL is
      `http://localhost:18011` (see the item above). A new user doesn't know which console, what
      the token looks like, or what to do if they missed it. Replace the Quick Start with:
      1. **First start**: the server prints `[API KEY] VIDEOANNOTATOR API KEY GENERATED`, then a
         `va_…` key and a one-click `http://127.0.0.1:18011/viewer-connect?token=…` link. Opening
         the link logs the viewer in; there's no need to paste anything.
      2. **Missed it, or need another key**: in a terminal on the server machine, run
         `videoannotator generate-token` (in a `uv` checkout, `uv run videoannotator
         generate-token`). It prints a new key and a one-click link. Keys can't be shown again
         once printed.
      3. **Someone else runs the server**: ask them to run step 2 for you with `--user <your
         email>`.
      4. **Auth off** (`AUTH_REQUIRED=false`): leave the token blank.
      Include a copy button for the command, and show the key format (`va_` + 43 characters) as
      placeholder text so a pasted wrong value (for example the whole `Bearer …` line) is caught
      before "Test Connection". Server side (done 2026-09-30): the first-run banner now prints the
      viewer-connect link and `videoannotator generate-token`. Before this it pointed at
      `localhost:8000` and a `scripts.manage_tokens` module that pip installs don't have.
      Done: `TokenHelp` (the four steps, copy button) on the Settings page and its Help tab, the
      key format as placeholder, and an inline check that blocks saving a pasted `Bearer …`, a
      truncated key or `dev-token`. Every `dev-token` instruction is gone; the client rejects it.
- [x] **Face boxes and OpenFace landmarks almost never show** (2026-09-30): both overlays draw a
      face only within ±0.1 s of its timestamp (`JD(face_analysis, t, 0.1)` and the
      `openface3_faces` filter). The face pipelines sample about once a second (0.97 s apart on a
      30 fps clip), so a face flashes for 0.2 s per second during playback and never shows when
      paused between samples. The data is in the job's `*_face_detections.json` and
      `*_openface3_analysis.json`, and it parses. Hold each face until the next sample of its
      track, capped at about 1.5× the sample interval, which is taken from the data itself rather
      than a constant, so a face that disappears doesn't stay on screen. Server side (fixed
      2026-09-30): DeepFace's "no face found" stand-in, a full-frame box with confidence 0 and
      an invented emotion, was saved as a real face (143 of 255 boxes in the e2e jobs). Jobs run
      before the fix still contain those boxes.
      Fixed: `viewer/src/lib/sampledAtTime.ts` draws the latest sampled frame's detections until
      the next sample, for at most 1.5× the median sample interval. Pose uses it too: its ±0.5 s
      window stacked several frames' skeletons when sampling was dense.
- [x] **`openface3_detailed.json` detected as face analysis**: the face check (`"emotions"` in the
      first 8 KB) matches its `metadata.model_info.features`, and it runs before the OpenFace
      check. The file parses to nothing, so today the face boxes survive only because
      `face_detections.json` sorts first in the zip. Check for OpenFace before face analysis, or
      match on structure (`metadata.pipeline` + `faces`) instead of substrings.
      Fixed (`47cc786`), found by the contract test along with a worse one: `openface3_analysis`
      was taken as person tracking and replaced it (see the contract test item above).
- [x] **Results view isn't batch-aware**: opening a video from a batch loses the batch. Show which
      batch and video (n of N) you're on, previous/next between the batch's videos, and a way back
      to the batch page. Overlaps Phase 6's previous/next item; do the navigation here, before
      release.
      Done: the results header shows the batch's name and "n of N" (a link to the batch page),
      with previous/next to the nearest videos that have results, from any entry point (the job
      carries its `batch_id`); Back goes to the batch.
- [x] **First-run download total double-counts shared weights**: the "Preparing… downloads about
      1.4 GB" line sums each pipeline's `weights_not_cached` notes, so a model two pipelines share
      (`pyannote/speaker-diarization-3.1`, for audio_processing and speaker_diarization) counts
      twice. Deduplicate by the note's `name` before summing.
      Fixed early in the viewer (`4ec8d0b`).

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
- [x] **Pipeline review** (`docs/development/pipeline_review_v1.6.0.md`). Start from what the
      field asks, not from what we have: caregiver speech, infant vocalisations, who is speaking,
      faces and expressions, movement, gaze and joint attention, touch. For each question, which
      tool answers it best today. For each existing pipeline: which question it answers, whether
      its upstream is maintained, download and install cost, licence, how it does on infants and
      young children, and overlap with other pipelines. Decide for each: keep, replace, mark
      experimental, or drop (a dropped pipeline can come back as a v1.7.0 plugin). The review sets
      the scope of the audit and of Phase 5's model work, and gives Phase 7 its "which pipeline
      for which question" page. With no users yet, dropping a pipeline breaks no one.
- [x] **Dependency audit** (`docs/development/dependency_audit_v1.6.0.md`), for the pipelines the
      review keeps. For every direct dependency of the core package, each extras group and the
      viewer (merged in Phase 0): current constraint,
      latest release, why it is pinned (git history, CHANGELOG), what upgrading changes (API,
      outputs, install: wheels, CUDA, system libraries), and a decision: upgrade, keep with a
      stated reason, replace, or drop. Also covers the Python version, the CUDA index and driver
      floor, Docker base images, Node and the JS package manager, pre-commit hooks and GitHub
      Actions versions.
- [x] **Specs**: one spec-kit spec per coherent change the audit calls for (`/speckit-specify`).
      Expected, subject to the review and audit: removing dropped pipelines; Python 3.13 (with a
      3.14 CI job, made required once TensorFlow ships for it, or once deepface is dropped) and the
      core dependencies; the torch stack and CUDA index; the pyannote.audio 4 migration; the
      viewer's dependencies.
      (Done 2026-10-01: specs 012 Python 3.13, 013 core dependencies and tooling, 014 drop the
      LAION pipelines, 015 torch 2.11 + pyannote.audio 4, 016 one models directory.)
- [x] **Implement** the specs before rc1. Re-baseline the v1.4.x acceptance fixtures once, on
      purpose, and record the before/after differences on the demo video in the CHANGELOG.
      (Specs 012–016 implemented and passing CI on `1.6-dev`. Re-baselined 2026-10-02 against
      v1.5.0, not v1.4.4: the v1.4.4 fixtures were never captured, and v1.5.0 is the release these
      upgrades follow. Outputs unchanged up to GPU noise (CHANGELOG). The baseline is the viewer
      contract fixtures, checked by `tests/integration/test_output_baseline.py`, real models.)
- [ ] **Docker image size**: build `Dockerfile.cpu`/`Dockerfile.gpu` slim and with
      `--build-arg EXTRAS=all`, and record both sizes against the v1.4.3 baseline. v1.5.0 set a
      target of 80% smaller but couldn't measure it (no Docker where it was checked).
      Needs Docker on the host (2026-10-02: build commands given to Caspar). Also open: the GPU
      images' base, `nvidia/cuda:12.6.0-runtime-ubuntu24.04`, was built 2024-08-12 and never
      rebuilt (two years of Ubuntu security fixes missing). torch's wheels carry their own CUDA
      12.6 and cuDNN, so the base could be plain `ubuntu:24.04` like the CPU image (~1.3 GB smaller
      compressed); the one dependency on it is TensorFlow (DeepFace) loading `cusolver`,
      `cusparse` and `nvJitLink` from the base, which the venv also has. Verify TensorFlow still
      reaches the GPU before switching; fallback is `12.6.3` plus `apt-get upgrade`. Ubuntu 26.04
      only has CUDA 13 images. The Dockerfile headers still say cu124 and port 8000.
- [x] **Windows froze with the dev container running** (2026-10-02; handover in
      `docs/development/handover_windows_devcontainer_freeze.md`). Cause: memory exhaustion. The
      host runs at ~80% of its 31 GB with everyday apps, and the WSL VM (uncapped, up to 15.6 GB)
      is the one thing that can grow by ~11 GB: page cache and process memory count as VM memory
      until the VM hands them back. A resume from hibernate added the last burst; the machine sat
      at ~99% memory until a power-button reset. No test run was involved.
      Done (merged 2026-10-02): `.venv`, models and `viewer/node_modules` in named volumes; a
      start-up warning for a Windows-drive workspace (`scripts/check_workspace_mount.sh`); the
      container capped at `--memory=12g` in `devcontainer.json`. Measured under that cap in a
      volume clone (handover, Update 6): installing the extras alone fills the cap with page
      cache (reclaimed, no OOM kills); the full default `pytest` peaks at 5.4 GiB and takes 96 s;
      a six-pipeline job on the demo video peaks at 6.45 GiB; test collection is 10–25× faster than
      over the bind mount.
- [ ] Follow-ups from the freeze, not blocking:
  - Size the container cap from one long-video job (likely 8g or 10g; 12g until then).
  - Check whether CUDA's "Shared GPU memory" sits outside both caps (Task Manager during a job).
  - Docs (Phase 2): the volume clone, the `uv sync --inexact --extra all` step a fresh clone needs
    (~5 min), the 16 GB-machine guidance, and the OOM symptom (`Killed`, exit 137) with the cap
    in `devcontainer.json`.

**Also in this phase: one place for model weights.** Today the weights end up wherever each
library puts them by default. Whisper, YOLO and the LAION pipelines use `./models/<name>`
(relative to the server's working directory, so a different directory means another download).
Hugging Face uses `~/.cache/huggingface`, pyannote 3.x uses `~/.cache/torch/pyannote`, and
DeepFace uses `~/.deepface`. Users can't find them, and in a container they're lost on every
rebuild. The devcontainer points all of them into `models/` with environment variables (2026-09-28),
but only the devcontainer does.
- [x] One setting, `VIDEOANNOTATOR_MODELS_DIR` (default decided in the spec: `./models` or a
      per-user data directory), resolved to an absolute path once at startup. Under it, one
      subdirectory per source: `huggingface/`, `pyannote/`, `torch/`, `deepface/`, `whisper/`,
      `yolo/`, and so on.
- [x] The server and CLI set `HF_HUB_CACHE` (not `HF_HOME`, which holds the login token), `TORCH_HOME`, `PYANNOTE_CACHE` and `DEEPFACE_HOME` from it
      before any pipeline library is imported, unless the user has set them. Pipelines' own
      `cache_dir` defaults come from the same place.
- [x] Readiness (`api/readiness.py`) finds weights through the same resolver, so "downloads about
      N MB" is never wrong about where it looked.
- [x] Moving the default means existing installs download once more. Say so in the CHANGELOG and
      print the old and new locations the first time the server starts. Drop the devcontainer's own
      environment variables once the app sets them.
- [x] `videoannotator diagnose` shows where the models directory is and how much it holds.
(Done in spec 016, 2026-10-01: default is the per-user data directory.)
- [x] Logs go to `./logs` relative to the working directory, like the models used to: put them
      under a per-user directory too (found during spec 016). (Done 2026-10-02:
      `VIDEOANNOTATOR_LOG_DIR`, per-user default; `/app/logs` in Docker, `<repo>/logs` in the dev
      container.)
- [x] **The database lives in `./videoannotator.db` too**, under wherever the server starts, so
      starting it from another directory shows an empty job list (found 2026-10-02). There are two
      database layers: the storage backend (`api/database.py`, honours `VIDEOANNOTATOR_DB_PATH`)
      and the SQLAlchemy one (`database/database.py`: users, tokens, jobs routes, datasets,
      presets), which reads only `DATABASE_URL`. Set `VIDEOANNOTATOR_DB_PATH` alone and the second
      still opens `./videoannotator.db`. (Done 2026-10-02, no spec: with no users yet, an
      existing database isn't migrated, only mentioned in the CHANGELOG. One resolver,
      `database_location.py`, for both layers; per-user default; `/app/database` plus a named
      volume in Docker; `<repo>/videoannotator.db` in the dev container.)
- [ ] **Download weights ahead of time** (spec 011's FR-017, deferred there): an action per
      pipeline, in the viewer, the API and the CLI, that downloads its declared weights into the
      models directory before the first job, reusing the extras install job's table and statuses.
      Needs a download step per model library (Hugging Face, torch hub, Ultralytics, DeepFace,
      Whisper), which this phase's one models directory makes practical.

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
- [ ] Docs have one entry page (`docs/README.md`) and a link check in CI, so moving files can't
      leave dead links. (Unfinished since spec 001: T055, T056, T059.)
- [ ] No placeholder pages. The Datasets page says "Coming Soon" behind a disabled button, although
      spec 007's backend shipped in v1.5.0: wire it up or hide it.
      (Preset load/save in the job wizard landed in v1.5.0; the Datasets page, saved datasets and
      import/export from spec 007's viewer handoff are what's left.)
- [ ] Viewer: zero `tsc --noEmit` errors (24 on 2026-09-26), with typechecking in CI.
      Constitution 1.1.0 makes this an Engineering Standard, as are a 300 KB gzipped initial bundle
      (304 KB on 2026-10-02) and overlays naming the pipeline and version that drew them
      (Principle VI; not shown yet). All three are open follow-ups in its Sync Impact Report.
      Note (2026-10-01): plain `bunx tsc --noEmit`, the documented check, compiles nothing (the root
      `tsconfig.json` has `"files": []` and only references); run `-p tsconfig.app.json`. Still 24
      errors in 13 files, e.g. the OpenAPI `paths` type in `src/api/client.ts` lacks `/api/v1/jobs`.
- [ ] Viewer: one function decides which pipeline produced a file. Today there are four
      (`merger.ts`, `fileUtils.ts`, and two arrays in `FileUploader.tsx`) and they disagree.
      Start from `merger.ts`'s `detectJSONStructure` (2026-10-01), which classifies a parsed file by
      its fields and is covered by the contract test against real outputs.
- [ ] A structural pass along the one path users take: install → add videos → run → review →
      export. Anything off that path moves or goes. A Playwright "first-time user" run on a clean
      machine files each point of friction as an issue.
- [ ] **OpenFace 3 results are sensitive to GPU nondeterminism** (found 2026-10-01, spec 013): a
      1-pixel difference in RetinaFace's box between two runs of the same video changes that
      face's gaze and action-unit intensities by up to ~2.4. Offer a deterministic mode
      (`torch.use_deterministic_algorithms`, cuDNN deterministic) and record it in provenance, and
      measure run-to-run spread for every pipeline in the Phase 5 benchmark.
- [ ] **A pipeline that fails must say so.** `speech_recognition` catches errors in transcription,
      logs them only to the pipeline log, and returns no annotations, so the job reports the
      pipeline as completed with an empty transcript (found 2026-10-01 through a Triton cache
      error). Raise instead, so the job shows the pipeline as failed with its error; check the
      other pipelines for the same pattern.
- [ ] **`GET /api/v1/jobs/{id}/results/files/{pipeline}` returns `OUTPUT_FILE_MISSING`** for
      every result stored in the database (`output_file: database:/annotations/...`), which is
      what each job's results list advertises as its `download_url`. Same in v1.5.0 (found
      2026-10-02). Serve the file from the job folder, or stop advertising the URL.
- [ ] `videoannotator process <video>` is listed in `--help` but only prints "Direct processing
      is not yet implemented" (found 2026-10-01). Implement it on the shared job-execution path,
      or remove it.
- [ ] **Queue position**: a pending job shows how many jobs are ahead of it ("3rd in queue"), in
      the job list, the job page and `GET /api/v1/jobs/{id}`. Today a queued job looks the same as
      a stuck one. (Planned since spec 001's T066.)

**Run it again**: the path doesn't end at review. A researcher who likes a result wants the same
settings on more videos; one who doesn't wants to tweak and rerun. Today both mean rebuilding the
job in the wizard from memory, although every job already stores its `selected_pipelines` and
`config`.
- [ ] **Rerun a job**: same videos, same settings, one click, as a new job that links back to the
      original (`rerun_of`). "Edit and rerun" opens the wizard prefilled.
- [ ] **Reuse settings on new videos**: "Use these settings" on a job or batch opens the wizard at
      "Choose videos" with pipelines and config filled in; "Save as preset" writes the existing
      `saved_pipeline_presets` table.
- [ ] **UX cues that lead there**, not only buttons in a menu: the actions sit on the job/batch
      result page where the user is looking when they decide; the wizard's first step offers
      "Start from a previous job" and recent presets before the blank form; a failed or partial job
      says "Fix settings and rerun"; a completed batch suggests "Run on more videos". Check each
      in the Playwright first-time-user run.
- [ ] **Prompt library**: every VLM prompt used, in a job or a preview, saved to the database once
      (deduplicated by SHA-256, the same hash as the methods paragraph's provenance) with its
      model, first/last used and the jobs that used it. A browser to search, view, diff, name,
      star and reuse them; reusing one fills the prompt field.
- [ ] **Prompt workbench**: the "test prompt" panel from inside the VLM pipeline config as a
      standalone page, since prompt design is iterative and deserves more room than a wizard step.
      Pick a video and frame (or burst), a model and a prompt; run; compare responses side by side
      across prompt versions or models; send the winner to a job or preset. Backend exists
      (`POST /api/v1/vlm/preview`, `GET /api/v1/vlm/models`); new work is the page, the prompt
      table and its endpoints. Also through the CLI/MCP (Phase 4).
- [ ] **Compare two VLM jobs** on the same video: their labels on one timeline, with the frames
      where they disagree listed, and ELAN ground truth as a third row when there is one. The
      workbench compares prompts on single frames; this compares whole runs. Asked for in spec
      009's viewer handoff; comparing across a whole dataset stays in Phase 6.

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

**Candidates**: [`pipeline_landscape_v1.6.0.md`](pipeline_landscape_v1.6.0.md) (2026-10-01) lists,
per kind of processing, what we have, what's obsolete, the current best tools with licences, and
what developmental research uses that we don't do at all. Top of its list: voice type
classification (VTC 2) with adult word counts (ALICE) and conversational turns; infant looking
(iCatcher+); motion energy and dyadic synchrony; adult/child role per person; de-identified
export; caregiver prosody.

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
      recording. The pyannote.audio 4 library upgrade itself happens in Phase 1. Score NVIDIA
      Streaming Sortformer alongside, with dependency health as a criterion: pyannote is what holds
      torch at 2.11 (via the discontinued torchaudio) and upstream shows no plan to drop it.
      Background and decision: `dependency_audit_v1.6.0.md` §8.
- [ ] Before that benchmark: a half-day Sortformer spike (installs on torch 2.14 without
      torchaudio? NeMo's telemetry? install size? demo-clip output). Audit §8.
- [ ] Caspar: ask LAAC-LSCP about VTC 2's licence (the repository has none). VTC 2 also depends on
      pyannote, so adopting it keeps the torch cap. Audit §8.
- [ ] **Person**: `yolo11n-pose` → `yolo26n-pose`.
- [ ] **OpenFace 3**: confirm `openface-test` is the maintained distribution (the Phase 1
      audit keeps it at `==0.1.13`).

#### 5d. New pipelines: whichever the pilot labs ask for

Candidates, none committed (the Phase 1 pipeline review may promote some before the pilot):
- **Voice Type Classifier**: key child, other child, female adult, male adult. Already standard in
  child-language research. VTC 2 depends on pyannote.audio and has no stated licence (audit §8).
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
- [x] Previous/next between a batch's videos in the results viewer (item 4 of spec 008's viewer
      handoff).
      Done in Phase 0 (see "Results view isn't batch-aware").

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

**Last Updated**: 2026-09-30
**Target Release**: Early 2027, ahead of BCCCD (7–9 Jan 2027)
**Status**: Planning Phase — Public Release
