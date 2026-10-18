# 🚀 VideoAnnotator v1.6.0 Development Roadmap

> **Rescoped 2026-10-17 (Caspar).** v1.6.0 was to be the public release. Announcing it and piloting
> it just before the viewer is redesigned would leave the public version in an awkward state, and
> pilot labs would report on a viewer about to change. So v1.6.0 is now a **quiet consolidation
> release**: tagged, citable and installable, not announced. The public release, with the
> redesigned viewer, the tutorial, the pilot and the announcements, is v1.7.0
> ([`roadmap_v1.7.0.md`](roadmap_v1.7.0.md)); the plugin ecosystem moved to v1.8.0. Phases 4–7
> below were moved there, except the model-default refresh (Phase 5c), which stays so the public
> version's outputs are stable. New here: Phase 3b, the pipeline guide (spec 026). References to
> Phases 4, 5, 6 and 7 in earlier entries mean the plan before this date; in v1.7.0 they are Phase 5
> (Agents), Phase 2 (Connectors, old 5a), spec 029 (Corpus overview) and Phase 3 (Docs).

## Release Overview

VideoAnnotator v1.6.0 is the **Consolidation Release**: one repository, current dependencies, a
clean first contact, and results out in formats researchers use. It is released quietly; the
public release is v1.7.0.

v1.5.0 made the tool usable by us without a terminal: in-app pipeline installs, batches, folder
ingest, the VLM pipeline, and the viewer bundled at `/viewer` (see
[`roadmap_v1.5.0.md`](roadmap_v1.5.0.md)). v1.6.0 is what it takes for **a lab we have never met**
to install it, run it on their own videos, and get the results into their own analysis without
asking us.

**Why this shape** (written for the public release, which v1.7.0 now is): on 2026-09-24 the JOSS submission
([#10182](https://github.com/openjournals/joss-reviews/issues/10182)) was sent for an editorial
scope check because no one outside the author team has used or contributed to it yet. JOSS accepts
publications, outside use, integration with other research tools, or benchmarks with reproducible
materials as evidence. JOSS declined on scope (2026-10-06) and is no longer a goal; the shape stands
for its own reason, outside use. The pilot follows the release, and a benchmark is not in v1.6.0
(see Phase 5b).

> Until 2026-09-24 this file held the plugin-ecosystem plan. That plan is now
> [`roadmap_v1.8.0.md`](roadmap_v1.8.0.md) (v1.7.0 until 2026-10-17), with its git history.

**Target Release**: as soon as Phases 3b and 5c are done. No deadline (BCCCD dropped 2026-10-14). The
pilot (and spec 024's T083) moved to v1.7.0's release candidates.
**Current Status**: Phases 0–1 done bar non-blocking items; Phase 2 done bar the long-video checks
(spec 024 walkthroughs done 2026-10-17); Phase 3 (Results Out, spec 025) built 2026-10-16; Phase 3b
(pipeline guide, spec 026) built 2026-10-17; Phase 5c (spec 030) done 2026-10-18
**Main Goal**: everything an outside lab will need under v1.7.0's new viewer, done and stable:
install, pipelines, results out
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
- ✅ **No model change without a reason on record.** A pipeline default changes on the upstream
  model's published evaluation, after its output on the demo clip has been checked against the
  baseline (`tests/integration/test_output_baseline.py`). The CHANGELOG and model card say why.
- ✅ **Pilot on what will be announced.** Outside testing runs on v1.7.0's release candidates, not
  on a viewer about to be replaced (2026-10-17).

---

## 📋 v1.6.0 Deliverables

Phases 0–3 made the release candidates; Phases 3b and 5c finish the release.

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
- [x] **Docker image size**: build the image slim and with `--build-arg EXTRAS=all`, and record
      both sizes against the v1.4.3 baseline. v1.5.0 set a target of 80% smaller but couldn't
      measure it. 2026-10-02: the three Dockerfiles became one `Dockerfile` on `ubuntu:24.04` (no
      `nvidia/cuda` base: torch's wheels carry CUDA 12.6 and cuDNN). Measured on the host
      (`handover_docker_images.md`): slim 1.35 GB (347 MB compressed), everything 14.9 GB
      (4.78 GB), v1.4.3 26.1 GB (8.89 GB): slim −95%, everything −43%. The 80% target holds for
      the slim default only. TensorFlow and torch both reach the GPU without the CUDA base; a
      six-pipeline job completes on GPU and CPU. The run found two install bugs (fixed): `httpx`
      missing from core, and no first API key on a fresh database. A CPU-wheel variant (~3.6 GB
      smaller) is blocked: TensorFlow then open_clip segfaults on CPU torch wheels; revisit when
      Phase 5 replaces DeepFace.
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
  - ~~Docs (Phase 2): the volume clone, the `uv sync --inexact --extra all` step a fresh clone needs
    (~5 min), the 16 GB-machine guidance, and the OOM symptom (`Killed`, exit 137) with the cap
    in `devcontainer.json`.~~ Done 2026-10-05: `INSTALLATION.md` "Dev Container" and the
    troubleshooting entry rewritten around the volume clone and the container's own cap, with the
    measured numbers; `.wslconfig` is now a fallback, not the first step. `Docker.md` has the
    measured image sizes.

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
*default model* (for example to `speaker-diarization-community-1`) is Phase 5c's. If a library upgrade can't keep the current default model runnable, the audit says so and
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
- [x] README rewritten for researchers: what it does, one screenshot, install in three steps per
      OS, open the viewer. Done 2026-10-05 (471 lines to about 190); the start script and shell
      aliases moved to `docs/usage/GETTING_STARTED.md`, API details to the docs.
- [x] Internal material out of the public docs: `docs/testing/` team handoffs (`Jerry-issues.md`,
      `Issues for Server Team.md`, `TEAM_HANDOFF_PACKAGE.md` and others) and `docs/Figure 1.docx`
      move to `docs/archive/` or go. Done 2026-10-05: those, the dated fix notes, the coding-agent
      setup guide and the v1.3 scripts audit are in `docs/archive/`.
- [x] Docs site (mkdocs-material on GitHub Pages) built from `docs/`, archive excluded.
      `mkdocs.yml`; CI's `docs` job builds it with `--strict`, `docs-site.yml` publishes it from
      master. Links that leave `docs/` become GitHub links in the site (`scripts/mkdocs_hooks.py`).
      Pages must be switched on once (Settings → Pages → Source: GitHub Actions).
- [x] Docs have one entry page (`docs/README.md`) and a link check in CI, so moving files can't
      leave dead links. (Unfinished since spec 001: T055, T056, T059.) `docs/README.md` rewritten
      by task (install, use, run it for a group, contribute); `scripts/validate_docs_links.py`
      checks relative links in README, CONTRIBUTING and `docs/` (archive aside).
- [x] No placeholder pages. The Datasets page says "Coming Soon" behind a disabled button, although
      spec 007's backend shipped in v1.5.0: wire it up or hide it.
      (Preset load/save in the job wizard landed in v1.5.0; the Datasets page, saved datasets and
      import/export from spec 007's viewer handoff are what's left.)
      Done 2026-10-04: hidden. The page had no menu link, only a URL; `/datasets` now goes to the
      Library. Wiring up saved datasets is a feature, so it gets a spec (below).
- [x] **Saved datasets in the viewer** (spec 007's viewer handoff): list, create, rename and delete
      the server's saved datasets (`/api/v1/datasets`, shipped in v1.5.0), start a job from one,
      and import/export them. Write a spec first. Decide with Caspar whether it's before rc1.
      Spec: `specs/018-saved-datasets-viewer/` (2026-10-04). Caspar: before rc1.
      Done 2026-10-04: Datasets page, wizard save/reuse with differences shown, export/import for
      datasets and presets, `videoannotator dataset` CLI, `GET /ingest/scan`. Checked in Chromium
      against a real server.
- [x] **`videoannotator job submit|status|results|list` send no API key** (found 2026-10-04): with
      authentication on, the default, they get 401. Give them `--api-key`/`VIDEOANNOTATOR_API_KEY`
      like `pipelines install` and `dataset`, and default `--server` to `127.0.0.1`.
      Done 2026-10-04 with spec 019.
- [x] Viewer: zero `tsc --noEmit` errors (24 on 2026-09-26), with typechecking in CI.
      Constitution 1.1.0 makes this an Engineering Standard, as are a 300 KB gzipped initial bundle
      (304 KB on 2026-10-02) and overlays naming the pipeline and version that drew them
      (Principle VI; not shown yet). All three are open follow-ups in its Sync Impact Report.
      Note (2026-10-01): plain `bunx tsc --noEmit`, the documented check, compiles nothing (the root
      `tsconfig.json` has `"files": []` and only references); run `-p tsconfig.app.json`. Still 24
      errors in 13 files, e.g. the OpenAPI `paths` type in `src/api/client.ts` lacks `/api/v1/jobs`.
      Done 2026-10-04: zero errors (`bun run typecheck`, app and node configs) and a CI step. The
      types are regenerated from the server (`scripts/gen_viewer_api_types.sh`). One was a real bug:
      the job page's `refetchInterval` read `.status` off the query, so the page never polled.
      The bundle size and overlay labels are split into the next item.
- [x] Viewer: the constitution's other two open Engineering Standards. A 300 KB gzipped initial
      bundle (304 KB on 2026-10-02), and overlays naming the pipeline and version that drew them
      (Principle VI).
      Overlay labels: spec `specs/017-output-provenance/` (2026-10-04), which also records the
      provenance Phase 3's methods paragraph needs.
      Done 2026-10-04 (spec 017): every output carries a provenance record, the job keeps it per
      pipeline, and the viewer labels every overlay with its pipeline and version (details on
      demand). Older files say "version not recorded".
      Bundle done 2026-10-04: every page but Home is lazy-loaded (`App.tsx`). The initial download
      fell from 301 KB JS + 14 KB CSS gzipped to 165 + 14 KB. Every route was checked in Chromium
      against the production build. Overlay labels are still open.
- [x] Viewer: one function decides which pipeline produced a file. Today there are four
      (`merger.ts`, `fileUtils.ts`, and two arrays in `FileUploader.tsx`) and they disagree.
      Start from `merger.ts`'s `detectJSONStructure` (2026-10-01), which classifies a parsed file by
      its fields and is covered by the contract test against real outputs.
      Done 2026-10-04: `lib/fileDetection.ts` (`detectFileType`, re-exported from `merger.ts`).
      `fileUtils.ts` and merger's prefix-sampling checks are gone. A file-name suffix is used only
      when the JSON is valid but says nothing (no annotations). Fixed along the way: the upload
      screen called every JSON "unknown", so it warned on every result set, and it couldn't open
      a video with only ELAN ground truth.
- [ ] A structural pass along the one path users take: install → add videos → run → review →
      export. Anything off that path moves or goes. A Playwright "first-time user" run on a clean
      machine files each point of friction as an issue.
      2026-10-05: first run done (Linux, fresh clone and home, README steps, Scene Detection):
      `docs/development/handover_first_time_user_run.md`. It found two blockers, both fixed: the
      first-start key wasn't admin, and Linux extras installs couldn't resolve torch. Job folders
      then moved beside the database (they followed the start directory). The other points are
      tracked in that note, not as GitHub issues, while the work is in-house (Caspar,
      2026-10-05); some need a decision (results-folder prompt, what moves or goes). Windows, macOS, GPU, a long video and the
      VLM and datasets paths are still to run.
- [x] **OpenFace 3 results are sensitive to GPU nondeterminism** (found 2026-10-01, spec 013): a
      1-pixel difference in RetinaFace's box between two runs of the same video changes that
      face's gaze and action-unit intensities by up to ~2.4. Offer a deterministic mode
      (`torch.use_deterministic_algorithms`, cuDNN deterministic) and record it in provenance, and
      measure run-to-run spread for every pipeline.
      Also order-dependent (found 2026-10-04): `scene_detection`'s CLIP scores on the demo clip
      match the baseline exactly when it runs after `face_openface3_embedding` and
      `person_tracking` in the same job, but differ in the third decimal (0.6352 vs 0.6340) when
      it runs alone. Something an earlier pipeline loads changes global torch/cuDNN state. A
      deterministic mode should pin those settings at the start of each pipeline.
      Done 2026-10-04. The cause of both was OpenFace's STAR `set_environment()`, which turned on
      `cudnn.benchmark` (also anomaly detection and denormal flushing) for the whole process.
      `utils/torch_settings.py`: every pipeline starts from the same settings, re-applied after
      its `initialize()`, and restored afterwards. `"deterministic": true` in the job config adds
      cuDNN deterministic and `use_deterministic_algorithms(warn_only=True)`. It is recorded in the
      job config and logged per pipeline. Not yet in the output files: that comes with Phase 3's
      provenance. Measured on the demo clip: two normal runs, two deterministic runs, and
      single-pipeline vs five-pipeline jobs all give identical files. Scene and OpenFace fixtures
      re-baselined (README). Not measured: spread across GPUs and drivers.
- [x] **A pipeline that fails must say so.** `speech_recognition` catches errors in transcription,
      logs them only to the pipeline log, and returns no annotations, so the job reports the
      pipeline as completed with an empty transcript (found 2026-10-01 through a Triton cache
      error). Raise instead, so the job shows the pipeline as failed with its error; check the
      other pipelines for the same pattern.
      Done 2026-10-03. Every pipeline was audited, and seven had the pattern (see CHANGELOG). The
      worst was `scene_detection`, which invented one whole-video scene on failure. Frame loops
      keep skipping single bad frames, but `FrameFailures` (`base_pipeline.py`) fails the run
      when every frame fails. Tests: `tests/unit/pipelines/test_pipeline_failures.py`.
      Left as is: per-face enrichment failures still only log a warning, so those faces lack the
      fields (OpenFace 3 landmarks/AUs, DeepFace emotion falling back to a bare box,
      person_tracking's size-based labels). Surfacing degraded fields belongs with the provenance
      work.
- [ ] **The API stops answering while a job runs** (found 2026-10-03): during the real-model
      baseline job, `GET /api/v1/jobs/{id}` timed out (over 30 s) or returned a body with no
      `status`. The host was using 6.4 GB of the 8 GB GPU, so every pipeline ran 2–4× slower than
      usual. Jobs run in an executor thread (`background_tasks.py`), so the likely cause is CPU or
      GIL starvation. Reproduce on an idle machine, and make the status route stay responsive. It
      is what the viewer polls.
      2026-10-04: not reproduced. The same test passed through the server in 32 s with the host
      still holding 5.3 GB of the GPU. STAR's anomaly detection, which the job used to inherit,
      was measured and doesn't slow inference. Keep this open until the Playwright run (E) has
      polled during a long job.
      2026-10-05, first-time-user run: 344 polls during a scene-detection job, median 25 ms,
      slowest 349 ms. Still to do with a long video and the heavy pipelines.
- [x] **`GET /api/v1/jobs/{id}/results/files/{pipeline}` returns `OUTPUT_FILE_MISSING`** for
      every result stored in the database (`output_file: database:/annotations/...`), which is
      what each job's results list advertises as its `download_url`. Same in v1.5.0 (found
      2026-10-02). Serve the file from the job folder, or stop advertising the URL.
      Done 2026-10-04: served from the job folder by the suffix in each pipeline's registry
      `outputs[].file`; results list every file (`files`, `?name=`) and advertise only files that
      exist. Tests: `tests/api/test_result_files.py`.
- [x] `videoannotator process <video>` is listed in `--help` but only prints "Direct processing
      is not yet implemented" (found 2026-10-01). Implement it on the shared job-execution path,
      or remove it.
      Done 2026-10-04 (`batch/local_job.py`): validated as a submission is, recorded in the
      server's database, run through `run_job_pipelines`. `--pipelines` is required: a default of
      every pipeline would be slow and need every extra. Tests: `tests/unit/cli/test_cli_process.py`.
- [x] **Queue position**: a pending job shows how many jobs are ahead of it ("3rd in queue"), in
      the job list, the job page and `GET /api/v1/jobs/{id}`. Today a queued job looks the same as
      a stuck one. (Planned since spec 001's T066.)
      Done 2026-10-04. The API already returned `queue_position` (oldest pending first, the order
      the worker takes them) on submit, status and list; the viewer now shows it
      (`lib/queuePosition.ts`) in `JobsTable` (job and batch lists) and on the job page.

**Run it again**: the path doesn't end at review. A researcher who likes a result wants the same
settings on more videos; one who doesn't wants to tweak and rerun. Today both mean rebuilding the
job in the wizard from memory, although every job already stores its `selected_pipelines` and
`config`.
- [x] **Rerun a job**: same videos, same settings, one click, as a new job that links back to the
      original (`rerun_of`). "Edit and rerun" opens the wizard prefilled.
      Spec (with the next two items): `specs/019-run-it-again/`.
      Done 2026-10-04 (all three items): rerun endpoints and CLI, job/batch page actions, wizard
      edit/settings modes, recent jobs and presets on the first step. Checked in Chromium against a
      real server. The Playwright first-time-user run (E) still has to confirm the cues are found.
- [x] **Reuse settings on new videos**: "Use these settings" on a job or batch opens the wizard at
      "Choose videos" with pipelines and config filled in; "Save as preset" writes the existing
      `saved_pipeline_presets` table.
- [x] **UX cues that lead there**, not only buttons in a menu: the actions sit on the job/batch
      result page where the user is looking when they decide; the wizard's first step offers
      "Start from a previous job" and recent presets before the blank form; a failed or partial job
      says "Fix settings and rerun"; a completed batch suggests "Run on more videos". Check each
      in the Playwright first-time-user run.
- [x] **Prompt library**: every VLM prompt used, in a job or a preview, saved to the database once
      (deduplicated by SHA-256, the same hash as the methods paragraph's provenance) with its
      model, first/last used and the jobs that used it. A browser to search, view, diff, name,
      star and reuse them; reusing one fills the prompt field.
      Spec (with the workbench): `specs/020-prompt-library-workbench/`.
      Done 2026-10-04 (with the workbench): library tables and API, recording from jobs and
      previews, Prompts page, Workbench page, CLI. Tested for real with Ollama (gemma4:e4b,
      qwen3.5:9b): a VLM job's provenance carries the model digest and quantisation, and a
      2 × 2 × 2 workbench round ran in 109 s. That test found the CRLF bug above.
- [x] **Prompt workbench**: the "test prompt" panel from inside the VLM pipeline config as a
      standalone page, since prompt design is iterative and deserves more room than a wizard step.
      Pick a video and frame (or burst), a model and a prompt; run; compare responses side by side
      across prompt versions or models; send the winner to a job or preset. Backend exists
      (`POST /api/v1/vlm/preview`, `GET /api/v1/vlm/models`); new work is the page, the prompt
      table and its endpoints. Also through the CLI/MCP (Phase 4).
- [x] **Compare two VLM jobs** on the same video: their labels on one timeline, with the frames
      where they disagree listed, and ELAN ground truth as a third row when there is one. The
      workbench compares prompts on single frames; this compares whole runs. Asked for in spec
      009's viewer handoff; comparing across a whole dataset stays in Phase 6.
      Spec: `specs/021-compare-vlm-jobs/`.
      Done 2026-10-04. Tested for real: gemma4:e4b every 5 s vs qwen3.5:9b every 2 s on the demo
      clip. 3 moments paired, 3 unpaired, all 3 disagreeing (TOUCH vs NO_TOUCH); the video seeks
      to a selected moment.
- [x] **Videos read where they are, results in one visible folder** (the first user is a
      researcher on their own laptop): "My folders" is the wizard's default on the server's own
      computer, with individual videos selectable and nothing copied; every run's results go to
      `~/VideoAnnotator/<run> (<date>)/<video>/` with a `run.json`, shown, opened and downloaded
      (one zip per run) from the viewer; moved videos are named and can be located again; Docker
      gets `VIDEOS_DIR`/`RESULTS_DIR` on loopback. Spec: `specs/022-videos-read-in-place/`.
      Done 2026-10-06. Its Docker walkthrough is replaced by spec 024 (below).
- [ ] **A container that feels local**: a start-up program for Docker and Podman that asks which
      folder the videos are in, shares it read-only at its real path, remembers it, and never shows
      the container's own folders. Spec: `specs/024-container-feels-local/`.
      Built 2026-10-06 (`videoannotator-start` for Linux/macOS and Windows, the slim image on GHCR,
      installed pipelines restored after recreation, Settings' Stop sharing). Quickstart
      walkthroughs done 2026-10-17: Windows by hand (T084), Linux reliable in daily use (T082).
      Open: the CI end-to-end runs with Docker and Podman (first run on a push to `master`), and a
      pilot researcher on macOS (T083, with v1.7.0's pilot).
- [ ] **The API and CLI from a launcher install** (found 2026-10-16): a researcher started with
      `videoannotator-start` has no `videoannotator` command on the host, and the API key it
      made sits in `start.conf`, which no command prints. Phase 3's R and Python notebooks and
      Phase 4's MCP server both need the URL and key. Proposal: `videoannotator-start api`
      prints the URL, the key and a three-line R, Python and curl example, and
      `videoannotator-start cli ARGS` runs the CLI inside the container with the key already
      set. Docs that say `docker compose exec videoannotator va ...` (`accessing_results.md`,
      `demo_commands.md`) move to whatever this settles on.
- [ ] **`INSTALLATION.md` down to the researcher's install** (2026-10-16: 729 lines): it opens
      with the launcher and then runs through the developer uv install, a manual OpenFace 3
      build that the `face-openface3` extra replaced, Docker Compose, the dev container, a
      second troubleshooting section, an architecture overview and performance tips. Keep the
      launcher steps, sharing, GPU, Ollama, system requirements and where results go. Move the
      uv install and dev container to a developer page linked from `CONTRIBUTING.md`, Docker
      Compose to its own page, troubleshooting into `troubleshooting.md`, and drop the manual
      OpenFace build.

- [x] **Fixes from 2026-10-16/17**, found while designing spec 023 and using the server daily:
      - **API keys were written to the log** (security): request logging put every request's
        headers, `Authorization: Bearer va_…` included, into `logs/api_server.log`, and auth
        failures logged 10–16 key characters. Headers are no longer logged; a key appears only as a
        sha256 fingerprint (`c2edfbd`). Anyone with logs from before this should delete them and
        make a new key.
      - **A terminal that says what matters** (`c2edfbd`): the server prints ready, each job
        started/finished/failed (with duration, results folder or reason), pipeline installs and
        stopping; per-request lines and library warnings go to `logs/` only. A viewer calling
        without a working key gets one plain hint instead of a stream of 401s.
      - **Hugging Face token from the viewer, Settings or the CLI, no restart** (spec 031:
        `0ffd059`, `ab400a7`, `fdc5346`, `9e8b530`): an admin saves it to a 0600 file beside the
        database, checked with Hugging Face first and never returned;
        `videoannotator settings huggingface-token` does the same from a terminal.
      - **Timeline colours that don't mislead** (`41a38d2`): every speaker had the same colour
        (the hue came from the ID's first letter, always "S"); hues now come from each value's
        rank. Timeline switches with no lane behind them are gone.
      - **The version on every docs page**, and `videoannotator --version` (`79582bd`).

**Not in this phase**: a visual redesign. It is v1.7.0's first phase (design in
[`specs/023-viewer-overhaul`](../../specs/023-viewer-overhaul/spec.md), built by specs 026–029).

---

### Phase 3: Results Out

**Problem**: each pipeline writes its own native format (COCO, WebVTT, RTTM, scene JSON). That is
right for provenance and wrong for analysis: to get one table in R, a researcher has to write a
parser per pipeline.

**Solution**:
- [x] **Tidy export**: one row per event (video, pipeline, track or person, start, end, label,
      value, confidence, model), as CSV and Parquet, per job and per batch or dataset, from the
      API, the CLI and a viewer button. This table also feeds the corpus view (Phase 6) and agents
      (Phase 4).
      Done 2026-10-16 (spec 025, `specs/025-results-out/`): one long table, a row per event and
      measure, 19 fixed columns (`contracts/table.md`, format version 1). Built from the result
      files and their provenance by `src/videoannotator/exporters/`, so
      `videoannotator export table <folder>` works on any results folder with no server, and the
      API and the viewer's Export menu give the same rows. Parquet is the optional `parquet`
      extra.
- [x] **ELAN export**: `.eaf` with one tier per pipeline track. The viewer already parses `.eaf`
      (`src/lib/parsers/elan.ts`), so this closes the loop with the tool many labs already code in.
      Done 2026-10-16: EAF 3.0, validated against ELAN's schema in tests; built from the table's
      rows; frame detections become "in view" spans. The viewer contract test opens the export
      and a copy edited in ELAN. Not yet opened in ELAN itself by a person (spec 025 quickstart).
- [x] **Methods paragraph**: `GET /api/v1/jobs/{id}/methods`, a CLI command and a viewer button.
      Returns the models, versions, weight revisions, parameters and citations as prose plus
      BibTeX, ready for a methods section. Needs the provenance fields from the reproducibility
      item in [`roadmap_v1.8_to_v2.0.md`](roadmap_v1.8_to_v2.0.md), pulled forward: model revision
      SHA, prompt SHA-256, quantisation.
      Done 2026-10-16, for jobs and runs (`/batches/{id}/methods` too). Citations live in each
      pipeline's metadata (`citations:`). Found on the way, not fixed: some metadata defaults
      differ from the pipelines' code defaults (scene `min_scene_length` 1.0 vs 2.0, Whisper's
      punctuation strings), so the paragraph lists those settings as changed.
- [x] Two worked analysis notebooks, one R and one Python, reading the tidy export of the demo
      video. Done 2026-10-16: `examples/notebooks/`; the Python one runs in CI, the R one was run
      on R 4.3 and is checked before each release (`tests/manual/results_out_r_notebook.md`).

**→ Release candidates**: rc1 was cut on 2026-10-14 to test the install before Phase 3 landed;
rc2 carries Phase 3.

---

### Phase 3b: Pipeline Guide

- [x] [`specs/026-pipeline-guide`](../../specs/026-pipeline-guide/spec.md): each pipeline's metadata
      says what it does in behavioural terms, splits its settings into behavioural (shown) and
      technical (collapsed), and gives its homepage, published benchmarks and citation; the
      viewer, the API and `videoannotator pipelines describe` show the same guide. Groundwork for
      v1.7.0's run setup and model cards; useful on today's Pipelines page.
      Done 2026-10-17: all 8 pipelines have a guide with sourced figures (spec 026
      research.md, "Sourced figures"); two settings the code never read are removed.
- [x] Remove the DeepWiki links (they point at the archived viewer repo) and "VAV" from Getting
      started. Done 2026-10-17: the footer's and Getting started's docs links go to the docs site.

---

### Phase 5c: Refresh Model Defaults

Each on the upstream evaluation plus a demo-clip check (principle: no model change without a reason
on record). Done before the public release so its outputs are stable.

Spec: [`specs/030-model-defaults`](../../specs/030-model-defaults/spec.md) (specified and clarified
2026-10-17; next: plan, tasks). Library upgrades are checked on the demo clip on their own, before
the model changes (FR-002). Each switch needs a real-model run on a GPU and a deliberate
re-baseline; spec 026's benchmark entries change with each default. Backward compatibility is
waived for this change (Principle V exception, recorded in the spec): old diarization and pose
models stay reachable as a setting, the old speech runtime goes.

- [x] **Speech**: `openai-whisper` `base` → faster-whisper with `large-v3-turbo`, in two steps
      (runtime with `base`, then the model). openai-whisper is removed (Q3), so v1.6.0rc1 is the
      last version with it.
      Done 2026-10-18: the runtime alone gave the identical transcript; turbo rewords 3 of the demo
      clip's 4 cues (hears "Peek-a-boo!"). A translate request with turbo now fails clearly.
- [x] **Diarization**: pyannote `speaker-diarization-3.1` → `speaker-diarization-community-1`
      (Q2): lower DER on 10 of 12 published sets, and one model licence to accept instead of three.
      Switched 2026-10-18. Demo clip: one speaker where 3.1 found two; listening showed 3.1 had
      split the mother's sentence, so community-1 stays.
- [x] **Person**: `yolo11n-pose` → `yolo26n-pose` (COCO-pose mAP50-95 50.0 → 57.2). Ultralytics
      stays for v1.6.0. Done 2026-10-18 (ultralytics 8.4 first: same output; it had moved the
      analytics switch-off, now followed).
- [x] **Face**: age and gender off by default, and their models neither downloaded nor run when off.
      Today `_deepface_actions()` (`pipelines/face_analysis/face_pipeline.py`) runs all three
      models whenever any is on: a bug, fixed here. DeepFace's own expression model stays (Q1),
      described in the guide as trained on adult faces; replacing `face_analysis` is later work.
      The detector is already RetinaFace (replaced the Haar cascade earlier in v1.6.0); its reason
      goes into the metadata and guide. Face analysis joins the demo-clip baseline if RetinaFace
      finds faces in it.
      Done 2026-10-18: 125 MB first download instead of 1.2 GB; RetinaFace finds 19 faces in the
      demo clip, so face analysis joined the baseline.
- [x] **OpenFace 3**: `openface-test` confirmed as the distribution CMU names (dormant since
      2025-06, non-commercial licence); the `==0.1.13` pin stays. Spec 030 writes this into its
      metadata (FR-026).
- [ ] Caspar: ask LAAC-LSCP about VTC 2's licence (the repository has none). Open; record the
      date asked. VTC 2 is a v1.7.0+ candidate whatever the answer.
- Moved to v1.7.0 (2026-10-17, spec 030 Q2): the half-day Sortformer spike. It informs lifting the
  torch cap later, not this release's default. Audit §8.

**Benchmark: not in v1.6.0 or v1.7.0** (was Phase 5b).

Dropped 2026-10-14 (Caspar). The pipelines wrap off-the-shelf models that come with their own
published evaluations, and a proper benchmark is a project in itself: no human-coded infant video
of our own, and the open sets either cover speech only (HomeBank's public VanDam corpora,
CC BY-NC-SA 3.0) or can't be redistributed (BabyView, on Databrary). Model cards (Phase 7) cite the
upstream evaluations and say plainly that few were trained or tested on infants. Labs can check a
pipeline against their own ELAN codes in the viewer's Compare page. Revisit after the pilot.

---

### Release (quiet)

- [x] ~~`v1.6.0rc2` with Phase 3~~: skipped (maintainer, 2026-10-18); a real-video test run and
      the 8/8 demo-clip baseline stood in for it.
- [x] Tag `v1.6.0`, GitHub release, `CITATION.cff` updated. No announcements. Released
      2026-10-18 (`31b26d9`, PR #12), after security updates for the dependencies Trivy and
      `bun audit` flagged. Zenodo's version DOI: check it was minted. **PyPI not yet**: the repo
      has no `PYPI_API_TOKEN`; once it does, re-run the release workflow's `build-and-publish` job.
- [x] Release notes say plainly that the viewer is being redesigned for v1.7.0, the public
      release.

---

## ➡️ Moved to v1.7.0 (2026-10-17)

The pilot, Phase 4 (Agents), Phase 5a (Connectors) and 5d (New pipelines), Phase 6 (Corpus
Overview, now spec 029's Videos page), Phase 7 (Docs and Tutorials) and the Launch. See
[`roadmap_v1.7.0.md`](roadmap_v1.7.0.md).

## ❌ Out of Scope for v1.6.0

- The viewer overhaul (v1.7.0).
- Publicity, the pilot, the tutorial and screencast (v1.7.0, written against the new viewer).
- Plugin ecosystem: entry-point discovery, `Dispatcher` ABC, `videoannotator-utils`
  ([`roadmap_v1.8.0.md`](roadmap_v1.8.0.md)).
- Remote and HPC dispatch ([`roadmap_v1.8_to_v2.0.md`](roadmap_v1.8_to_v2.0.md)).

---

## ✅ Success Criteria

- [x] Changing viewer code cannot ship a stale bundle: CI fails on a mismatch (Phase 0,
      `scripts/build_viewer.sh --check`).
- [x] Every version constraint in `pyproject.toml` and the viewer's `package.json` has a recorded
      reason in the dependency audit, and the release supports Python 3.13 (3.14 as soon as
      TensorFlow does).
- [x] Every change of default model has its reason in the CHANGELOG (Phase 5c).
- [x] The viewer has zero TypeScript errors (checked 2026-10-16).
- [ ] The viewer has no placeholder pages.
- [ ] Nothing leaves the user's machine unless they switched on a hosted backend.

Moved to v1.7.0: model cards for every default model; everything the viewer does available from
the CLI with `--json` and through MCP; the pilot's measures.

---

**Last Updated**: 2026-10-18 (released)
**Target Release**: as soon as Phases 3b and 5c are done
**Status**: Release candidates — Consolidation Release
