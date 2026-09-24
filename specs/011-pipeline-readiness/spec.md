# Feature Specification: Pipeline Readiness — Discover, Install, Activate, Set Up

**Feature Branch**: `011-pipeline-readiness`
**Created**: 2026-09-23
**Status**: Implemented 2026-09-24, pending the SC-001 manual run
([tests/manual/pipeline_readiness_e2e.md](../../tests/manual/pipeline_readiness_e2e.md)).
FR-017 (weight prefetch, optional P3) is deferred: it needs a download step per model library, and
the first-run size notes (FR-016) already make the wait visible.
**Input**: User description: "Pipelines should be discoverable and loadable by an end user. Today a
core-only server shows the viewer a single placeholder stub pipeline; getting any real pipeline
running still needs a terminal (install extras, restart the server, export tokens). Close that gap
end to end, backend and viewer together."

## Relationship to Existing Specs

This is the completion of a goal that specs 004 and 005 each delivered part of:

- **004** (extras-based install) made the core install slim and gave each pipeline `available` and
  `install_hint`. The listing *hides* unavailable pipelines by default.
- **005** (extras install) added an admin-only, API-triggered install job and a `restart_required`
  flag, plus a [viewer handoff](../005-pipeline-extras-install/viewer-handoff.md) describing the
  locked-card / Install / restart-banner UI.

What went wrong: each spec was "done" inside its own repo, but the user-level outcome never
happened. The viewer *source* implemented 005's handoff (locked cards, Install, admin detection via
`/auth/me`, restart banner) in late August, but the copy bundled in
`src/videoannotator/viewer_static/` was last rebuilt on 2026-08-24, before any of it, so `/viewer`
users never saw it. Re-copying the bundle (commit `636d8a4`, 2026-09-23) fixed that. Driving it in a
browser against a real server then showed the 005 flow working end to end, after these fixes:

- Install ran `uv sync` against `<root>/.venv` rather than the running server's environment, and on
  Windows failed overwriting the running `videoannotator.exe` (fixed, `b2cd25a`:
  `UV_PROJECT_ENVIRONMENT=sys.prefix`, `--no-install-project`).
- On completion the viewer never re-fetched the catalog, so the restart banner never appeared and the
  card silently reverted to "Not installed" (fixed in the viewer, `bbb5b68`).
- The embedded build had a developer's API token compiled in, and so did every bundle committed since
  `bddb546`, including `master` (fixed in the viewer, `80ef5d4`: `build:embedded` uses
  `--mode embedded` with a committed `.env.embedded` that blanks it).

What remains, and what this spec is for, are the gaps that still force a terminal:

1. **Restart is manual** (005 FR-014 deliberately deferred it). The user has to get to the server
   host to activate what they just installed.
2. **`available` means "Python packages present", not "usable".** Speaker diarization still needs
   `HF_AUTH_TOKEN` plus a model licence accepted on Hugging Face; `vlm_annotation` needs a reachable
   Ollama with a pulled model; most pipelines silently download model weights (often GBs) on first
   run. None of this is visible before a job fails.
3. **Installing re-syncs the whole lockfile while the server runs.** `uv sync --extra X --inexact`
   also moves every *already installed* package to its `uv.lock` version. In an environment that has
   drifted from the lock (any `git pull` that changes `uv.lock`), that rewrites packages the server
   has imported, and on Windows fails outright replacing a loaded native extension (observed:
   `yaml/_yaml.cp312-win_amd64.pyd`, os error 5). On Linux it "succeeds" and forces a restart
   that wasn't needed for the extra itself. See FR-005a.

(A fourth gap in the first draft, a test-fixture stub pipeline in the production metadata directory,
turned out to be a local untracked copy of `tests/fixtures/stub_pipeline.yaml`. It was never in git
and has been deleted. FR-004 keeps a guard so it can't come back.)

This spec replaces the binary `available` model with a **readiness state** that carries the next
action, and adds the backend actions needed so every state can be resolved from the viewer. It
**supersedes 005 FR-014** (auto-restart out of scope) and 005's handoff non-goal "Auto-restart";
everything else in 004/005 stays as is, and all changes are additive.

**Scope spans two repos.** Backend (this repo) is specified here; the viewer work is specified in
[viewer-handoff.md](viewer-handoff.md) for the `video-annotation-viewer` repo's own
`/speckit-specify`. Both build against [contracts/readiness-contract.md](contracts/readiness-contract.md),
which is the single source of truth for shapes. **Neither side is done until SC-001 passes** — that
is the lesson from 004/005.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - See every pipeline and what it takes to use it (Priority: P1)

A researcher opens Select Pipelines on a fresh core-only server. They see every
pipeline the server knows about, each labelled with where it stands — Ready, Not installed (with the
download size), Installing, Needs restart, Needs setup (with what's missing) — and the single next
step to move it forward.

**Why this priority**: Discoverability is the precondition for everything else; today the user
cannot even learn that face/audio/scene/person pipelines exist.

**Independent Test**: With no extras installed, `GET /api/v1/pipelines?include_unavailable=true`
returns all shipped pipelines (and no stub), each with a `readiness.state` of `not_installed` and a
`readiness.next_action` of `install`, and the extras group's approximate download size.

**Acceptance Scenarios**:

1. **Given** a core-only server, **When** the pipeline list is requested with
   `include_unavailable=true`, **Then** every shipped pipeline is returned with a readiness state,
   and the stub fixture is not among them.
2. **Given** the `audio` extras are installed but `HF_AUTH_TOKEN` is unset, **When** the list is
   requested, **Then** `speaker_diarization` is `needs_setup` with a blocker naming the missing
   token and linking to where to get it, while `speech_recognition` (no token needed) is `ready`.
3. **Given** the `llm` extras are installed but Ollama is unreachable, **When** the list is
   requested, **Then** `vlm_annotation` is `needs_setup` with a blocker saying Ollama isn't
   reachable at the configured URL.
4. **Given** an old client that calls `GET /api/v1/pipelines` with no parameters, **When** the
   response arrives, **Then** it contains exactly what it did before this spec (available pipelines
   only; existing fields unchanged) plus the new additive fields.

---

### User Story 2 - Installed means usable, without a terminal (Priority: P1)

An admin clicks Install on a locked pipeline. When the install finishes, the pipeline becomes Ready
on its own if that's safe; if the install replaced a library the running server already loaded, the
viewer offers **Restart server**, the admin clicks it, the viewer waits through the restart, and the
pipeline comes back Ready.

**Why this priority**: The manual restart is the step that most visibly breaks "self-service".
005 shipped the install; without activation it still ends in a terminal.

**Independent Test**: On a core-only server, install `scene` via the API; confirm the job completes
with `activation: "live"` and `scene_detection` is `ready` without restarting. Separately, simulate
an install that changes the version of an already-imported distribution; confirm
`activation: "restart_required"`, call `POST /api/v1/system/restart`, poll health until `boot_id`
changes, and confirm the pipeline is `ready`.

**Acceptance Scenarios**:

1. **Given** an install that only adds distributions not yet imported by the server process,
   **When** the install job completes, **Then** the server refreshes its import caches and
   availability, the job reports `activation: "live"`, the global `restart_required` stays false,
   and the affected pipelines report `ready` (or `needs_setup`) on the next listing.
2. **Given** an install that changed the version of a distribution already imported in this process
   (e.g. numpy), **When** the job completes, **Then** it reports `activation: "restart_required"`,
   lists the conflicting distributions, and the affected pipelines report `restart_required`.
3. **Given** an admin and a server started in restartable mode, **When** they call
   `POST /api/v1/system/restart`, **Then** the server responds `202` and re-executes itself; the
   health endpoint's `boot_id` changes once it is back.
4. **Given** annotation jobs are currently running, **When** restart is requested without
   `force=true`, **Then** it is refused with `409` naming the running jobs.
5. **Given** a server running with `--reload` or `--workers > 1`, or under a supervisor that
   does not support self-restart, **When** restart is requested, **Then** it is refused with `409`
   `RESTART_UNSUPPORTED` and a human-readable instruction for restarting manually.

---

### User Story 3 - See what a pipeline still needs, before a job fails (Priority: P2)

A pipeline says "Needs setup: HF_AUTH_TOKEN isn't set on the server", with a link to get a token and
a reminder to accept the pyannote model licence. For the VLM pipeline, "Needs setup: Ollama not
reachable" links to the existing Ollama diagnostics.

**Decision (Caspar, 2026-09-24)**: the Hugging Face token is *deployment configuration*, set in the
container environment (`docker-compose.yml` and the devcontainer pass `HF_AUTH_TOKEN` through from
the host), not entered in the viewer. So there is no secrets API: readiness reports what is missing
and where to set it, and the operator sets it. This drops the first draft's secrets API (old FR-013 to FR-015)
and contract section 6.

**Why this priority**: Only some pipelines have setup requirements, but for those it's a hard wall
today and the failure only shows up mid-job.

**Independent Test**: With `audio` installed and `HF_AUTH_TOKEN` unset, `speaker_diarization` is
`needs_setup` with a `secret` blocker naming `HF_AUTH_TOKEN`; start the server with it set and it is
`ready` with a licence note.

**Acceptance Scenarios**:

1. **Given** a pipeline that declares `secret: HF_AUTH_TOKEN`, **When** the variable is unset in the
   server's environment, **Then** it is `needs_setup` and the blocker's message says to set it in
   the server's (container's) environment. The value itself is never read into any response.
2. **Given** the variable is set, **When** the list is requested, **Then** the pipeline is not
   blocked by it (the licence stays a note: it can't be checked locally).

---

### User Story 4 - Know about big first-run downloads before they happen (Priority: P3)

Before installing or first running a pipeline, the user can see roughly how much it will download
(packages and model weights), and optionally pre-fetch the weights so the first real job doesn't
spend ten minutes downloading.

**Why this priority**: Nice to have. Makes the slow parts honest; not a blocker.

**Independent Test**: `GET /api/v1/pipelines/extras` lists each extras group with an approximate
download size; a pipeline whose weights aren't cached reports a `weights_cached: false` note; `POST
/api/v1/pipelines/{name}/prefetch` starts a trackable job that downloads them.

**Acceptance Scenarios**:

1. **Given** a core-only server, **When** the extras groups are listed, **Then** each shows its
   pipelines, whether it's installed, and an approximate download size.
2. **Given** an installed pipeline whose weights are not cached, **When** listed, **Then** it is
   still `ready` (weights are not a blocker) but carries a note with the approximate weight size.

---

### Edge Cases

- Install completes while another install is running: activation is evaluated per job; a restart
  offered by one must not interrupt the other (restart refuses with `409` while any install job is
  `pending`/`running`, regardless of `force`).
- Server killed mid-restart or fails to come back: the viewer's wait must time out with guidance;
  the server must not persist a "restarting" state that survives a failed boot.
- Hot activation succeeds but the pipeline then fails to import (e.g. a native lib missing): the
  pipeline reports `needs_setup` with a blocker of kind `import_error` and the captured message —
  never `ready`.
- Docker deployment: `uv run` in the container re-syncs the environment at start and will prune
  extras installed at runtime unless `UV_NO_SYNC=1` / `--inexact` is honoured (see 005 research.md
  §1). Restart must not undo an install.
- Secret file present but the environment also sets the same variable: environment wins, API reports
  `source: "environment"`.
- A third-party plugin (v1.6.0 plugin work) declares `requires_setup`: handled by the same schema;
  secret names from plugins are allowlisted the same way.

## Requirements *(mandatory)*

### Functional Requirements

**Discoverability**

- **FR-001**: Every pipeline entry in `GET /api/v1/pipelines` and `GET /api/v1/pipelines/{name}` MUST
  include a `readiness` object: `state` ∈ {`ready`, `not_installed`, `installing`,
  `restart_required`, `needs_setup`}, `next_action` ∈ {`none`, `install`, `restart`, `setup`, `wait`},
  `blockers` (list), `notes` (list), and `extras_group`. Exact shape in the contract.
- **FR-002**: `available` and `install_hint` MUST keep their current meaning; `restart_required` at the
  top level MUST keep its meaning. Default listing behaviour (hide unavailable unless
  `include_unavailable=true`) MUST NOT change.
- **FR-003**: `GET /api/v1/pipelines/extras` MUST list every declared extras group with: name,
  pipelines it enables, installed (bool), approximate download size (MB), whether it pulls a GPU
  torch build, and the in-flight install job id if any.
- **FR-004**: A test MUST assert that no metadata file in `src/videoannotator/registry/metadata/`
  has a `module_path` under `tests.`. (`tests/api/test_ingest_endpoints.py` uses `"stub_pipeline"`
  but doesn't need it registered: unknown names pass `validate_pipeline_selection`.)

**Activation**

- **FR-005a**: While the server is running, an extras install MUST only add what the extras group
  needs, never re-sync the rest of the environment to the lockfile. Install the group's own
  requirements (from the installed distribution's metadata, `extra == "<group>"`) into
  `sys.executable`'s environment, e.g. `uv pip install --python <sys.executable> <reqs>`, falling
  back to `python -m pip install`. This keeps the "only new distributions" case, which FR-005's
  `live` activation depends on, as the common case, and avoids Windows file-lock failures.
- **FR-005**: On a completed install, the server MUST invalidate import caches and recompute
  availability. It MUST decide `activation` as `restart_required` iff any distribution whose version
  changed during the install has a top-level module present in `sys.modules`; otherwise `live`.
- **FR-006**: The install job MUST expose `activation` and, when `restart_required`,
  `conflicting_distributions` (name, old version, new version). Global `restart_required` MUST only
  become true for `activation: "restart_required"`.
- **FR-007**: `POST /api/v1/system/restart` (admin-only) MUST return `202` and then re-execute the
  server process with the same arguments, after the response is flushed. It MUST refuse with `409`:
  `RESTART_UNSUPPORTED` when running with `--reload`, `--workers > 1`, or when the server was not
  started in restartable mode; `JOBS_RUNNING` when annotation jobs are running and `force` is not
  set; `INSTALL_IN_PROGRESS` whenever an install job is pending/running.
- **FR-008**: The health endpoints (`GET /health`, `GET /api/v1/health`, `GET /api/v1/system/health`)
  MUST include a `boot_id` (random per process start) and `started_at`, so a client can detect that a restart has
  completed.
- **FR-009**: A restart MUST NOT prune runtime-installed extras. Docker/compose server commands and
  the devcontainer MUST use a non-pruning start (`UV_NO_SYNC=1` or `uv run --no-sync`). The
  devcontainer half is done (`a3d62fd`); Docker/compose remains.

**Setup**

- **FR-010**: `PipelineMetadata` MUST gain an optional `requires_setup` list. Each item has `kind` ∈
  {`secret`, `service`, `licence`}, `name`, `description`, optional `help_url`. Absent means none.
  Additive; the stub-style forward-compat guarantee of 004 SC-007 still holds.
- **FR-011**: `speaker_diarization` MUST declare `secret: HF_AUTH_TOKEN` and `licence:
  pyannote/speaker-diarization` (with the Hugging Face URL). `vlm_annotation` MUST declare `service:
  ollama`. `audio_processing` also runs pyannote diarization (`audio_pipeline_modular.py`) and
  needs the same declarations as `speaker_diarization`. Other shipped pipelines are audited and
  declare whatever they actually need.
- **FR-012**: Readiness evaluation MUST check each requirement: `secret` → set in the server's environment;
  `service: ollama` → reuse `diagnostics.ollama.diagnose_ollama` (cached ≤ 30 s so
  listing stays fast); `licence` → cannot be verified locally, always reported as a `note`, never a
  blocker.
- **FR-013**: A `secret` blocker's `message` MUST say where to set it ("Set HF_AUTH_TOKEN in the
  server's environment, e.g. the container env, and restart"). Secret values MUST never appear in
  any response or log line. The server MUST NOT offer an API for setting secrets (decision under
  User Story 3). `docker-compose.yml` and the devcontainer MUST pass the declared secrets through
  from the host environment (done for `HF_AUTH_TOKEN`).

**Weights (P3)**

- **FR-016**: Pipelines MAY declare `weights` in metadata (model ids + approx MB). When declared and
  not in the local HF/torch cache, readiness MUST include a `weights_not_cached` note (not a
  blocker).
- **FR-017**: `POST /api/v1/pipelines/{name}/prefetch` (admin) MAY start a trackable job that
  downloads declared weights, reusing the extras install job table/status vocabulary. **Deferred** (see Status).

**Release / cross-repo**

- **FR-018**: Re-copying `viewer_static/` is part of *every* spec that has viewer work, not just
  releases: the spec isn't done until `bun run build:embedded` output from the matching viewer commit
  is committed here and SC-001 has been run against it. The commit message names the viewer commit.
  Before committing, check the bundle contains no `va_` API key (the embedded build blanks
  `VITE_API_TOKEN`; this check catches a regression).

### Key Entities

- **Pipeline readiness**: derived, per request, from installed extras, in-flight install jobs,
  activation outcome, declared setup requirements and their current status. Not persisted.
- **Setup requirement**: a declared precondition of a pipeline (secret, external service, licence).
- **Secret**: a named value a pipeline needs, read from the server's environment only.
- **Boot identity**: `boot_id` + `started_at`, per server process.

## API Contract for Downstream Consumers

See [contracts/readiness-contract.md](contracts/readiness-contract.md). Once published, state names,
action names and field names are a stable contract under the same forward-compatibility rules as
004's registry schema.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001 (the one that matters)**: On a fresh core-only install, using only the bundled viewer in
  a browser, a user with an admin key gets `face_analysis` from not visible to a completed annotation
  job **with zero terminal commands** after `videoannotator server` was started. Repeat for
  `speaker_diarization` with `HF_AUTH_TOKEN` set in the container env. Recorded as a manual test script in
  `tests/manual/pipeline_readiness_e2e.md` and run before release.
- **SC-002**: A core-only listing with `include_unavailable=true` returns every shipped pipeline and
  zero test fixtures.
- **SC-003**: Installs that add only new distributions activate live in 100% of cases in the test
  matrix (`face`, `scene`, `person`, `audio`, `llm` from core-only); no restart offered.
- **SC-004**: After `POST /system/restart`, the server is serving again with a new `boot_id` within
  30 s on the dev container (excluding install time).
- **SC-005**: No response or log line contains a secret's value (the readiness check only tests
  whether the variable is set).
- **SC-006**: Listing latency with `include_unavailable=true` stays under 200 ms p95 on the dev
  container with Ollama unreachable (service checks are cached and time-boxed).
- **SC-007**: Existing clients see no breaking change: all current `tests/api` pipeline tests pass
  unmodified except those that relied on the stray stub.

## Assumptions

- Single-user local deployments are the main case; `generate-token` already makes the first user
  admin, so admin gating rarely blocks. Multi-user role design is out of scope.
- Self-restart via `os.execv` of the original `sys.argv` is acceptable for `videoannotator server`
  run directly or under `uv run --no-sync`. Under Docker, a container restart policy
  (`restart: unless-stopped`) plus process exit is an acceptable alternative; the implementation
  plan picks one per launch mode and reports it via a `restart_mode` field in the health responses.
- Licence acceptance on Hugging Face cannot be checked without making a network call with the
  user's token; we surface it as guidance, and a failed model download at job time still reports
  clearly (existing behaviour).
- Cancelling an in-flight install remains out of scope (unchanged from 005).
- Size estimates are maintained by hand in metadata and may drift; they are labelled "approx.".
