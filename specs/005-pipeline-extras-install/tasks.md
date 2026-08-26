---
description: "Task list for Pipeline Extras Discoverability & Self-Service Install"
---

# Tasks: Pipeline Extras Discoverability & Self-Service Install

**Input**: Design documents from `/specs/005-pipeline-extras-install/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: Included. Not explicitly requested in spec.md's own text, but the ratified
constitution's Engineering Standards ("New pipelines and new public CLI/API surface MUST ship
with tests", coverage ≥80%) makes this a standing repo-wide requirement, matching how
`specs/004-extras-based-install/tasks.md` treated the same clause.

**Organization**: Tasks are grouped by user story (US1, US2, US3 — spec.md priorities P1, P1, P1;
all three are equal priority). Ordering below follows spec.md's own US1→US2→US3 listing, but the
piece User Story 3 depends on most — admin authentication — is pulled into Setup as **T001**, ahead
of every install-specific task, since spec.md's own User Story 1 acceptance scenarios already
presuppose "an authenticated admin caller" and building an install-triggering endpoint before its
authorization guard exists would mean shipping an unsafe endpoint, even temporarily.

## Format: `[ID] [P?] [Story] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (US1, US2, US3)

## Path Conventions

Single project (`src/videoannotator/`, `tests/`) per plan.md's Structure Decision.

---

## Phase 1: Setup

**Purpose**: Wire up admin authentication — the one primitive every install-related task below
requires, and a gap that exists today independent of this feature (see plan.md research.md §3).

- [X] T001 Extend `_validate_api_key_header()` in `src/videoannotator/api/dependencies.py` to
      include `"is_admin": user.is_admin` in its returned dict, and add the same field to
      whichever dict the `get_current_user`/TokenManager-backed path returns for a DB-backed user
      (same file). Add a `require_admin` dependency function to
      `src/videoannotator/api/middleware/auth.py` that depends on the existing
      `validate_required_api_key`, raises `HTTPException(403, detail="Administrator privileges
      required for this action.")` when `user.get("is_admin")` is falsy, and otherwise returns the
      user dict unchanged — per `contracts/restart-required-signal.md`'s Admin authentication
      section.
- [X] T002 [P] Unit tests for `require_admin` in `tests/unit/auth/test_require_admin.py`: no
      credentials → 401 (via the underlying `validate_required_api_key`), authenticated non-admin
      → 403 with the exact detail string above, authenticated admin → passes through and returns
      the user dict. Depends on T001.

**Checkpoint**: Admin authorization exists and is independently tested before any install-specific
code is written.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: The job-tracking model and the install-execution primitives every user story's
endpoints build on.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T003 Add an `ExtrasInstallJob` SQLAlchemy model to `src/videoannotator/database/models.py`,
      following the existing `Job` model's column conventions in the same file: `id` (GUID,
      primary key), `extra_name` (String, indexed), `requested_by_user_id` (GUID, FK →
      `users.id`), `status` (String: `pending`/`running`/`completed`/`failed`), `command_output`
      (Text, nullable), `created_at` (DateTime, server default now), `started_at` (DateTime,
      nullable), `finished_at` (DateTime, nullable) — per data-model.md. No new migration script
      needed; picked up automatically by the existing `Base.metadata.create_all` path.
- [X] T004 [P] Add a `known_extras() -> list[str]` helper to
      `src/videoannotator/registry/pipeline_loader.py` returning the currently-declared
      `[project.optional-dependencies]` group names, reusing whichever metadata source
      `extras_available()`/`install_hint()` already read (per research.md — validate against the
      actual running install's declared groups, never a hardcoded list, since the set can differ
      across versions).
- [X] T005 Create `src/videoannotator/api/extras_install.py` with: a `threading.Lock`-guarded dict
      tracking in-flight `extra_name → job_id` (FR-010 dedup); a module-level boolean flag with a
      `restart_required() -> bool` accessor (starts `False`, set `True` only when a job reaches
      `completed`, per `contracts/restart-required-signal.md`); a `resolve_install_command
      (extra_name: str) -> list[str]` function implementing research.md §1's pip-vs-uv detection
      (editable/source checkout of this repo with `uv` on `PATH` → `["uv", "sync", "--extra",
      extra_name, "--inexact"]` run with cwd at the project root; otherwise → `[sys.executable,
      "-m", "pip", "install",
      f"videoannotator[{extra_name}]=={importlib.metadata.version('videoannotator')}"]`); and a
      `run_install(job_id, extra_name, session_factory)` function that marks the job `running`,
      runs the resolved command via `subprocess.run(capture_output=True, text=True)`, stores
      combined stdout/stderr in `command_output`, sets `status` to `completed` (flipping the
      restart-required flag) or `failed`, sets `finished_at`, and clears the in-flight dedup entry
      — designed to run on a background `threading.Thread`, not the request thread. NOTE:
      `--inexact` was added after T019's real-world validation exposed that a bare `uv sync
      --extra X` prunes other, previously-installed extras groups — see research.md §1's
      correction and T022's regression-test finding below.
- [X] T006 [P] Add startup handling in `src/videoannotator/api/main.py` (alongside existing startup
      logic) that finds any `ExtrasInstallJob` rows still `pending`/`running` from a previous
      process and transitions them to `failed` with `command_output` noting the interruption —
      the crash/unclean-shutdown edge case from spec.md.
- [X] T007 [P] Unit tests for `resolve_install_command`'s pip-vs-uv branching and the dedup lock's
      in-flight tracking in `tests/unit/api/test_extras_install.py` — mock `subprocess.run`,
      `shutil.which`, and the filesystem checks; MUST NOT invoke a real install. Depends on T005.

**Checkpoint**: Foundation ready — admin auth, the job model, and the install-runner primitives all
exist and are independently unit-tested before any endpoint wires them together.

---

## Phase 3: User Story 1 - Install a needed pipeline without a terminal (Priority: P1) 🎯 MVP

**Goal**: An admin can trigger installing an extras group through the API and track it to
completion, without a terminal.

**Independent Test**: With only the core install present, call the install-trigger endpoint for
the `face` extras group; poll the returned job until it reports success; confirm the underlying
dependencies are now present in the environment.

### Implementation for User Story 1

- [X] T008 [US1] Implement `POST /api/v1/pipelines/extras/{extra}/install` in
      `src/videoannotator/api/v1/pipelines.py`: gated by `require_admin` (T001); validates `extra`
      against `known_extras()` (T004), returning `422` with the `known_extras` list on an unknown
      value per `contracts/extras-install-endpoints.md` *before* touching the dedup dict or DB;
      checks the in-flight dedup dict (T005) and returns the existing job's `{job_id, extra_name,
      status}` if one is already `pending`/`running` for that `extra_name`; otherwise creates an
      `ExtrasInstallJob` row (`pending`), spawns `run_install` (T005) on a `threading.Thread`, and
      returns `202` with `{job_id, extra_name, status}`.
- [X] T009 [US1] Implement `GET /api/v1/pipelines/extras/install-jobs/{job_id}` in
      `src/videoannotator/api/v1/pipelines.py`: `require_admin`-gated, `404` for an unknown
      `job_id`, otherwise returns `{job_id, extra_name, status, created_at, started_at,
      finished_at, command_output}` per `contracts/extras-install-endpoints.md` (the
      `restart_required` field is added in User Story 2's phase, not here).
- [X] T010 [US1] Implement the "already satisfied" fast path (FR-011) in the T008 handler: before
      creating a job row, check `extras_available([extra_name])` (existing
      `registry/pipeline_loader.py` helper); if already `True`, create a job that resolves
      directly to `completed` with `command_output` noting nothing needed to change, without
      spawning a subprocess.
- [X] T011 [P] [US1] API tests in `tests/api/test_pipeline_extras_endpoints.py` covering: a
      successful trigger returns `202` with the documented shape (mock `run_install`/`threading.
      Thread` so no real subprocess runs); polling a job through `pending → running → completed`
      (drive the row directly rather than waiting on a real thread); a `failed` job's
      `command_output` is non-empty; the already-satisfied fast path (T010); the dedup fast path
      (T008 returns the same `job_id` for a second concurrent request); `404` for an unknown
      `job_id`. Depends on T008–T010.
- [X] T012 [US1] Run `quickstart.md` §3 manually against a real dev server (one real `scene`
      extras install) to confirm the documented `curl` flow matches actual behavior; fix any
      drift back into the contract docs if found. RESULT: ran for real against a live dev server
      — trigger, poll through pending/running/completed, and the captured `command_output` all
      matched the contract exactly (~8 min real `uv sync --extra scene`, dominated by torch's CUDA
      wheels). Drift found and fixed: quickstart.md incorrectly claimed `scene` has no torch
      dependency (it does — `face` is the actual no-torch group per pyproject.toml); corrected in
      quickstart.md §3.

**Checkpoint**: User Story 1 fully functional — an admin can trigger and track a real install
end-to-end.

---

## Phase 4: User Story 2 - Know an install needs a restart to take effect (Priority: P1)

**Goal**: A completed install is clearly distinguishable, via the API, from "still unavailable,
restart to activate."

**Independent Test**: Complete an install job for a previously-uninstalled extras group without
restarting the server; confirm the pipeline catalog surfaces a "restart required" signal rather
than just `available: false` with no context, and that the signal clears after the server is
restarted and the pipeline becomes genuinely available.

### Implementation for User Story 2

- [X] T013 [US2] Add `restart_required: bool` (from `extras_install.restart_required()`, T005) to
      the `GET .../install-jobs/{job_id}` response built in T009.
- [X] T014 [US2] Add a top-level `restart_required` field to the existing `GET /api/v1/pipelines`
      response in `src/videoannotator/api/v1/pipelines.py`, sourced from the same
      `extras_install.restart_required()` accessor — additive only; the existing per-pipeline
      `available`/`install_hint` fields from spec 004 are unchanged (FR-012).
- [X] T015 [P] [US2] API tests in `tests/api/test_pipeline_extras_endpoints.py` asserting the
      semantics from `contracts/restart-required-signal.md`: `restart_required` is `false` on a
      fresh app instance; becomes `true` on both endpoints immediately after a job reaches
      `completed` (drive this by calling into the completion path directly / monkeypatching, not a
      real subprocess); stays `true` across a subsequent `failed` job for a *different* extra; a
      job that only ever reaches `failed` never sets it `true`. Depends on T013, T014.
- [X] T016 [US2] Run `quickstart.md` §4 manually, including an actual server restart, to confirm
      the flag reads `true` pre-restart and `false` + the pipeline genuinely `available:true`
      post-restart. RESULT: confirmed against the real dev server started in T012 — before
      restart, `scene_detection.available` was still `false` despite the completed install; after
      a real process restart, `GET /api/v1/pipelines` showed `restart_required: false` and
      `scene_detection.available: true`, and the T012 job's own status endpoint also reported
      `restart_required: false`. Matches contract exactly, no drift found.

**Checkpoint**: User Stories 1 and 2 both work independently and together.

---

## Phase 5: User Story 3 - Install requests are safe by construction (Priority: P1)

**Goal**: Prove, with tests, the guarantees T001/T004/T008 already build in: only an authenticated
administrator can trigger an install, and only for one of the project's own declared extras
groups — never an arbitrary string, never for an unauthenticated or non-admin caller.

**Independent Test**: Attempt to trigger an install (a) unauthenticated, (b) as a non-admin
authenticated user, and (c) naming an extras group that does not exist in the project's declared
set; confirm all three are rejected before any subprocess runs. (This test can run as soon as
Phase 3's T008 endpoint exists — it does not require T008's success path, T009, or Phase 4 to be
complete, which is why it is safe to build/verify T001's guard early per the Setup-phase note
above, even though its dedicated test task is sequenced here to match spec.md's own priority
listing.)

### Implementation for User Story 3

- [X] T017 [P] [US3] API tests in `tests/api/test_pipeline_extras_endpoints.py` for the three
      rejection scenarios: unauthenticated → `401` and no `ExtrasInstallJob` row created;
      authenticated non-admin → `403` and no row created; admin naming an unknown extras group →
      `422` (per `contracts/extras-install-endpoints.md`'s `known_extras` error body), no row
      created, and `run_install`/`subprocess.run` never invoked (assert via mock). Also assert the
      required ordering (FR-002/FR-003): an unauthenticated request naming an *invalid* extras
      group still returns `401`, not `422` — auth is checked first. Depends on T008 (the endpoint
      under test) and T001/T004 (the guards being verified).
- [X] T018 [US3] Concurrency test for FR-010 in `tests/api/test_pipeline_extras_endpoints.py`: fire
      two trigger requests for the same `extra_name` back-to-back and assert only one
      `ExtrasInstallJob` row/background thread is created, and both responses carry the same
      `job_id`. Depends on T008.
- [X] T019 [US3] Run `quickstart.md` §2 and the crash-recovery edge case in §5 manually against a
      real dev server (kill the process mid-install, restart it, confirm the orphaned job resolves
      to `failed` via T006, not stuck `running`). RESULT: confirmed against the real dev server —
      §2's unauthenticated (401) and invalid-extras-with-valid-admin (422) checks both matched;
      for the crash case, triggered a real `person` install, `SIGKILL`ed the server process while
      the job was `running`, restarted it, and the startup log showed "[STARTUP] Marked 1 orphaned
      extras-install job(s) as failed" — the job's status endpoint confirmed `status: failed` with
      an `[interrupted]`-prefixed `command_output`, never stuck `running`.

**Checkpoint**: All three user stories independently functional and verified together — this is
the complete feature.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Documentation and hygiene gates required by the Constitution's Engineering Standards.

- [X] T020 [P] Update `README.md`'s admin/pipeline-install section to mention the new in-app
      install action and link to the two new endpoints (FastAPI's auto-generated `/docs` covers
      exact request/response schemas — no separate API reference doc to hand-maintain).
- [X] T021 [P] Add an entry under `## [Unreleased]` in `CHANGELOG.md` describing the new
      self-service extras-install API and the `restart_required` signal. Also added two `Fixed`
      entries for bugs found and fixed during this feature's own real-world validation (T012/T016/
      T019): `start_server.sh`'s admin-email prompt firing on every restart, and its bare `uv sync`
      silently pruning previously-installed extras groups on every restart (see T005/research.md
      §1's `--inexact` correction).
- [X] T022 Run `ruff check .` and `mypy src/videoannotator` and fix any violations introduced by
      this feature's new/changed files (`api/extras_install.py`, `api/dependencies.py`,
      `api/middleware/auth.py`, `api/v1/pipelines.py`, `api/main.py`, `database/models.py`,
      `registry/pipeline_loader.py`). Both clean. FINDING: while re-running the full test suite as
      part of this sweep, `tests/api/test_api_server.py::TestJobEndpoints::test_submit_job_with_config`
      failed with "Pipeline 'scene_detection' is not available" — root-caused to T019's real
      `person` extras install (via a bare, non-`--inexact` `uv sync --extra person`) having pruned
      T012's earlier real `scene` install out of this dev container's venv. This is what led to the
      `--inexact` fix (see T005). Re-verified clean after the fix; not a pre-existing failure.
- [X] T023 Confirm test coverage for the new code stays ≥80% per the Constitution's Engineering
      Standards (`pytest --cov=videoannotator.api.extras_install
      --cov=videoannotator.database.models --cov-report=term-missing tests/unit tests/api`);
      add tests for any uncovered branch found. RESULT: initial run was 48.35% on
      `extras_install.py` (T007/T011/T015/T017/T018 deliberately mock around `run_install`/
      `start_install`/`_editable_checkout_root`'s real bodies, per this feature's "no real
      subprocess in the automated suite" rule). Added `TestEditableCheckoutRootRealFilesystem`,
      `TestRunInstall` (success/failure/OSError/missing-job branches, `subprocess.run` mocked but
      `run_install`'s own logic exercised for real against a temp DB), and `TestStartInstall` to
      `tests/unit/api/test_extras_install.py` — raised `extras_install.py` to 97.80%. Also found
      and removed `ExtrasInstallJob.to_dict()` (`database/models.py`) as genuinely dead code (the
      endpoint builds its response from ORM attributes directly, never calls it) rather than
      writing a test for unused code. Final: 97.80% / 80.41%, both ≥80%.

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately. T001 is deliberately the very
  first task in the whole feature (see Organization note above).
- **Foundational (Phase 2)**: Depends on Setup (T001's `require_admin` is referenced by name in
  T008's design, though T003–T007 themselves don't call it) — BLOCKS all user stories.
- **User Stories (Phase 3-5)**: All depend on Foundational phase completion.
  - US1 (Phase 3) has no dependency on US2 or US3's *tasks*, but its endpoint (T008) is written
    with T001/T004's guards already wired in from the start (see Organization note).
  - US2 (Phase 4) extends the endpoints US1 builds (T009, T014 both edit code T008/T009 introduce)
    — sequentially dependent on Phase 3, not independent in the usual spec-kit sense. This reflects
    that US2 is additive fields on US1's endpoints, not a separate endpoint.
  - US3 (Phase 5) is test-only against guards already built in Setup/Foundational/T008 — its tasks
    depend on T008 existing but not on Phase 4.
- **Polish (Phase 6)**: Depends on all three user stories being complete.

### Parallel Opportunities

- T002 can run once T001 lands (different file).
- T004, T006, T007 (marked [P]) can run in parallel with each other once T003/T005 (their
  respective prerequisites) land — different files, no shared state.
- T011, T015, T017 (marked [P]) can each be drafted in parallel with their sibling non-test tasks
  within the same phase once the endpoint(s) they test exist.
- T020, T021 (Polish, marked [P]) can run in parallel with each other and with T022/T023.

---

## Parallel Example: Foundational Phase

```bash
# Once T003 (model) and T005 (runner module) exist, these three can proceed together:
Task: "Add known_extras() helper in src/videoannotator/registry/pipeline_loader.py"
Task: "Add orphaned-install-job startup handling in src/videoannotator/api/main.py"
Task: "Unit tests for resolve_install_command and the dedup lock in tests/unit/api/test_extras_install.py"
```

---

## Implementation Strategy

### MVP First (Setup + Foundational + User Story 1)

1. Complete Phase 1: Setup (T001-T002) — admin auth exists and is tested in isolation.
2. Complete Phase 2: Foundational (T003-T007) — job model and install runner exist and are unit
   tested with no real subprocess execution.
3. Complete Phase 3: User Story 1 (T008-T012) — a real, safely-gated install can be triggered and
   tracked end-to-end.
4. **STOP and VALIDATE**: run `quickstart.md` §1-§3; confirm a real `scene` extras install works
   through the API from a clean core install.

### Incremental Delivery

1. Setup + Foundational → admin auth and job-tracking primitives ready, independently tested.
2. Add User Story 1 → install trigger + tracking works end-to-end (MVP).
3. Add User Story 2 → restart-required signal layered on top of US1's endpoints.
4. Add User Story 3 → the safety guarantees already built in Setup/US1 get their dedicated proof
   (rejection-path and concurrency tests).
5. Polish → docs, lint/type gates, coverage check.

---

## Notes

- [P] tasks = different files, no dependencies.
- [Story] label maps task to specific user story for traceability.
- No task in this feature invokes a real `pip install`/`uv sync` inside the automated test suite —
  every test task above explicitly mocks `subprocess.run`/`threading.Thread`/the install runner's
  entry point. Only the `quickstart.md`-validation tasks (T012, T016, T019) run a real install,
  and those are manual/CI-optional steps against a dev server, not part of `pytest`.
- Commit after each task or logical group, per this repo's atomic-commit workflow
  (`.specify/memory/constitution.md` Development Workflow).
