# Phase 0 Research: Pipeline Extras Discoverability & Self-Service Install

Four open questions from the spec's Technical Context were resolved by reading the existing
codebase rather than by external research — this feature extends established patterns already in
`videoannotator-core` (spec 002's storage abstraction, spec 004's extras/registry design, the
existing job-processing architecture) rather than introducing new technology.

## 1. What actually runs the install?

**Decision**: Shell out to `{sys.executable} -m pip install "videoannotator[<extra>]==<running
version>"` for a normally-installed (PyPI/wheel) deployment. When the running process is an
editable/source checkout of this repo itself (detected by an importable `videoannotator` whose
`__file__` resolves under a `src/videoannotator` tree with a sibling `pyproject.toml` declaring
`name = "videoannotator"`, and `uv` present on `PATH`), run `uv sync --extra <name> --inexact` in
that project root instead.

**`--inexact` correction (found during implementation, not originally planned):** a real
end-to-end run during this feature's own implementation (installing `scene`, then later `person`)
showed that a bare `uv sync --extra <name>` — without `--inexact` — resolves "core + `<name>`" and
*removes* any installed package outside that closure, including a *different* extras group
installed by an earlier call to this same function. Installing `audio` would silently uninstall a
previously-installed `face`. `--inexact` ("do not remove extraneous packages") is required to make
every install additive-only, matching what "install X, keep everything else working" actually
means. The same defect existed independently in `scripts/start_server.sh`'s own bare `uv sync` —
every restart via the documented path would have pruned any extras installed by this feature (or
manually) right before starting the server, defeating the "restart to activate" workflow this
spec's whole User Story 2 depends on. Both call sites now pass `--inexact`; verified by installing
`scene`, then running a bare `uv sync --inexact` (simulating a restart) and confirming `scene`
stayed installed, where it had been silently removed without the flag.

**Rationale**: `scripts/start_server.sh` already establishes `uv sync` as the install mechanism for
this repo's own dev/self-hosted deployment mode (the mode this session is running in), but `uv` is
a *development* tool for this project — it is not installed in a typical end user's environment who
ran `pip install videoannotator[scene]` per spec 004's User Story 1. `pip`, by contrast, is
guaranteed present (it is how the package got there in the first place, directly or via a wheel
installer). Pinning to the currently-running version (`importlib.metadata.version
("videoannotator")`) avoids an install action silently upgrading the whole package — and therefore
its already-imported, already-running code — out from under the live server process.

**Alternatives considered**:
- Always use `uv`: rejected — not guaranteed present outside this repo's own dev/self-hosted
  deployment, and the whole point of spec 004 was to support plain `pip install` end users.
- Always use unpinned `pip install videoannotator[<extra>]`: rejected — could silently pull a newer
  `videoannotator` release mid-install, which is a bigger, unrelated change to trigger from what the
  user asked for (installing one extras group).
- A vendored/bundled installer (e.g. shipping wheels): rejected as unnecessary complexity; standard
  PyPI resolution already handles this per spec 004.

## 2. Where does an install job live, and how is its lifecycle tracked?

**Decision**: A small, dedicated persistence record (new SQLAlchemy model alongside `User`, `APIKey`,
`Job` in `database/models.py`), auto-created via the existing `Base.metadata.create_all` path — no
new migration script needed, consistent with how the existing schema already picks up new models.
Execution itself is a single background thread spawned immediately on trigger (not a polled queue).

**Rationale**: The spec's FR-005/FR-010 language ("reuse the existing background-job/polling
pattern... same job-status model") refers to the *shape* of job tracking that annotation jobs
already use — an id, a small closed set of states (pending/running/completed/failed), timestamps,
and captured output/error — not literally the same table or the same `StorageBackend` abstraction
from spec 002. `StorageBackend` (`storage/base.py`) is shaped specifically around annotation output
(`save_annotations`, `BatchJob` metadata, reports); an extras-install job has no video, no pipeline
output, and no annotation — forcing it through that interface would be a worse fit than a small
sibling table, and spec 002 does not claim to be a generic job-queue abstraction.

A polled queue (mirroring `BackgroundJobManager` in `api/background_tasks.py`, which polls the DB
every `WORKER_POLL_INTERVAL` seconds for pending annotation jobs) is unnecessary here: annotation
jobs are queued because there can be many, submitted faster than `MAX_CONCURRENT_JOBS` can run them.
Install requests are rare, human-triggered, one-at-a-time-per-extras-group actions (FR-010 already
requires de-duplicating concurrent requests for the *same* group) — a thread spawned at request time
is sufficient and avoids adding polling latency to something a human is actively watching.

**Alternatives considered**:
- Reuse `StorageBackend`/`BatchJob`: rejected, wrong shape (see above).
- Reuse `BackgroundJobManager`'s poll loop: rejected as needless latency and complexity for a
  low-volume, synchronously-triggered action.
- In-memory-only job tracking (no DB row): rejected — fails the spec's crash/restart edge case
  ("MUST NOT be reported as silently running forever" after a crash) if the record doesn't survive
  process restart.

## 3. How does "admin-only" actually get enforced?

**Decision**: Add a new `require_admin` FastAPI dependency in `api/middleware/auth.py`, and extend
the API-key validation path (`_validate_api_key_header` in `api/dependencies.py`, plus the
`get_current_user`/TokenManager path it falls back to) to include the `is_admin` column already
present on the `User` model in the returned user dict.

**Rationale**: `User.is_admin` (`database/models.py`) already exists as a column — `setup-db`
already creates the bootstrap user with it set true — but nothing in the current API auth layer
reads or exposes it; `_validate_api_key_header`'s returned dict has no `is_admin` key at all. Spec
004's existing endpoints only ever needed "authenticated or not," never "authenticated as admin
specifically," so this gap was never exercised before. This feature is the first to need it, since
running an install subprocess on the server is a materially more sensitive action than reading a
pipeline list or submitting a video job.

**Alternatives considered**:
- Reuse `validate_required_api_key` alone (any authenticated user): rejected — does not satisfy
  spec User Story 3 / FR-003, which specifically requires *administrator* authentication, not just
  "any known API key."
- A separate admin-only API key type/prefix: rejected as unnecessary — the `is_admin` column and
  the existing single-admin bootstrap flow (`setup-db`) already model this; no new credential type is
  needed, only surfacing the existing flag through the auth dependency chain.

## 4. How does the restart-required signal get exposed without over-coupling to one endpoint?

**Decision**: A small module-level (in-process) flag, set true the moment any install job
transitions to `completed`, checked by both the install-job-status endpoint and
`GET /api/v1/pipelines` (extending the existing `available`/`install_hint` response, mirroring how
that endpoint already calls `extras_available()` per pipeline from `registry/pipeline_loader.py`).
Naturally resets to false on process restart, since it is in-process state — which is exactly the
condition ("has the server restarted since this install completed") the signal needs to track; no
separate persistence or manual clearing logic is needed.

**Rationale**: Simplest mechanism that exactly matches the semantics required (FR-007/FR-009): true
from the moment of a successful install until the process restarts, false otherwise, with zero risk
of drifting out of sync with the actual process's import state (unlike a DB-persisted flag, which
would need explicit code to clear it "on restart" — using in-process memory makes restart-clearing
free and correct by construction).

**Alternatives considered**:
- A persisted "restart_required" row/column, cleared by explicit start-up code: rejected — strictly
  more code than the in-process flag for identical behavior, and any clearing bug would make the
  signal lie about server state.
