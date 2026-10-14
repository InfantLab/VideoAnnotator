# Research: Python 3.13 Support

Evidence base: the trial in [`dependency_audit_v1.6.0.md`](../../docs/development/dependency_audit_v1.6.0.md)
§7 (worktree `py313-trial`, CPython 3.13.15, 2026-10-01).

## R1. Version range and lockfile

- **Decision**: `requires-python = ">=3.12,<3.14"`; regenerate `uv.lock` with no other change.
- **Rationale**: the trial's lock for this range resolves 311 packages and only adds 3.13-only
  backports (`audioop-lts`, `standard-aifc`, `standard-chunk`, `standard-sunau`, `pyyaml-ft`),
  each behind a `python_version >= '3.13'` marker, so 3.12 installs don't change (FR-002, FR-004).
  An upper bound stays because 3.14 doesn't resolve for `face` (no TensorFlow for 3.14).
- **Alternatives**: `>=3.13` (drops 3.12; rejected, constitution V and the spec keep 3.12 for
  v1.6.0); no upper bound (rejected: 3.14 users would get a resolution failure deep in the `face`
  extra instead of a clear refusal, SC-004).

## R2. Unsupported-version message (SC-004)

- **Decision**: rely on the installers' check of `requires-python` for installs, and add a
  **startup warning** in the CLI and the server when the running Python is outside the supported
  range.
- **Rationale**, tested 2026-10-01 against a 3.14.7 environment with the new range:
  - pip refuses in the first step: `ERROR: Package 'videoannotator' requires a different Python:
    3.14.7 not in '<3.14,>=3.12'`. Meets SC-004.
  - `uv pip install .` (a local source checkout) **installs anyway**: uv doesn't check the
    local project's own `requires-python` there. Such a user would only find out later, from an
    obscure failure in an extra. The warning names the supported versions at the first run.
  - A warning, not an error: a core-only install may well work on 3.14, and someone trying it
    deliberately shouldn't be blocked.
- **Alternatives**: raise at import (rejected: blocks deliberate experiments, and breaks tooling
  that only imports the package); nothing (rejected: the `uv pip` path gives no signal at all).

## R3. Default Python for development (`.python-version`)

- **Decision**: add `.python-version` containing `3.13`.
- **Rationale**: uv uses it to choose (and download if needed) the interpreter for `uv sync` and
  `uv run`, so the dev container, a fresh clone and the Docker builds all get 3.13 without each
  setting it (FR-006, FR-007). uv downloads managed Pythons itself, which solves "Ubuntu 24.04
  ships 3.12".
- **Consequence**: CI's 3.12 jobs must override it (R4), or they'd silently test 3.13.
- **Alternatives**: `UV_PYTHON` in each Dockerfile and the devcontainer only (rejected: four places
  to keep in step, and a fresh clone outside them would still default to whatever Python is first
  on PATH).

## R4. CI matrix

- **Decision**: `python-version: ["3.12", "3.13"]` across the existing ubuntu / macOS / Windows
  matrix, with `UV_PYTHON: ${{ matrix.python-version }}` in the job's environment. ruff, the
  formatter check and mypy run once (Ubuntu, 3.13), as mypy does today (Ubuntu only).
- **Rationale**: FR-005 asks for the suite on both versions on every pull request. The `test` job
  already takes about 4–5 minutes per runner, so six runners in parallel cost little wall time.
  Windows keeps `continue-on-error`.
- **Alternatives**: 3.13 on Ubuntu only (cheaper, but the macOS/Windows packaging differences are
  exactly where 3.13 problems would show).

## R5. Lint and type-check targets

- **Decision**: keep ruff `target-version = "py312"` and mypy `python_version = "3.12"`; add the
  `Programming Language :: Python :: 3.13` classifier.
- **Rationale**: both settings mean "the oldest Python this code must run on". Keeping them at
  3.12 stops 3.13-only syntax or APIs slipping in while 3.12 is supported. They move to 3.13 when
  3.12 is dropped (v1.7.0).
- **Alternatives**: raising them to 3.13 (rejected: would allow code that breaks 3.12 users).

## R6. Docker images

- **Decision**: in `Dockerfile.cpu`, `Dockerfile.gpu` and `Dockerfile.dev`, run
  `uv python install` (reads `.python-version`) before `uv sync`, so the interpreter is part of the
  image layer. The apt `python3` packages stay for now (system tools may use them); the server runs
  under uv's 3.13.
- **Rationale**: the GPU image is also the dev container's base (`SKIP_IMAGE_UV_SYNC=true`), so
  installing 3.13 at build time means the container's post-create `uv sync` doesn't download it on
  every rebuild.
- **Alternatives**: a `python:3.13` base for the CPU image (rejected: different base from the GPU
  image, and the CUDA image has no 3.13 variant).

## R7. Verifying "same outputs" (SC-003, FR-004)

- **Decision**: a small script, `scripts/compare_pipeline_outputs.py`, runs a job through the
  server's `JobProcessor` on a given video in the current environment and writes each pipeline's
  annotations to JSON; a second mode compares two such dumps and reports differences.
  Documented in `quickstart.md`.
- **Rationale**: the trial used exactly this (a scratch version) and found
  `speaker_diarization` identical on 3.12 and 3.13. `videoannotator process` would be the natural
  tool but isn't implemented (roadmap Phase 2).
- **Alternatives**: the v1.4.x acceptance fixtures (`tests/integration/test_v144_parity.py`):
  their golden outputs were never captured, and re-baselining them is the library-upgrade specs'
  job.

## R8. Constitution and docs

- **Decision**: amend the constitution's Engineering Standards line "currently 3.12; 3.13 added when
  upstream deps allow" to "3.12 and 3.13" (PATCH version bump), and update `README.md`,
  `docs/installation/INSTALLATION.md`, `docs/usage/GETTING_STARTED.md` and `CLAUDE.md`'s Active
  Technologies line from "3.12+" / "3.12" to "3.12 or 3.13" (FR-009). CHANGELOG entry per FR-011.
- **Rationale**: the docs' "3.12+" has been wrong since v1.2.0 (only 3.12 installs).

## R9. Things checked and not needed

- **Source builds**: only `openai-whisper` (sdist) builds from source, as on 3.12; it built on
  3.13 in the trial with the existing `extra-build-dependencies`.
- **Packages whose classifiers stop at 3.12** (`open-clip-torch`, `tf-keras`, `webvtt-py`): all
  install and work on 3.13 in the trial; classifiers lag releases.
- **Free-threaded 3.13 (`3.13t`)**: out of scope; standard builds only.

## R10. Triton's compiled-launcher cache (found during implementation)

- **Finding**: after a 3.13 run on the same machine, `speech_recognition` on 3.12 returned no
  transcript. Whisper's word timestamps run a Triton kernel on GPU; Triton caches compiled C
  launchers in `~/.triton/cache` without keying them on the Python version, and a launcher built
  by 3.13 fails to load in 3.12 (`SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#'
  formats`). With a fresh cache, 3.12 transcribes correctly.
- **Decision**: `videoannotator/__init__.py` sets `TRITON_CACHE_DIR` to
  `~/.triton/cache/py<major>.<minor>` unless the user has set it. Verified: 3.12 → 3.13 → 3.12 on
  one machine all transcribe correctly.
- **Why it matters here**: this spec is what makes two Python versions on one machine normal
  (CI images, the dev container's switch to 3.13, users trying both).
- **Related, out of scope**: the pipeline turned the exception into "completed, no annotations"
  instead of a failure. Recorded on the v1.6.0 roadmap (Phase 2).
