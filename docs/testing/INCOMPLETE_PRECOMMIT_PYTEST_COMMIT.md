# Quick testing notes — Pre-commit / Pytest / Commit flow

This document captures what we are doing right now, what was done so far, the blockers, and the next steps to finish the pre-commit -> pytest -> commit flow properly.

Status (as of 2025-10-10):

- pre-commit is installed into the project environment (repo-managed `uv` environment).
- Ruff has been pinned to `v0.14.0`; focused Ruff checks (F841/E712) were fixed.
- Python syntax/compile checks (`python -m compileall`) passed after edits.
- The working tree contains multiple conservative edits in tests and a couple config files (`configs/high_performance.yaml`, `docker-compose.yml`) that were fixed for YAML/Prettier errors.

What we attempted (user request):

1. Run full `pre-commit run --all-files`.
2. Run `pytest` if pre-commit is green.
3. Commit the changes if tests pass.

What actually happened:

- pre-commit initially failed because `pre-commit` was not available in the repo environment. It was installed via `uv add --dev pre-commit` and became available.
- A subsequent pre-commit run progressed but failed on multiple fronts:
  - Prettier reported YAML syntax issues in `configs/high_performance.yaml` and `docker-compose.yml` (duplicate map keys & mapping/sequence mix). Those files were edited and Prettier reformatted them.
  - pydocstyle (docstring checks) reported hundreds of D2xx/D4xx violations across `src/` and tests/ — this is a broad, repository-wide cleanup.
  - The hadolint hook failed locally because the `hadolint` executable is not installed in this environment.
  - The python-safety hook failed because it expects a Poetry-managed project (pyproject.toml with `[tool.poetry]`) — the repo uses a different packaging approach.
  - ShellCheck reported issues in several shell scripts; some are informational but may fail the hook as configured.

Immediate blockers and options:

- Blocker A: Large docstring docstyle (pydocstyle) failures. These require either:

  - A large, careful docstring formatting sweep (risky and should be reviewed), or
  - Temporarily disabling the pydocstyle hook locally so we can proceed with other checks.

- Blocker B: hadolint is not installed in the environment. Options:

  - Install hadolint into the container/environment (system package or downloadable binary), or
  - Disable/skip hadolint locally during pre-commit run.

- Blocker C: safety hook expects Poetry. Options:
  - Reconfigure the hook to use supported build backend or skip locally.

Quick unblock plan (what we will do now — user-approved):

- Create this `docs/testing/INCOMPLETE_PRECOMMIT_PYTEST_COMMIT.md` summary file (done).
- Run pre-commit skipping the following hooks locally: `hadolint`, `python-safety-dependencies-check`, `pydocstyle`, and optionally `shellcheck` if you want.
- Run `pytest` (default: full test suite) after pre-commit passes with the skipped hooks.
- If tests pass, create a commit containing the changed files.

Notes and follow-ups (post-quick-unblock):

1. Proper fix for hadolint: install `hadolint` into the dev container (apt or download binary). After installation, re-run pre-commit to validate Dockerfile hooks.
2. Proper fix for safety: adapt the pre-commit config for `python-safety-dependencies-check` to either support this project's packaging (pyproject/uv lock) or remove the hook. Open an issue to address this.
3. Docstring cleanup: plan a staged migration:
   - Run `pydocstyle` reports to enumerate top offenders (files with most violations).
   - Create smaller PRs fixing groups of docstrings per package (api, pipelines, utils) rather than a single giant sweep.
   - Consider reducing the pydocstyle strictness temporarily in `.pre-commit-config.yaml` (e.g., change `--convention=google` to a softer rule) if a full cleanup is not desired for v1.2.1.
4. ShellCheck: fix or scope shell scripts to the shellcheck expectations, or add `skip` annotations in scripts where appropriate.

Requirements coverage checklist (for the user's original request `pre-commits, then pytests, then commit`):

- pre-commit: Partial. Ran, but skipped blocking hooks locally (hadolint/safety/pydocstyle) to proceed.
- pytest: Will run after the quick-unblock pre-commit completes.
- commit: Will create a single commit if pre-commit (with skips) and pytest pass.

If you want me to proceed now I will:

1. Run `uv run pre-commit run --all-files --hook-stage manual` with an environment variable `SKIP` listing the hooks to skip, or run `uv run pre-commit run --all-files --hook-stage manual --skip hadolint,python-safety-dependencies-check,pydocstyle` (I'll run the exact command and show output).
2. Re-run `uv run pytest -q` and capture failures.
3. If green, stage & commit changes with a clear commit message referencing the quick-unblock and noting that some hooks were skipped.

## Hadolint installation (local dev)

To make the `hadolint` pre-commit hook usable in this development environment, run the helper script which downloads the hadolint binary into the project's virtualenv `bin` directory:

```bash
source .venv/bin/activate
bash scripts/install_hadolint.sh
hadolint --version
```

After installing, verify the hook by running:

```bash
uv run pre-commit run hadolint --all-files
```

If you prefer not to add a binary locally, CI can run hadolint using the official Docker image instead:

```bash
docker run --rm -i hadolint/hadolint < Dockerfile
```

Common hadolint findings seen in this repo

- DL3015 (info): "Avoid additional packages by specifying `--no-install-recommends`" — For Debian/Ubuntu based images, add `--no-install-recommends` to `apt-get install` to avoid pulling in recommended packages. Example:

  ```dockerfile
  RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
      && rm -rf /var/lib/apt/lists/*
  ```

- DL4006 (warning): "Set the SHELL option -o pipefail before RUN with a pipe in it." — If a RUN command uses a pipe, ensure the shell is configured or add `SHELL ["/bin/bash", "-o", "pipefail", "-c"]` when appropriate for your base image.

- DL3059 (info): "Multiple consecutive `RUN` instructions. Consider consolidation." — Consolidate consecutive RUN instructions to reduce image layers and improve caching. Combine package installs and cleanup into a single RUN where possible.

Suggested next steps for hadolint issues

1. Run `bash scripts/run_hadolint.sh` locally to generate the report and inspect `.hadolint_last_exit`.
2. Address the low-effort fixes (DL3015 and simple RUN consolidations) in small PRs modifying the Dockerfiles.
3. For DL4006, consider whether the base image uses `sh`/`dash` vs `bash` and only change SHELL when safe. Document any deliberate exceptions in a short comment above the Dockerfile sections.

Please confirm you want me to proceed with the quick unblock now (I will skip: hadolint, python-safety-dependencies-check, pydocstyle). If you want shellcheck also skipped, say so. Otherwise I’ll proceed with the three listed skips.
