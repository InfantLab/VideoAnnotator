"""In-app extras-group install: command selection, execution, and state.

Backs the self-service install endpoints in `api/v1/pipelines.py`
(specs/005-pipeline-extras-install). Deliberately a background thread
spawned at request time rather than a polled queue -- installs are rare,
human-triggered, one-at-a-time-per-extras-group actions, not the
many-concurrent-video-jobs case `api/background_tasks.py` was built for
(research.md §2).
"""

import importlib.metadata
import logging
import os
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path

import videoannotator as _videoannotator_pkg

from ..database import database as _db_module
from ..database.models import ExtrasInstallJob, ExtrasInstallJobStatus

LOGGER = logging.getLogger("videoannotator.api.extras_install")

_DISTRIBUTION_NAME = "videoannotator"

# --- Dedup tracking (FR-010): at most one in-flight install per extra_name ---
_lock = threading.Lock()
_in_flight: dict[str, str] = {}  # extra_name -> job_id

# --- Restart-required signal (in-process only, see research.md §4) ---
_restart_required = False


def restart_required() -> bool:
    """Whether any install has completed successfully since this process
    started, and the process has not been restarted since."""
    return _restart_required


def _mark_restart_required() -> None:
    global _restart_required
    _restart_required = True


def try_begin_install(extra_name: str, job_id: str) -> str | None:
    """Register `job_id` as the in-flight install for `extra_name`.

    Returns the *existing* in-flight job_id if one is already
    pending/running for this extra_name (caller should return that job
    instead of starting a new one) -- otherwise registers `job_id` and
    returns None.
    """
    with _lock:
        existing = _in_flight.get(extra_name)
        if existing is not None:
            return existing
        _in_flight[extra_name] = job_id
        return None


def _end_install(extra_name: str) -> None:
    with _lock:
        _in_flight.pop(extra_name, None)


def _editable_checkout_root() -> Path | None:
    """Return this repo's project root if running from an editable/source
    checkout of videoannotator itself, else None.

    Detected by walking up from the installed package's own `__file__` to
    find a sibling `pyproject.toml` declaring `name = "videoannotator"` --
    the signature of `src/videoannotator/__init__.py` living under a
    checked-out project root rather than site-packages (research.md §1).
    """
    pkg_file = Path(_videoannotator_pkg.__file__).resolve()
    # src/videoannotator/__init__.py -> src/videoannotator -> src -> root
    candidate_root = pkg_file.parent.parent.parent
    pyproject = candidate_root / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    if 'name = "videoannotator"' not in text:
        return None
    return candidate_root


def resolve_install_command(extra_name: str) -> tuple[list[str], Path | None]:
    """Return `(command, cwd)` to run to install `extra_name`.

    Prefers `uv sync --extra <name> --inexact` in this repo's own project
    root when running from an editable/source checkout with `uv` on PATH
    (mirrors `scripts/start_server.sh`'s own install mechanism); otherwise
    falls back to a version-pinned `pip install videoannotator[<extra>]==
    <running version>`, which works regardless of how the package was
    installed and never silently upgrades the running videoannotator
    release (research.md §1). `cwd` is None for the pip path (no specific
    directory required).

    `--inexact` is required, not cosmetic: a bare `uv sync --extra X`
    resolves "core + X" and *removes* anything installed outside that
    closure -- including a different extras group installed by an earlier
    call to this same function (e.g. installing `audio` would silently
    uninstall a previously-installed `face`). Discovered via a real end-to-
    end run during this feature's own implementation, not a hypothetical.

    `--no-install-project` is required too: videoannotator itself is already
    installed (it is the running server), and without the flag uv rebuilds and
    reinstalls it, which on Windows means overwriting the locked
    `Scripts/videoannotator.exe` the server was started from, failing every
    install with os error 32.
    """
    root = _editable_checkout_root()
    if root is not None and shutil.which("uv"):
        return (
            [
                "uv",
                "sync",
                "--extra",
                extra_name,
                "--inexact",
                "--no-install-project",
            ],
            root,
        )

    version = importlib.metadata.version(_DISTRIBUTION_NAME)
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        f"{_DISTRIBUTION_NAME}[{extra_name}]=={version}",
    ]
    return (command, None)


def install_env() -> dict[str, str]:
    """Environment for the install subprocess.

    `uv sync` ignores which interpreter is running and installs into
    `<project root>/.venv` unless told otherwise. A server started from any
    other environment (a second venv, a Windows venv beside a WSL `.venv`)
    would "install" successfully into the wrong one and the pipeline would
    never become available. Pointing UV_PROJECT_ENVIRONMENT at `sys.prefix`
    makes uv target the environment actually serving requests; the pip
    fallback already does, via `sys.executable`, and ignores the variable.
    """
    return {**os.environ, "UV_PROJECT_ENVIRONMENT": sys.prefix}


def run_install(job_id: str, extra_name: str) -> None:
    """Run the install for `extra_name` and update the `ExtrasInstallJob`
    row identified by `job_id` as it progresses.

    Intended to run on a background `threading.Thread`, not the request
    thread that triggered it (FR-004 -- the endpoint must not block on
    this). Opens its own DB session since it does not run within a request.

    Looks up `SessionLocal` on the `database` module fresh at call time
    (rather than importing the name once at module load) so it stays
    correct if a test fixture swaps in a different engine/session factory
    after this module was first imported.
    """
    db = _db_module.SessionLocal()
    try:
        job = db.query(ExtrasInstallJob).filter(ExtrasInstallJob.id == job_id).first()
        if job is None:
            LOGGER.error("run_install: job %s vanished before it could start", job_id)
            return

        job.status = ExtrasInstallJobStatus.RUNNING
        job.started_at = datetime.now()
        db.commit()

        command, cwd = resolve_install_command(extra_name)
        LOGGER.info("Installing extras group %r: %s", extra_name, " ".join(command))
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                env=install_env(),
                capture_output=True,
                text=True,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")
            job.command_output = output
            job.status = (
                ExtrasInstallJobStatus.COMPLETED
                if result.returncode == 0
                else ExtrasInstallJobStatus.FAILED
            )
        except OSError as exc:
            job.command_output = f"Failed to launch install command: {exc}"
            job.status = ExtrasInstallJobStatus.FAILED

        job.finished_at = datetime.now()
        db.commit()

        if job.status == ExtrasInstallJobStatus.COMPLETED:
            _mark_restart_required()
    finally:
        db.close()
        _end_install(extra_name)


def start_install(job_id: str, extra_name: str) -> None:
    """Spawn `run_install` on a background thread."""
    thread = threading.Thread(
        target=run_install,
        args=(job_id, extra_name),
        name=f"extras-install-{extra_name}",
        daemon=True,
    )
    thread.start()
