"""In-app extras-group install: command selection, execution, and state.

Backs the self-service install endpoints in `api/v1/pipelines.py`
(specs/005-pipeline-extras-install). Deliberately a background thread
spawned at request time rather than a polled queue -- installs are rare,
human-triggered, one-at-a-time-per-extras-group actions, not the
many-concurrent-video-jobs case `api/background_tasks.py` was built for
(research.md §2).
"""

import importlib
import importlib.metadata
import logging
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from ..database import database as _db_module
from ..database.models import ExtrasInstallJob, ExtrasInstallJobStatus
from ..registry import pipeline_loader

LOGGER = logging.getLogger("videoannotator.api.extras_install")

_DISTRIBUTION_NAME = "videoannotator"

# --- Dedup tracking (FR-010): at most one in-flight install per extra_name ---
_lock = threading.Lock()
_in_flight: dict[str, str] = {}  # extra_name -> job_id

# --- Restart-required signal (in-process only, see research.md §4) ---
_restart_required = False

# --- Activation outcome per completed install (spec 011 FR-005/FR-006) ---
# In-process like the flag above: after a restart every install is active, so
# an outcome recorded by a previous process has nothing left to say.
_activation: dict[str, dict[str, Any]] = {}  # job_id -> outcome


def restart_required() -> bool:
    """Whether any install completed since this process started needs a
    restart to take effect (activation `restart_required`)."""
    return _restart_required


def activation_for(job_id: str) -> dict[str, Any] | None:
    """`{"activation", "conflicting_distributions"}` for a job completed in
    this process, else None."""
    return _activation.get(job_id)


def _mark_restart_required() -> None:
    global _restart_required
    _restart_required = True


def in_flight_job_ids() -> list[str]:
    """Job ids of installs currently pending/running (spec 011: a restart
    must not interrupt one)."""
    with _lock:
        return list(_in_flight.values())


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


def extra_requirements(extra_name: str) -> list[str]:
    """The requirement strings the `extra_name` extras group adds, read from
    this package's installed metadata. A group that includes other groups
    (`all = ["videoannotator[face,audio,...]"]`) is expanded into theirs."""
    try:
        dist = importlib.metadata.distribution(_DISTRIBUTION_NAME)
    except importlib.metadata.PackageNotFoundError:
        return []

    requirements: list[str] = []
    seen_groups: set[str] = set()

    def collect(group: str) -> None:
        if group in seen_groups:
            return
        seen_groups.add(group)
        for req_str in dist.requires or []:
            req = Requirement(req_str)
            # Evaluated against this interpreter, so platform-specific
            # requirements are already resolved; the marker can then go.
            if not (req.marker and req.marker.evaluate({"extra": group})):
                continue
            if canonicalize_name(req.name) == _DISTRIBUTION_NAME:
                for sub_group in sorted(req.extras):
                    collect(sub_group)
                continue
            req.marker = None
            if str(req) not in requirements:
                requirements.append(str(req))

    collect(extra_name)
    return requirements


def resolve_install_command(extra_name: str) -> tuple[list[str], Path | None]:
    """Return `(command, cwd)` to run to install `extra_name`.

    Installs only the group's own requirements into the running interpreter's
    environment (spec 011 FR-005a): `uv pip install --python <sys.executable>`
    when uv is on PATH, else `python -m pip install`. Both leave already
    satisfied packages alone, so an install normally only *adds*
    distributions and can activate without a restart.

    This replaced `uv sync --extra <name> --inexact`, which re-synced the
    whole environment to `uv.lock` while the server was running: it rewrote
    packages the server had already imported (on Windows, failing outright on
    a loaded `.pyd`), targeted `<root>/.venv` rather than the running
    environment, and reinstalled videoannotator itself (the locked
    `videoannotator.exe` on Windows).

    `[tool.uv.sources]` isn't consulted this way. The only source there is
    the cu124 torch index on Linux, and PyPI's Linux torch 2.6.0 wheels are
    already cu124 builds.

    Falls back to the version-pinned `videoannotator[<extra>]==<version>` if
    the group's requirements can't be read from metadata. `cwd` is always
    None; kept in the signature for callers.
    """
    requirements = extra_requirements(extra_name)
    if not requirements:
        version = importlib.metadata.version(_DISTRIBUTION_NAME)
        requirements = [f"{_DISTRIBUTION_NAME}[{extra_name}]=={version}"]

    if shutil.which("uv"):
        return (
            ["uv", "pip", "install", "--python", sys.executable, *requirements],
            None,
        )
    return ([sys.executable, "-m", "pip", "install", *requirements], None)


def _installed_distributions() -> tuple[dict[str, str], dict[str, set[str]]]:
    """Snapshot `{distribution: version}` and `{distribution: top-level
    modules}` for the running environment."""
    importlib.invalidate_caches()
    versions = {}
    for dist in importlib.metadata.distributions():
        name = dist.name
        if name:
            versions[canonicalize_name(name)] = dist.version
    modules: dict[str, set[str]] = {}
    for module, dist_names in importlib.metadata.packages_distributions().items():
        for dist_name in dist_names:
            modules.setdefault(canonicalize_name(dist_name), set()).add(module)
    return versions, modules


def decide_activation(
    before: dict[str, str],
    after: dict[str, str],
    modules: dict[str, set[str]],
    loaded: set[str] | None = None,
) -> dict[str, Any]:
    """FR-005: `restart_required` iff a distribution whose version changed
    (or that was removed) has a top-level module already imported in this
    process; otherwise `live`. Newly added distributions can't be loaded
    yet, so they never force a restart."""
    loaded = set(sys.modules) if loaded is None else loaded
    conflicts = []
    for name, old_version in sorted(before.items()):
        new_version = after.get(name)
        if new_version == old_version:
            continue
        # A module name matching the distribution name covers packages whose
        # metadata lists no top-level modules.
        dist_modules = modules.get(name, set()) | {name.replace("-", "_")}
        if dist_modules & loaded:
            conflicts.append(
                {"name": name, "old_version": old_version, "new_version": new_version}
            )
    return {
        "activation": "restart_required" if conflicts else "live",
        "conflicting_distributions": conflicts,
    }


def _activate_live() -> None:
    """Make packages installed while running visible to this process: fresh
    import finders, and availability recomputed rather than memoised."""
    importlib.invalidate_caches()
    pipeline_loader._is_distribution_installed.cache_clear()
    pipeline_loader._packages_for_extra.cache_clear()


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
        before, modules_before = _installed_distributions()
        try:
            result = subprocess.run(
                command,
                cwd=cwd,
                capture_output=True,
                text=True,
                check=False,
            )
            output = (result.stdout or "") + (result.stderr or "")
            job.command_output = output
            status = (
                ExtrasInstallJobStatus.COMPLETED
                if result.returncode == 0
                else ExtrasInstallJobStatus.FAILED
            )
        except OSError as exc:
            job.command_output = f"Failed to launch install command: {exc}"
            status = ExtrasInstallJobStatus.FAILED

        # Settle activation before the job reads as `completed`: a client
        # re-fetches the pipeline list as soon as it sees `completed`, and
        # must get the post-install availability, not the memoised one.
        if status == ExtrasInstallJobStatus.COMPLETED:
            after, modules_after = _installed_distributions()
            # Removed distributions only appear in the "before" module map.
            modules = {**modules_after, **modules_before}
            outcome = decide_activation(before, after, modules)
            if outcome["activation"] == "restart_required":
                LOGGER.warning(
                    "Install of %r changed already-imported distributions, restart "
                    "required: %s",
                    extra_name,
                    outcome["conflicting_distributions"],
                )
                _mark_restart_required()
            else:
                _activate_live()
                LOGGER.info("Install of %r activated without a restart", extra_name)
            _activation[job_id] = outcome

        job.status = status
        job.finished_at = datetime.now()
        db.commit()
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
