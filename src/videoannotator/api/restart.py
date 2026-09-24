"""Server boot identity and self-restart (specs/011-pipeline-readiness FR-007/FR-008).

A restart is a graceful uvicorn shutdown followed by a re-launch of the same
`videoannotator server ...` command, not an in-place reload: shutting down
first releases the port and runs the app's lifespan shutdown (background job
manager, DB), and only then does the CLI start a fresh process. That's why
self-restart needs the CLI to own the `uvicorn.Server` (`enable_execv`), and
is unsupported under `--reload`, `--workers > 1`, or any other launcher.

Restart modes (reported as `restart_mode` in the health responses):

- `execv`: started by `videoannotator server` with one worker, no reload.
- `exit_for_supervisor`: opted in with VIDEOANNOTATOR_RESTART_MODE for a
  deployment whose supervisor restarts an exited process (e.g. a container
  with `restart: unless-stopped`). The process just exits cleanly.
- `unsupported`: everything else; the endpoint refuses with a manual hint.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from typing import Any

LOGGER = logging.getLogger("videoannotator.api.restart")

RESTART_MODE_ENV = "VIDEOANNOTATOR_RESTART_MODE"

BOOT_ID = uuid.uuid4().hex[:16]
STARTED_AT = datetime.now(UTC).isoformat().replace("+00:00", "Z")

_server: Any = None  # uvicorn.Server, when the CLI registered one
_restart_requested = False


def boot_identity() -> dict[str, str]:
    """Fields every health response carries so a client can tell a restart
    has completed: `boot_id` changes once per process start."""
    return {
        "boot_id": BOOT_ID,
        "started_at": STARTED_AT,
        "restart_mode": restart_mode(),
    }


def enable_execv(server: Any) -> None:
    """Called by `videoannotator server` before running `server`, when it is
    safe to restart by shutting it down and re-launching (single worker, no
    reload)."""
    global _server
    _server = server


def restart_mode() -> str:
    if _server is not None:
        return "execv"
    if os.environ.get(RESTART_MODE_ENV) == "exit_for_supervisor":
        return "exit_for_supervisor"
    return "unsupported"


def unsupported_hint() -> str:
    return (
        "This server can't restart itself (it was started with --reload, "
        "--workers > 1, or not via `videoannotator server`). Restart it the way "
        "it was started, e.g. stop it and run `videoannotator server` again."
    )


def request_restart() -> None:
    """Begin a graceful shutdown; the CLI re-launches afterwards (execv mode)
    or the supervisor does (exit_for_supervisor). Call after the response
    has been sent."""
    global _restart_requested
    mode = restart_mode()
    LOGGER.warning("Restart requested (mode=%s, boot_id=%s)", mode, BOOT_ID)
    if mode == "execv":
        _restart_requested = True
        _server.should_exit = True
    elif mode == "exit_for_supervisor":
        os.kill(os.getpid(), signal.SIGTERM)


def restart_requested() -> bool:
    return _restart_requested


def relaunch_command() -> list[str]:
    """The same `videoannotator ...` invocation, via the module rather than
    the console-script path (which differs across platforms and installers)."""
    return [sys.executable, "-m", "videoannotator.cli", *sys.argv[1:]]


def relaunch() -> None:
    """Replace this process with a fresh server. Only call once uvicorn has
    fully shut down (port released).

    POSIX `execv` keeps the pid, so a parent waiting on it (`uv run`, a shell,
    a container runtime) sees one continuous process. Windows has no real
    exec (its `os.execv` also mangles arguments containing spaces), so there
    a child is started and this process exits.
    """
    command = relaunch_command()
    LOGGER.warning("Re-launching server: %s", " ".join(command))
    if os.name == "nt":
        subprocess.Popen(command)
        return
    os.execv(sys.executable, command)
