"""Pipeline readiness: where each pipeline stands and the one next step
(specs/011-pipeline-readiness, contracts/readiness-contract.md §1-2).

`available` (004) only says the extras packages are on disk. Readiness also
knows about installs in flight, installs that need a restart, setup the
pipeline still needs (a secret in the server's environment, a reachable
Ollama), imports that failed, and model weights still to download. It is
derived per request and never stored.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

from ..config_env import default_ollama_base_url, huggingface_token
from ..models_dir import source_dir
from ..registry import pipeline_loader
from ..registry.pipeline_loader import extras_available
from ..registry.pipeline_registry import (
    PipelineMetadata,
    SetupRequirement,
    WeightSpec,
)
from . import extras_install

# Hand-maintained, approximate download sizes (MB) for each extras group's own
# packages, excluding torch (added separately below, and only when torch isn't
# installed yet, since groups share it). Labelled "approx." wherever shown.
_EXTRA_OWN_MB: dict[str, int] = {
    "face": 650,  # deepface + tensorflow/tf-keras + opencv
    "face-openface3": 120,
    "audio": 300,  # openai-whisper, librosa, pyannote.*, torchaudio
    "scene": 120,  # open-clip, scenedetect, opencv
    "person": 200,  # ultralytics, supervision, torchvision, opencv
    "llm": 1,  # the ollama client; models are pulled into Ollama separately
}
# Linux resolves torch to the CUDA build (with its NVIDIA libraries);
# macOS/Windows get the CPU build.
_TORCH_MB = 3000 if sys.platform.startswith("linux") else 250

_NEXT_ACTION = {
    "installing": "wait",
    "not_installed": "install",
    "restart_required": "restart",
    "needs_setup": "setup",
    "ready": "none",
}

# Ollama reachability, cached so listing pipelines stays fast (FR-012, SC-006).
_OLLAMA_TTL_S = 30.0
# An unreachable result expires sooner, so starting Ollama is noticed quickly.
_OLLAMA_UNREACHABLE_TTL_S = 5.0
_OLLAMA_TIMEOUT_S = 1
_ollama_cache: dict[str, tuple[float, dict[str, Any]]] = {}
_ollama_lock = threading.Lock()
_ollama_refreshing: set[str] = set()

# Whether the configured Hugging Face token works and its account has accepted
# each gated model's licence. Cached like Ollama; a failing result expires
# sooner so accepting a licence or fixing the token is noticed within a minute.
_HF_OK_TTL_S = 600.0
_HF_RETRY_TTL_S = 60.0
_HF_TIMEOUT_S = 5
_hf_cache: dict[tuple[str, tuple[str, ...]], tuple[float, dict[str, Any]]] = {}
_hf_lock = threading.Lock()
_hf_refreshing: set[tuple[str, tuple[str, ...]]] = set()


def _extra_requires_torch(extra: str) -> bool:
    try:
        dist = importlib.metadata.distribution("videoannotator")
    except importlib.metadata.PackageNotFoundError:
        return False
    for req_str in dist.requires or []:
        req = Requirement(req_str)
        if req.name == "torch" and req.marker and req.marker.evaluate({"extra": extra}):
            return True
    return False


def _torch_installed() -> bool:
    try:
        importlib.metadata.distribution("torch")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def approx_download_mb(extra: str) -> int | None:
    """Approximate download for installing `extra` now, or None if unknown."""
    own = _EXTRA_OWN_MB.get(extra)
    if own is None:
        return None
    if _extra_requires_torch(extra) and not _torch_installed():
        own += _TORCH_MB
    return own


def _check_ollama(base_url: str) -> dict[str, Any]:
    from ..diagnostics.ollama import diagnose_ollama

    status = diagnose_ollama(base_url, timeout=_OLLAMA_TIMEOUT_S)
    with _ollama_lock:
        _ollama_cache[base_url] = (time.monotonic(), status)
        _ollama_refreshing.discard(base_url)
    return status


def _ollama_status(base_url: str) -> dict[str, Any]:
    """Cached reachability. Only the very first check blocks (up to the
    timeout; on Windows even a refused localhost connection takes ~1 s). After
    that, a stale result is returned while a background thread refreshes it,
    so listing pipelines never waits on Ollama (SC-006)."""
    with _ollama_lock:
        cached = _ollama_cache.get(base_url)
        ttl = (
            _OLLAMA_TTL_S
            if cached and cached[1].get("ollama_reachable")
            else _OLLAMA_UNREACHABLE_TTL_S
        )
        stale = cached is None or time.monotonic() - cached[0] >= ttl
        start_refresh = (
            stale and cached is not None and base_url not in _ollama_refreshing
        )
        if start_refresh:
            _ollama_refreshing.add(base_url)
    if cached is None:
        return _check_ollama(base_url)
    if start_refresh:
        threading.Thread(
            target=_check_ollama, args=(base_url,), name="ollama-readiness", daemon=True
        ).start()
    return cached[1]


def _hf_get(path: str, token: str) -> tuple[int | None, bytes]:
    """GET a Hub API path; `(status, body)`, status None if unreachable.
    Plain urllib, not huggingface_hub: that's an `audio` dependency, and its
    `auth_check` has no timeout."""
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    request = urllib.request.Request(
        endpoint + path, headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_HF_TIMEOUT_S) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, b""
    except (urllib.error.URLError, OSError, ValueError):
        return None, b""


def _check_hf_access(token: str, repos: tuple[str, ...]) -> dict[str, Any]:
    """`token`: ok | invalid | unknown; `repos[r]`: ok | not_accepted |
    unknown; `user`: the token's account name, when known."""
    import json

    status: dict[str, Any] = {
        "token": "unknown",
        "user": None,
        "repos": dict.fromkeys(repos, "unknown"),
    }
    code, body = _hf_get("/api/whoami-v2", token)
    if code == 401:
        status["token"] = "invalid"
    elif code == 200:
        status["token"] = "ok"
        try:
            status["user"] = json.loads(body).get("name")
        except (ValueError, AttributeError):
            pass
        for repo in repos:
            code, _ = _hf_get(f"/api/models/{repo}/auth-check", token)
            if code == 200:
                status["repos"][repo] = "ok"
            elif code in (401, 403):
                status["repos"][repo] = "not_accepted"
    key = (hashlib.sha256(token.encode()).hexdigest(), repos)
    with _hf_lock:
        _hf_cache[key] = (time.monotonic(), status)
        _hf_refreshing.discard(key)
    return status


def _hf_access(token: str, repos: tuple[str, ...]) -> dict[str, Any]:
    """Cached `_check_hf_access`, refreshed in the background once stale
    (same scheme as `_ollama_status`)."""
    key = (hashlib.sha256(token.encode()).hexdigest(), repos)
    with _hf_lock:
        cached = _hf_cache.get(key)
        all_ok = cached is not None and (
            cached[1]["token"] == "ok"
            and all(v == "ok" for v in cached[1]["repos"].values())
        )
        ttl = _HF_OK_TTL_S if all_ok else _HF_RETRY_TTL_S
        stale = cached is None or time.monotonic() - cached[0] >= ttl
        start_refresh = stale and cached is not None and key not in _hf_refreshing
        if start_refresh:
            _hf_refreshing.add(key)
    if cached is None:
        return _check_hf_access(token, repos)
    if start_refresh:
        threading.Thread(
            target=_check_hf_access,
            args=(token, repos),
            name="hf-readiness",
            daemon=True,
        ).start()
    return cached[1]


def _licence_blockers_and_notes(
    licences: list[SetupRequirement],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check gated-model licences against the configured token. Models
    already in the local cache were evidently accepted, so they're skipped
    without touching the network."""
    pending = [r for r in licences if not _hf_cached(r.name)]
    token = huggingface_token()
    if not pending or not token:
        # No token: the `secret` blocker already says so; keep the licence
        # reminders so they can be done at the same time.
        return [], [_licence_item(r, "unchecked") for r in pending]

    access = _hf_access(token, tuple(r.name for r in pending))
    if access["token"] == "invalid":
        return [
            {
                "kind": "secret",
                "name": "HUGGINGFACE_TOKEN",
                "message": (
                    "HUGGINGFACE_TOKEN is set, but Hugging Face rejects it (expired, "
                    "revoked or mis-pasted). Create a new token, put it in the "
                    "server's environment (e.g. .env) and restart the server."
                ),
                "help_url": "https://huggingface.co/settings/tokens",
            }
        ], []
    blockers, notes = [], []
    for req in pending:
        state = access["repos"].get(req.name, "unknown")
        if state == "not_accepted":
            blockers.append(_licence_item(req, state, access["user"]))
        elif state != "ok":
            notes.append(_licence_item(req, state))
    return blockers, notes


def _licence_item(
    req: SetupRequirement, state: str, user: str | None = None
) -> dict[str, Any]:
    account = f"Hugging Face account {user!r}" if user else "Hugging Face account"
    if state == "not_accepted":
        message = (
            f"The {account} that owns HUGGINGFACE_TOKEN hasn't accepted the "
            f"{req.name} licence. Open the model page signed in as that account, "
            "accept it, then check again."
        )
    elif state == "unknown":
        message = (
            f"Couldn't reach Hugging Face to check the {req.name} licence. Make "
            "sure the account that owns HUGGINGFACE_TOKEN has accepted it."
        )
    else:
        message = req.description or "Accept this model's licence on Hugging Face."
    item: dict[str, Any] = {
        "kind": "licence",
        "name": req.name,
        "message": message,
        "help_url": req.help_url,
    }
    if state != "not_accepted":
        item["approx_mb"] = None
    return item


def _hf_cached(repo_id: str) -> bool:
    """In the Hugging Face hub cache, or pyannote 3.x's own cache
    (`from_pretrained` defaults to `$PYANNOTE_CACHE`, else
    `~/.cache/torch/pyannote`, not the hub cache)."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        hub = Path(HF_HUB_CACHE)
    except ImportError:
        hub = (
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            / "hub"
        )
    pyannote = Path(
        os.environ.get("PYANNOTE_CACHE", Path.home() / ".cache" / "torch" / "pyannote")
    )
    folder = f"models--{repo_id.replace('/', '--')}"
    for cache in (hub, pyannote):
        snapshots = cache / folder / "snapshots"
        if snapshots.is_dir() and any(snapshots.iterdir()):
            return True
    return False


def _whisper_cached(model: str) -> bool:
    """The models directory (the speech pipelines' `cache_dir` default), or
    openai-whisper's own default cache."""
    roots = (
        source_dir("whisper"),
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper",
    )
    return any((root / f"{model}.pt").is_file() for root in roots)


def _deepface_cached(filename: str) -> bool:
    home = Path(os.environ.get("DEEPFACE_HOME", Path.home()))
    return (home / ".deepface" / "weights" / filename).is_file()


def _weights_cached(weight: WeightSpec) -> bool:
    if weight.cache == "whisper":
        return _whisper_cached(weight.id)
    if weight.cache == "deepface":
        return _deepface_cached(weight.id)
    return _hf_cached(weight.id)


def _secret_is_set(name: str, aliases: list[str]) -> bool:
    return any(os.environ.get(n) for n in [name, *aliases])


def _extras_group(meta: PipelineMetadata) -> str | None:
    return meta.requires_extras[0] if meta.requires_extras else None


def _blockers_and_notes(
    meta: PipelineMetadata,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    blockers: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []

    import_error = pipeline_loader.import_error_for(meta.name)
    if import_error:
        blockers.append(
            {
                "kind": "import_error",
                "name": meta.name,
                "message": f"The pipeline failed to load: {import_error}",
                "help_url": None,
            }
        )

    for req in meta.requires_setup:
        if req.kind == "secret":
            if not _secret_is_set(req.name, req.aliases):
                what = req.description or "A secret this pipeline needs"
                blockers.append(
                    {
                        "kind": "secret",
                        "name": req.name,
                        "message": (
                            f"{what} isn't set. Set {req.name} in the server's "
                            "environment (e.g. the container env or .env) and "
                            "restart the server."
                        ),
                        "help_url": req.help_url,
                    }
                )
        elif req.kind == "service" and req.name == "ollama":
            status = _ollama_status(_ollama_base_url(meta))
            if not status.get("ollama_reachable"):
                blockers.append(
                    {
                        "kind": "service",
                        "name": "ollama",
                        "message": _ollama_unreachable_message(status),
                        "help_url": req.help_url,
                    }
                )
            elif not status.get("models"):
                blockers.append(
                    {
                        "kind": "service",
                        "name": "ollama",
                        "message": (
                            f"Ollama is running at {status.get('base_url')} but has "
                            "no models pulled. Run 'ollama pull <model>'."
                        ),
                        "help_url": req.help_url,
                    }
                )
        # `licence` is handled below, all of a pipeline's licences at once.
        # Unknown kinds are ignored here rather than blocking a pipeline.

    licence_blockers, licence_notes = _licence_blockers_and_notes(
        [r for r in meta.requires_setup if r.kind == "licence"]
    )
    blockers.extend(licence_blockers)
    notes.extend(licence_notes)

    for weight in meta.weights:
        if not _weights_cached(weight):
            notes.append(
                {
                    "kind": "weights_not_cached",
                    "name": weight.id,
                    "message": (
                        f"The first run downloads about {weight.approx_mb} MB of "
                        f"model weights ({weight.id})."
                    ),
                    "help_url": None,
                    "approx_mb": weight.approx_mb,
                }
            )

    return blockers, notes


def pipeline_readiness(meta: PipelineMetadata) -> dict[str, Any]:
    """The `readiness` object for one pipeline (contract §1). States are
    evaluated in the contract's order; the first match wins."""
    group = _extras_group(meta)
    install_job_id = None
    for extra in meta.requires_extras:
        install_job_id = install_job_id or extras_install.in_flight_job_for(extra)

    blockers: list[dict[str, Any]] = []
    notes: list[dict[str, Any]] = []
    if install_job_id:
        state = "installing"
    elif not extras_available(meta.requires_extras):
        state = "not_installed"
    elif any(extras_install.restart_pending(e) for e in meta.requires_extras):
        state = "restart_required"
    else:
        blockers, notes = _blockers_and_notes(meta)
        state = "needs_setup" if blockers else "ready"

    return {
        "state": state,
        "next_action": _NEXT_ACTION[state],
        "extras_group": group,
        "install_job_id": install_job_id,
        "blockers": blockers,
        "notes": notes,
    }


def _ollama_unreachable_message(status: dict[str, Any]) -> str:
    from ..diagnostics.ollama import unreachable_hint

    base_url = str(status.get("base_url"))
    return f"Ollama isn't reachable at {base_url}. {unreachable_hint(base_url)}"


def _ollama_base_url(meta: PipelineMetadata) -> str:
    field = meta.config_schema.get("base_url")
    return str(field.default) if field and field.default else default_ollama_base_url()


def warm_up() -> None:
    """Fill the service-check cache off the request path (called at startup)."""
    from ..registry.pipeline_registry import get_registry

    try:
        reg = get_registry()
        reg.load()
        for meta in reg.list():
            if any(
                r.kind == "service" and r.name == "ollama" for r in meta.requires_setup
            ):
                _ollama_status(_ollama_base_url(meta))
            if extras_available(meta.requires_extras):
                _licence_blockers_and_notes(
                    [r for r in meta.requires_setup if r.kind == "licence"]
                )
    except Exception:  # best-effort; a listing will check again
        pass


def extras_groups(pipelines: list[PipelineMetadata]) -> list[dict[str, Any]]:
    """Every extras group that enables at least one pipeline (contract §2), in
    `pyproject.toml` order. Meta-groups like `all`/`dev` enable none directly
    and are left out."""
    try:
        dist = importlib.metadata.distribution("videoannotator")
        declared = dist.metadata.get_all("Provides-Extra") or []
    except importlib.metadata.PackageNotFoundError:
        declared = []

    groups = []
    for extra in declared:
        enabled = [m.name for m in pipelines if extra in m.requires_extras]
        if not enabled:
            continue
        groups.append(
            {
                "name": extra,
                "pipelines": enabled,
                "installed": extras_available([extra]),
                "approx_download_mb": approx_download_mb(extra),
                "includes_gpu_torch": sys.platform.startswith("linux")
                and _extra_requires_torch(extra),
                "install_job_id": extras_install.in_flight_job_for(extra),
            }
        )
    return groups
