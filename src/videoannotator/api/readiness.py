"""Pipeline readiness: where each pipeline stands and the one next step
(specs/011-pipeline-readiness, contracts/readiness-contract.md §1-2).

`available` (004) only says the extras packages are on disk. Readiness also
knows about installs in flight, installs that need a restart, setup the
pipeline still needs (a secret in the server's environment, a reachable
Ollama), imports that failed, and model weights still to download. It is
derived per request and never stored.
"""

from __future__ import annotations

import importlib.metadata
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement

from ..registry import pipeline_loader
from ..registry.pipeline_loader import extras_available
from ..registry.pipeline_registry import PipelineMetadata, WeightSpec
from . import extras_install

# Hand-maintained, approximate download sizes (MB) for each extras group's own
# packages, excluding torch (added separately below, and only when torch isn't
# installed yet, since groups share it). Labelled "approx." wherever shown.
_EXTRA_OWN_MB: dict[str, int] = {
    "face": 650,  # deepface + tensorflow/tf-keras + opencv
    "face-laion": 750,  # deepface/tensorflow + transformers + torchvision
    "face-openface3": 120,
    "audio": 300,  # openai-whisper, librosa, pyannote.*, torchaudio
    "audio-laion": 200,  # transformers, librosa
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


def _hf_cached(repo_id: str) -> bool:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache = Path(HF_HUB_CACHE)
    except ImportError:
        cache = (
            Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
            / "hub"
        )
    snapshots = cache / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    return snapshots.is_dir() and any(snapshots.iterdir())


def _whisper_cached(model: str) -> bool:
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "whisper"
    return (root / f"{model}.pt").is_file()


def _weights_cached(weight: WeightSpec) -> bool:
    if weight.cache == "whisper":
        return _whisper_cached(weight.id)
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
                        "message": (
                            f"Ollama isn't reachable at {status.get('base_url')}. "
                            "Start it with 'ollama serve'."
                        ),
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
        elif req.kind == "licence":
            notes.append(
                {
                    "kind": "licence",
                    "name": req.name,
                    "message": req.description
                    or "Accept this model's licence on Hugging Face.",
                    "help_url": req.help_url,
                    "approx_mb": None,
                }
            )
        # Unknown kinds are ignored here rather than blocking a pipeline.

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


def _ollama_base_url(meta: PipelineMetadata) -> str:
    field = meta.config_schema.get("base_url")
    return str(field.default) if field and field.default else "http://127.0.0.1:11434"


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
