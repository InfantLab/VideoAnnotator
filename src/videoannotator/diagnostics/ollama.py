"""Ollama reachability diagnostics for VideoAnnotator (spec 009).

Closes an already-open item from roadmap_v1.7.0.md: `videoannotator diagnose`
reports GPU/storage/database health but had no way to say whether the
vlm_annotation pipeline's local VLM backend is actually usable right now.
"""

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from videoannotator.config_env import (
    DEFAULT_OLLAMA_BASE_URL,
    OLLAMA_BASE_URL_ENV,
    default_ollama_base_url,
)
from videoannotator.utils.logging_config import get_logger

__all__ = ["DEFAULT_OLLAMA_BASE_URL", "diagnose_ollama", "unreachable_hint"]

logger = get_logger("diagnostics")


def _in_container() -> bool:
    return Path("/.dockerenv").exists() or Path("/run/.containerenv").exists()


def unreachable_hint(base_url: str) -> str:
    """What to do about an unreachable `base_url`. Inside a container a
    loopback URL is the container itself, so an Ollama running on the host
    is never reachable that way -- the usual cause when it "is running"."""
    host = urlparse(base_url).hostname or ""
    if _in_container() and host in ("127.0.0.1", "localhost", "::1"):
        return (
            "VideoAnnotator is running in a container, where this address is "
            "the container itself. If Ollama runs on the host, set "
            f"{OLLAMA_BASE_URL_ENV}=http://host.docker.internal:11434 for the "
            "server and restart it (and start Ollama on the host with "
            "OLLAMA_HOST=0.0.0.0:11434)."
        )
    return "Start it with 'ollama serve'."


def diagnose_ollama(base_url: str | None = None, timeout: int = 5) -> dict[str, Any]:
    """Check whether the configured Ollama server is reachable and, if so,
    which models are pulled.

    Never raises or hangs regardless of server state (spec 009 US3
    acceptance scenario 2) -- uses the same OllamaVLMClient.list_models()
    the preview/model-list API endpoints use, so this reports the same
    reachability a researcher would actually experience configuring
    vlm_annotation, not a separate check with different behavior.

    Returns:
        {
            "status": "ok" | "warning",
            "ollama_reachable": bool,
            "base_url": str,
            "models": list[str],
            "errors": [],
            "warnings": [],
        }
    """
    base_url = base_url or default_ollama_base_url()
    result: dict[str, Any] = {
        "status": "ok",
        "ollama_reachable": False,
        "base_url": base_url,
        "models": [],
        "errors": [],
        "warnings": [],
    }

    from videoannotator.pipelines.vlm_annotation.ollama_client import (
        OllamaUnavailableError,
        OllamaVLMClient,
    )

    try:
        client = OllamaVLMClient(base_url=base_url, timeout=timeout)
        models = client.list_models()
    except OllamaUnavailableError as e:
        result["status"] = "warning"
        result["warnings"].append(str(e))
        return result
    except Exception as e:
        # Any other failure (e.g. the `ollama` package itself missing) is
        # still "not usable right now", not a diagnose crash.
        result["status"] = "warning"
        result["warnings"].append(f"Could not check Ollama reachability: {e}")
        logger.warning(f"Ollama diagnostic error: {e}", exc_info=True)
        return result

    result["ollama_reachable"] = True
    result["models"] = models
    if not models:
        result["status"] = "warning"
        result["warnings"].append(
            f"Ollama is reachable at {base_url} but has no models pulled "
            "-- run 'ollama pull <model>' before using vlm_annotation."
        )
    return result
