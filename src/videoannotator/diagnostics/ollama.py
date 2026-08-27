"""Ollama reachability diagnostics for VideoAnnotator (spec 009).

Closes an already-open item from roadmap_v1.6.0.md: `videoannotator diagnose`
reports GPU/storage/database health but had no way to say whether the
vlm_annotation pipeline's local VLM backend is actually usable right now.
"""

from typing import Any

from videoannotator.utils.logging_config import get_logger

logger = get_logger("diagnostics")

DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def diagnose_ollama(base_url: str = DEFAULT_OLLAMA_BASE_URL) -> dict[str, Any]:
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
        client = OllamaVLMClient(base_url=base_url, timeout=5)
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
