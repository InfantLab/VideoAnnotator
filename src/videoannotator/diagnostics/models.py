"""Models directory diagnostics (spec 016): where weights live and their size."""

from typing import Any

from videoannotator.models_dir import (
    SOURCES,
    directory_size,
    legacy_locations,
    models_dir,
    source_dir,
)


def diagnose_models() -> dict[str, Any]:
    """Report the models directory, its size per source, and old copies elsewhere.

    Returns:
        {"status", "models_dir", "total_bytes", "sources": {name: bytes},
         "legacy_locations": [...], "errors", "warnings"}
    """
    root = models_dir()
    sources = {name: directory_size(source_dir(name)) for name in SOURCES}
    legacy = [str(path) for path in legacy_locations()]
    result: dict[str, Any] = {
        "status": "ok",
        "models_dir": str(root),
        "total_bytes": directory_size(root),
        "sources": sources,
        "legacy_locations": legacy,
        "errors": [],
        "warnings": [],
    }
    if legacy:
        result["status"] = "warning"
        result["warnings"].append(
            "Model weights from before v1.6.0 are outside the models directory "
            f"({', '.join(legacy)}). Move them into {root} to avoid downloading "
            "again, or delete them to free space."
        )
    return result
