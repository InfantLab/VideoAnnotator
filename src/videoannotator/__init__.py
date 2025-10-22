"""VideoAnnotator - Modern Video Annotation Pipeline.

A comprehensive toolkit for video analysis including scene detection,
person tracking, face analysis, and audio processing.
"""

import sys
import warnings
from typing import Any

from .version import (
    __author__,
    __license__,
    __version__,
    __version_info__,
    create_annotation_metadata,
    get_version_info,
    print_version_info,
)

# Heavy pipeline imports commented out to prevent pytest collection hangs
# Uncomment when needed for production use
# from .pipelines import (
#     BasePipeline,
#     SceneDetectionPipeline,
#     PersonTrackingPipeline,
#     FaceAnalysisPipeline,
#     AudioPipeline,
# )

# Note: After standards migration, these schemas are no longer used
# Pipelines now return native format dictionaries (COCO, WebVTT, RTTM, etc.)
# from .schemas import (
#     AnnotationBase,
#     VideoMetadata,
#     SceneSegment,
#     SceneAnnotation,
#     PersonDetection,
#     FaceDetection,
#     SpeechSegment,
# )

__all__ = [
    # Version information (explicitly imported above)
    "__version__",
    "__version_info__",
    "__author__",
    "__license__",
    "get_version_info",
    "print_version_info",
    "create_annotation_metadata",
    # Note: Public modules (api, pipelines, registry, storage, database, worker,
    # validation, exporters, utils, config, cli, main) are dynamically loaded
    # via __getattr__ and available via `from videoannotator import <module>`
]


def __getattr__(name: str) -> Any:
    """Intercept module-level attribute access for namespace forwarding.

    This allows imports like `from videoannotator.api import ...` to work
    by dynamically importing from the correct location.

    Also provides deprecation warnings for legacy `src.*` imports.

    Args:
        name: The attribute/module name being accessed

    Returns:
        The requested module or attribute

    Raises:
        AttributeError: If the attribute/module doesn't exist
    """
    # Map of public module names to their submodule names
    _PUBLIC_MODULES = {
        "api": "api",
        "pipelines": "pipelines",
        "registry": "registry",
        "storage": "storage",
        "database": "database",
        "worker": "worker",
        "validation": "validation",
        "exporters": "exporters",
        "utils": "utils",
        "config": "config",
        "cli": "cli",
        "main": "main",
    }

    if name in _PUBLIC_MODULES:
        submodule_name = _PUBLIC_MODULES[name]

        # Import the submodule using relative import
        try:
            # Use importlib for cleaner relative imports
            import importlib

            module = importlib.import_module(f".{submodule_name}", package=__name__)

            # Cache the module in this namespace for subsequent access
            setattr(sys.modules[__name__], name, module)

            return module
        except ImportError as e:
            raise AttributeError(
                f"Module 'videoannotator' has no attribute '{name}'. "
                f"Failed to import '.{submodule_name}': {e}"
            ) from e

    # If not a known module, raise AttributeError
    raise AttributeError(f"Module 'videoannotator' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Provide custom dir() output for the module.

    Returns:
        List of public names available in the module
    """
    return list(__all__) + list(_PUBLIC_MODULES.keys())


# Preserve the _PUBLIC_MODULES dict for __getattr__ and __dir__
_PUBLIC_MODULES = {
    "api": "api",
    "pipelines": "pipelines",
    "registry": "registry",
    "storage": "storage",
    "database": "database",
    "worker": "worker",
    "validation": "validation",
    "exporters": "exporters",
    "utils": "utils",
    "config": "config",
    "cli": "cli",
    "main": "main",
}
