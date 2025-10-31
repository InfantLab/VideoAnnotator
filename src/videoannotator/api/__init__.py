"""VideoAnnotator API package."""

import importlib

from ..version import __version__ as _videoannotator_version

__version__ = _videoannotator_version


# Lazy imports for commonly used API components
def __getattr__(name: str):
    """Lazy load API submodules."""
    if name == "main":
        return importlib.import_module(".main", __name__)
    elif name == "database":
        return importlib.import_module(".database", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["__version__"]
