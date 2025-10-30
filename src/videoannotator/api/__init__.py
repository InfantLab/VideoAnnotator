"""VideoAnnotator API package."""

from ..version import __version__ as _videoannotator_version

__version__ = _videoannotator_version


# Lazy imports for commonly used API components
def __getattr__(name: str):
    """Lazy load API submodules."""
    if name == "main":
        from . import main as _main

        return _main
    elif name == "database":
        from . import database as _database

        return _database
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = ["__version__"]
