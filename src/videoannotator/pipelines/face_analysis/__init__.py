"""Face Analysis Pipeline.

This module provides face detection, recognition, and emotion analysis
capabilities.
"""

__all__ = [
    "FaceAnalysisPipeline",
    "OpenFace3Pipeline",
]

# Each variant needs a different extras group (`face`, `face-openface3` — see
# 004-extras-based-install): FaceAnalysisPipeline needs deepface,
# OpenFace3Pipeline needs openface-test but *not* deepface. Importing any of them eagerly
# here would force every variant's deps onto whichever one you actually asked
# for, so resolve lazily instead.
_LAZY_ATTRS = {
    "FaceAnalysisPipeline": ".face_pipeline",
    "OpenFace3Pipeline": ".openface3_pipeline",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)
