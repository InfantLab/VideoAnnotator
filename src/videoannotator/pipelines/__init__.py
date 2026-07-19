"""VideoAnnotator Pipeline Modules.

This package contains modular pipeline implementations for video
annotation tasks.
"""

from .base_pipeline import BasePipeline

__all__ = [
    "AudioPipeline",
    "BasePipeline",
    "FaceAnalysisPipeline",
    "LAIONFacePipeline",
    "PersonTrackingPipeline",
    "SceneDetectionPipeline",
]

# Each pipeline family needs a different extras group (`face`, `face-laion`,
# `audio`, `person`, `scene` — see 004-extras-based-install); importing them
# eagerly here would mean loading *any* single pipeline via the registry
# forces every family's heavy deps to be installed, defeating extras
# isolation. PEP 562 lazy attribute access keeps `from videoannotator.pipelines
# import X` working for whichever families are actually installed, without
# paying that cost at package-import time.
_LAZY_ATTRS = {
    "AudioPipeline": ".audio_processing",
    "FaceAnalysisPipeline": ".face_analysis.face_pipeline",
    "LAIONFacePipeline": ".face_analysis.laion_face_pipeline",
    "PersonTrackingPipeline": ".person_tracking.person_pipeline",
    "SceneDetectionPipeline": ".scene_detection.scene_pipeline",
}


def __getattr__(name: str) -> object:
    module_path = _LAZY_ATTRS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_path, __name__)
    return getattr(module, name)
