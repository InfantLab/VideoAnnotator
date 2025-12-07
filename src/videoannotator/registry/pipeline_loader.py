"""Dynamic pipeline class loader using registry metadata.

This module provides centralized pipeline class loading to eliminate
hardcoded pipeline mappings in JobProcessor and BatchOrchestrator.
"""

import importlib
import logging

from .pipeline_registry import PipelineMetadata, get_registry

LOGGER = logging.getLogger("videoannotator.registry")


class PipelineLoader:
    """Load pipeline classes dynamically from registry metadata."""

    def __init__(self):
        """Initialize the pipeline loader."""
        self._class_cache: dict[str, type] = {}
        self._registry = get_registry()

    def load_all_pipelines(self) -> dict[str, type]:
        """Load all available pipeline classes from registry.

        Returns:
            Dictionary mapping pipeline names to their classes.
            Includes both primary names and any aliases.
        """
        self._registry.load()
        pipeline_classes = {}

        for meta in self._registry.list():
            pipeline_class = self._load_pipeline_class(meta)
            if pipeline_class:
                # Add primary name
                pipeline_classes[meta.name] = pipeline_class

                # Add common aliases based on pipeline_family
                if meta.pipeline_family:
                    # Short family name (e.g., 'face' for 'face_analysis')
                    if meta.pipeline_family not in pipeline_classes:
                        pipeline_classes[meta.pipeline_family] = pipeline_class

                    # Add variant-based alias if applicable
                    if meta.variant:
                        variant_name = f"{meta.pipeline_family}_{meta.variant}"
                        if variant_name != meta.name:
                            pipeline_classes[variant_name] = pipeline_class

        LOGGER.info(
            f"Loaded {len(set(pipeline_classes.values()))} unique pipeline classes "
            f"with {len(pipeline_classes)} total names/aliases"
        )

        return pipeline_classes

    def _load_pipeline_class(self, meta: PipelineMetadata) -> type | None:
        """Load a single pipeline class from metadata.

        Args:
            meta: Pipeline metadata containing module path

        Returns:
            Pipeline class or None if loading fails
        """
        # Check cache first
        if meta.name in self._class_cache:
            return self._class_cache[meta.name]

        # Get module path from metadata
        module_path = self._infer_module_path(meta)
        if not module_path:
            LOGGER.warning(
                f"Cannot load pipeline '{meta.name}': no module_path in metadata"
            )
            return None

        try:
            # Parse module_path format: "module.path:ClassName"
            if ":" not in module_path:
                LOGGER.error(
                    f"Invalid module_path format for '{meta.name}': {module_path}. "
                    "Expected 'module.path:ClassName'"
                )
                return None

            module_name, class_name = module_path.split(":", 1)

            # Dynamically import the module
            module = importlib.import_module(module_name)
            pipeline_class = getattr(module, class_name)

            # Cache the result
            self._class_cache[meta.name] = pipeline_class

            LOGGER.debug(f"Loaded pipeline class: {meta.name} -> {pipeline_class}")
            return pipeline_class

        except ImportError as e:
            LOGGER.warning(
                f"Failed to import pipeline '{meta.name}' from {module_path}: {e}"
            )
            return None
        except AttributeError as e:
            LOGGER.error(f"Pipeline class not found in module for '{meta.name}': {e}")
            return None
        except Exception as e:
            LOGGER.error(
                f"Unexpected error loading pipeline '{meta.name}': {e}",
                exc_info=True,
            )
            return None

    def _infer_module_path(self, meta: PipelineMetadata) -> str | None:
        """Infer module path from metadata if not explicitly provided.

        This provides backward compatibility for metadata files that don't
        yet have the module_path field.

        Args:
            meta: Pipeline metadata

        Returns:
            Inferred module path or None
        """
        # Mapping of pipeline names to their module paths (temporary fallback)
        LEGACY_MAPPINGS = {
            "scene_detection": "videoannotator.pipelines.scene_detection:SceneDetectionPipeline",
            "person_tracking": "videoannotator.pipelines.person_tracking:PersonTrackingPipeline",
            "face_analysis": "videoannotator.pipelines.face_analysis:FaceAnalysisPipeline",
            "audio_processing": "videoannotator.pipelines.audio_processing:AudioPipeline",
            "speech_recognition": "videoannotator.pipelines.audio_processing:SpeechPipeline",
            "speaker_diarization": "videoannotator.pipelines.audio_processing:DiarizationPipeline",
            "face_laion_clip": "videoannotator.pipelines.face_analysis.laion_face_pipeline:LAIONFacePipeline",
            "laion_voice": "videoannotator.pipelines.audio_processing.laion_voice_pipeline:LAIONVoicePipeline",
            "face_openface3_embedding": "videoannotator.pipelines.face_analysis.openface3_pipeline:OpenFace3Pipeline",
        }

        return LEGACY_MAPPINGS.get(meta.name)


# Singleton instance
_loader_instance: PipelineLoader | None = None


def get_pipeline_loader() -> PipelineLoader:
    """Get the shared PipelineLoader singleton instance."""
    global _loader_instance
    if _loader_instance is None:
        _loader_instance = PipelineLoader()
    return _loader_instance


__all__ = ["PipelineLoader", "get_pipeline_loader"]
