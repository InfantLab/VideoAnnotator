"""VideoAnnotator Pipeline Registry.

Central registry for pipeline metadata, discovery, and validation.
"""

from .pipeline_registry import PipelineRegistry

__all__ = ["PipelineRegistry", "pipeline_registry"]

# Global registry instance
pipeline_registry = PipelineRegistry()
