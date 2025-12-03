"""VideoAnnotator Pipeline Registry.

Central registry for pipeline metadata, discovery, and validation.
"""

from .pipeline_loader import PipelineLoader, get_pipeline_loader
from .pipeline_registry import PipelineRegistry

__all__ = [
    "PipelineLoader",
    "PipelineRegistry",
    "get_pipeline_loader",
    "pipeline_registry",
]

# Global registry instance
pipeline_registry = PipelineRegistry()
