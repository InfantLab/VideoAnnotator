"""Forward-compatibility gate: a stub pipeline with `requires_extras: []`
plus a non-ML `backends` value loads through the registry and the loader
with zero code changes (spec 004 User Story 3 / SC-007, quickstart.md §6).

This is the concrete proof that v1.6.0's Ollama `llm` extras group and
v1.7.0+'s HTTP/Slurm-dispatched pipelines can be added as pure data
(YAML + a `[project.optional-dependencies]` entry) without touching
PipelineMetadata or pipeline_loader.py again.
"""

from pathlib import Path

from videoannotator.registry.pipeline_loader import PipelineLoader
from videoannotator.registry.pipeline_registry import PipelineRegistry

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "fixtures"


class TestForwardCompatStub:
    def test_stub_pipeline_loads_via_registry(self):
        registry = PipelineRegistry(metadata_dir=FIXTURES_DIR)
        registry.load(force=True)

        meta = registry.get("stub_pipeline")

        assert meta is not None
        assert meta.requires_extras == []
        assert meta.backends == ["http"]
        assert meta.module_path == "tests.fixtures.stub_pipeline_module:StubPipeline"

    def test_stub_pipeline_class_loads_via_loader_with_zero_loader_changes(self):
        """requires_extras: [] must be vacuously "available" (research.md
        §2), so the loader resolves and imports the stub class exactly like
        any other pipeline — no special-casing for an empty extras list."""
        registry = PipelineRegistry(metadata_dir=FIXTURES_DIR)
        registry.load(force=True)

        loader = PipelineLoader()
        loader._registry = registry  # point the loader at our stub-only registry

        classes = loader.load_all_pipelines()

        assert "stub_pipeline" in classes
        stub_class = classes["stub_pipeline"]
        assert stub_class.__name__ == "StubPipeline"

        # It's a real (if unimplemented) pipeline: instantiating and calling
        # it raises NotImplementedError, not an import/loading error.
        instance = stub_class()
        try:
            instance.initialize()
            raise AssertionError("expected NotImplementedError")
        except NotImplementedError:
            pass
