"""Throwaway stub pipeline for the forward-compatibility test
(specs/004-extras-based-install User Story 3 / SC-007).

Not a real pipeline — it exists only to prove that a future non-local
pipeline (e.g. v1.6.0's Ollama-backed `llm` group, or v1.7.0+'s
HTTP/Slurm-dispatched pipelines) can declare `requires_extras: []` plus a
non-ML `backends` value and load through the registry with zero changes to
PipelineMetadata or the loader.
"""

from typing import Any

from videoannotator.pipelines.base_pipeline import BasePipeline


class StubPipeline(BasePipeline):
    """Stub pipeline — every method raises NotImplementedError."""

    def initialize(self) -> None:
        raise NotImplementedError("stub_pipeline is a forward-compatibility fixture")

    def process(
        self,
        video_path: str,
        start_time: float = 0.0,
        end_time: float | None = None,
        pps: float = 0.0,
        output_dir: str | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError("stub_pipeline is a forward-compatibility fixture")

    def cleanup(self) -> None:
        raise NotImplementedError("stub_pipeline is a forward-compatibility fixture")

    def get_schema(self) -> dict[str, Any]:
        raise NotImplementedError("stub_pipeline is a forward-compatibility fixture")
