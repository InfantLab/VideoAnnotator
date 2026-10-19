"""A stand-in for OllamaVLMClient on both sides of Ask AI: the pipeline a run
uses and the API's tries. Calls are recorded in one shared log, in order."""

from __future__ import annotations

import threading
from typing import Any, ClassVar

from videoannotator.pipelines.vlm_annotation.ollama_client import VLMCallResult


class FakeVLM:
    models: ClassVar[list[str]] = ["qwen3.5:9b"]
    log: ClassVar[list[str]] = []
    run_calls: ClassVar[int] = 0
    # The run's call number to hold until `release` is set; `held` says it is.
    hold_run_call: ClassVar[int | None] = None
    held: ClassVar[threading.Event] = threading.Event()
    release: ClassVar[threading.Event] = threading.Event()

    def __init__(self, base_url: str, timeout: int, role: str = "run"):
        self.role = role

    @classmethod
    def reset(cls) -> None:
        cls.log = []
        cls.run_calls = 0
        cls.hold_run_call = None
        cls.held = threading.Event()
        cls.release = threading.Event()

    def list_models(self) -> list[str]:
        return list(self.models)

    def preflight(self, model: str) -> None:
        pass

    def model_details(self, model: str) -> dict[str, Any]:
        return {"digest": "d" * 64, "quantization": "Q4_K_M"}

    def chat(self, *, images: list[bytes], **kwargs: Any) -> VLMCallResult:
        if self.role == "run":
            FakeVLM.run_calls += 1
            n = FakeVLM.run_calls
            FakeVLM.log.append(f"run-{n}")
            if FakeVLM.hold_run_call == n:
                FakeVLM.held.set()
                FakeVLM.release.wait(10)
            return VLMCallResult(
                "TOUCH" if n % 2 else "NO_TOUCH", "", 0.2, 0, 1, 1, 1.0
            )
        FakeVLM.log.append("try")
        return VLMCallResult("TOUCH", "", 0.1, 0, 1, 1, 1.0)


class FakeTryVLM(FakeVLM):
    def __init__(self, base_url: str, timeout: int):
        super().__init__(base_url, timeout, role="try")
