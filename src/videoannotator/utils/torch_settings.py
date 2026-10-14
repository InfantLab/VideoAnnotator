"""Process-wide torch settings, per pipeline.

torch keeps some settings per process, not per model, so a library that
changes them in one pipeline changes every pipeline after it in the same job.
OpenFace 3's landmark code turns on `cudnn.benchmark`, which picks convolution
algorithms by timing them, so results changed with what else ran first and
from run to run (found 2026-10-04: `scene_detection` gave different CLIP
scores alone and after OpenFace). Each pipeline now starts from the same
settings and can't leave its own behind.

`deterministic=True` also asks cuDNN and torch for deterministic algorithms,
for runs that must reproduce exactly. Slower; ops with no deterministic
version only warn.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

# cuBLAS needs this for deterministic matrix products on CUDA >= 10.2; it is
# read when cuBLAS starts, so set it before a model first uses the GPU.
_CUBLAS_WORKSPACE = ":4096:8"


def _torch() -> Any:
    try:
        import torch
    except ImportError:  # core install: no pipeline uses torch
        return None
    return torch


def current_torch_settings() -> dict[str, Any]:
    """The settings that affect results, as recorded with a pipeline run."""
    torch = _torch()
    if torch is None:
        return {}
    return {
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "anomaly_detection": torch.is_anomaly_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def apply_torch_settings(deterministic: bool) -> dict[str, Any]:
    """Set the settings a pipeline runs with; return them."""
    torch = _torch()
    if torch is None:
        return {}
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.autograd.set_detect_anomaly(False)
    torch.set_flush_denormal(False)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE)
    return current_torch_settings()


@contextlib.contextmanager
def restored_torch_settings() -> Iterator[None]:
    """Put the settings back as they were when the block exits."""
    torch = _torch()
    if torch is None:
        yield
        return
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    algorithms = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    anomaly = torch.is_anomaly_enabled()
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.use_deterministic_algorithms(algorithms, warn_only=warn_only)
        torch.autograd.set_detect_anomaly(anomaly)
        # No getter exists; off is torch's default.
        torch.set_flush_denormal(False)
