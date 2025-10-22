"""GPU diagnostics for VideoAnnotator.

Checks CUDA availability, device information, and memory.

v1.3.0: Phase 11 - T071
"""

from typing import Any

from videoannotator.utils.logging_config import get_logger

logger = get_logger("diagnostics")


def diagnose_gpu() -> dict[str, Any]:
    """Run comprehensive GPU diagnostics.

    Returns:
        Dictionary with GPU diagnostic information:
        {
            "status": "ok" | "warning" | "error",
            "cuda_available": true,
            "device_count": 1,
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "compute_capability": "8.6",
                    "total_memory_gb": 24.0,
                    "memory_allocated_gb": 2.5,
                    "memory_reserved_gb": 3.0
                }
            ],
            "cuda_version": "11.8",
            "pytorch_version": "2.0.1",
            "errors": [],
            "warnings": []
        }
    """
    result: dict[str, Any] = {
        "status": "ok",
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "cuda_version": None,
        "pytorch_version": None,
        "errors": [],
        "warnings": [],
    }

    try:
        # Try to import PyTorch
        try:
            import torch
        except ImportError:
            result["status"] = "warning"
            result["warnings"].append(
                "PyTorch not installed - GPU features unavailable"
            )
            return result

        result["pytorch_version"] = torch.__version__

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        result["cuda_available"] = cuda_available

        if not cuda_available:
            result["status"] = "warning"
            result["warnings"].append("CUDA not available - GPU acceleration disabled")
            return result

        # Get CUDA version
        try:
            result["cuda_version"] = torch.version.cuda
        except AttributeError:
            result["cuda_version"] = "unknown"

        # Get device information
        device_count = torch.cuda.device_count()
        result["device_count"] = device_count

        if device_count == 0:
            result["status"] = "warning"
            result["warnings"].append("No CUDA devices found")
            return result

        # Get details for each device
        devices = []
        for i in range(device_count):
            try:
                device_info = _get_device_info(i)
                devices.append(device_info)
            except Exception as e:
                logger.warning(f"Failed to get info for device {i}: {e}")
                result["warnings"].append(f"Device {i}: {e!s}")

        result["devices"] = devices

        # Check for warnings
        if result["warnings"] and result["status"] == "ok":
            result["status"] = "warning"

    except Exception as e:
        result["status"] = "error"
        result["errors"].append(f"GPU diagnostic failed: {e!s}")
        logger.error(f"GPU diagnostic error: {e}", exc_info=True)

    return result


def _get_device_info(device_index: int) -> dict[str, Any]:
    """Get detailed information for a specific GPU device.

    Args:
        device_index: Index of the device to check

    Returns:
        Dictionary with device information
    """
    import torch

    props = torch.cuda.get_device_properties(device_index)

    # Get memory information
    memory_total = props.total_memory
    memory_reserved = torch.cuda.memory_reserved(device_index)
    memory_allocated = torch.cuda.memory_allocated(device_index)

    return {
        "index": device_index,
        "name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_gb": round(memory_total / 1024**3, 2),
        "memory_allocated_gb": round(memory_allocated / 1024**3, 2),
        "memory_reserved_gb": round(memory_reserved / 1024**3, 2),
        "memory_free_gb": round((memory_total - memory_reserved) / 1024**3, 2),
        "multiprocessor_count": props.multi_processor_count,
    }
