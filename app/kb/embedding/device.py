"""Embedding device resolution and GPU diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


class EmbeddingDeviceError(RuntimeError):
    pass


@dataclass(frozen=True)
class DeviceInfo:
    requested_device: str
    resolved_device: str
    cuda_available: bool
    gpu_name: str | None = None
    gpu_total_memory_mb: float | None = None
    gpu_free_memory_mb: float | None = None

    def to_dict(self) -> dict:
        return {
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "cuda_available": self.cuda_available,
            "gpu_name": self.gpu_name,
            "gpu_total_memory_mb": self.gpu_total_memory_mb,
            "gpu_free_memory_mb": self.gpu_free_memory_mb,
        }


def cuda_is_available() -> bool:
    return torch.cuda.is_available()


def resolve_embedding_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "cuda":
        if not cuda_is_available():
            raise EmbeddingDeviceError(
                "EMBEDDING_DEVICE=cuda was requested but CUDA is not available on this machine"
            )
        return torch.device("cuda")
    if normalized == "cpu":
        return torch.device("cpu")
    raise EmbeddingDeviceError(f"Unsupported EMBEDDING_DEVICE: {requested!r}. Use 'cpu' or 'cuda'.")


def get_device_info(requested: str) -> DeviceInfo:
    cuda_available = cuda_is_available()
    if requested.strip().lower() == "cuda":
        resolved = "cuda" if cuda_available else "unavailable"
    elif requested.strip().lower() == "cpu":
        resolved = "cpu"
    else:
        resolved = "unsupported"

    gpu_name = None
    total_mb = None
    free_mb = None
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            total_mb = round(total_bytes / (1024 * 1024), 2)
            free_mb = round(free_bytes / (1024 * 1024), 2)
        except Exception:
            props = torch.cuda.get_device_properties(0)
            total_mb = round(props.total_memory / (1024 * 1024), 2)

    return DeviceInfo(
        requested_device=requested,
        resolved_device=resolved,
        cuda_available=cuda_available,
        gpu_name=gpu_name,
        gpu_total_memory_mb=total_mb,
        gpu_free_memory_mb=free_mb,
    )


def clear_cuda_cache() -> None:
    if cuda_is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def peak_gpu_memory_mb() -> float | None:
    if not cuda_is_available():
        return None
    try:
        return round(torch.cuda.max_memory_allocated(0) / (1024 * 1024), 2)
    except Exception:
        return None


def reset_peak_gpu_memory() -> None:
    if cuda_is_available():
        torch.cuda.reset_peak_memory_stats(0)


def is_cuda_oom_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    return "out of memory" in message or "cuda out of memory" in message
