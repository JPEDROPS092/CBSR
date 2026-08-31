"""Device and precision abstraction.

Keeps every model adapter free of ``torch.cuda`` checks: adapters ask the
:class:`DeviceManager` what to run on, and the same code path works on a
laptop CPU, a GPU server or an ONNX Runtime deployment.
"""

from __future__ import annotations

import functools
import os
import platform
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.enums import DeviceType, Precision
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DeviceProfile:
    """Resolved execution target."""

    device: DeviceType
    name: str
    index: int = 0
    total_memory_mb: float | None = None

    @property
    def torch_device(self) -> str:
        """Device string accepted by ``torch.device``."""
        return "cpu" if self.device is DeviceType.CPU else f"cuda:{self.index}"


def torch_available() -> bool:
    """Whether PyTorch is importable in this process."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def onnxruntime_available() -> bool:
    """Whether ONNX Runtime is importable in this process."""
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def cuda_available() -> bool:
    """Whether a usable CUDA device is present."""
    if not torch_available():
        return False
    import torch

    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - broken driver must not crash the API
        logger.warning("cuda_probe_failed", exc_info=True)
        return False


class DeviceManager:
    """Chooses the execution device and validates precision requests."""

    def __init__(self, preference: str | None = None, precision: str | None = None) -> None:
        settings = get_settings()
        self.preference = (preference or settings.DEVICE).lower()
        self.requested_precision = Precision(precision or settings.INFERENCE_PRECISION)

    @functools.cached_property
    def profile(self) -> DeviceProfile:
        """The device this process will use."""
        if self.preference == "cpu" or not cuda_available():
            if self.preference == "cuda":
                logger.warning("cuda_requested_but_unavailable", extra={"fallback": "cpu"})
            return DeviceProfile(
                device=DeviceType.CPU,
                name=platform.processor() or "cpu",
                total_memory_mb=None,
            )
        import torch

        index = int(os.getenv("CUDA_VISIBLE_DEVICE_INDEX", "0"))
        props = torch.cuda.get_device_properties(index)
        return DeviceProfile(
            device=DeviceType.CUDA,
            name=props.name,
            index=index,
            total_memory_mb=props.total_memory / (1024 * 1024),
        )

    def resolve_precision(self) -> Precision:
        """Downgrade unsupported precision requests instead of failing.

        Half precision on CPU is slower than fp32 in PyTorch and bf16 needs
        Ampere or newer, so an impossible request falls back to fp32 with a
        warning rather than producing a hard error at inference time.
        """
        precision = self.requested_precision
        if precision is Precision.FP32:
            return precision
        if self.profile.device is DeviceType.CPU:
            logger.warning("precision_downgraded", extra={"from": str(precision), "to": "fp32"})
            return Precision.FP32
        if precision is Precision.BF16 and torch_available():
            import torch

            if not torch.cuda.is_bf16_supported():
                logger.warning("bf16_unsupported", extra={"to": "fp16"})
                return Precision.FP16
        return precision

    def memory_used_mb(self) -> float | None:
        """Currently allocated VRAM, when running on CUDA."""
        if self.profile.device is not DeviceType.CUDA or not torch_available():
            return None
        import torch

        return torch.cuda.memory_allocated(self.profile.index) / (1024 * 1024)

    def empty_cache(self) -> None:
        """Release cached VRAM between analyses on shared GPU workers."""
        if self.profile.device is DeviceType.CUDA and torch_available():
            import torch

            torch.cuda.empty_cache()

    def describe(self) -> dict[str, object]:
        """Serializable snapshot for ``/health`` and logs."""
        return {
            "device": str(self.profile.device),
            "device_name": self.profile.name,
            "total_memory_mb": self.profile.total_memory_mb,
            "precision": str(self.resolve_precision()),
            "torch_available": torch_available(),
            "cuda_available": cuda_available(),
            "onnxruntime_available": onnxruntime_available(),
        }


@functools.lru_cache
def get_device_manager() -> DeviceManager:
    """Process-wide device manager."""
    return DeviceManager()
