"""ONNX Runtime adapter base class.

ONNX is the preferred deployment format for third-party models: the graph and
the weights travel together, so no architecture has to be reconstructed, and
the same file runs on CPU and CUDA.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ai.base import BaseOphthalmologyModel
from app.ai.results import Availability
from app.ai.weights import check_weights, resolve_weight_path, runtime_availability
from app.core.enums import DeviceType
from app.core.logging import get_logger

logger = get_logger(__name__)


class OnnxModelAdapter(BaseOphthalmologyModel):
    """Runs an ONNX graph declared by ``metadata.weights``."""

    def availability(self) -> Availability:
        base = super().availability()
        if not base.available:
            return base
        runtime = runtime_availability(needs_onnx=True)
        if not runtime.available:
            return runtime
        if self.metadata.weights is None:
            return Availability(
                available=False,
                reason="Adapter declares no ONNX graph.",
                remediation="Set metadata.weights to the .onnx file this model expects.",
            )
        return check_weights(self.metadata.weights)

    def _providers(self) -> list[str]:
        """Execution providers, GPU first when available."""
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        providers: list[str] = []
        if self._device_manager.profile.device is DeviceType.CUDA:
            for candidate in ("TensorrtExecutionProvider", "CUDAExecutionProvider"):
                if candidate in available:
                    providers.append(candidate)
        providers.append("CPUExecutionProvider")
        return providers

    def _load(self) -> None:
        import onnxruntime as ort

        assert self.metadata.weights is not None  # guarded by availability()
        path = resolve_weight_path(self.metadata.weights)
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session: Any = ort.InferenceSession(
            str(path), sess_options=options, providers=self._providers()
        )
        self._input_name = self._session.get_inputs()[0].name
        logger.info(
            "onnx_session_created",
            extra={"model_id": self.model_id, "providers": self._session.get_providers()},
        )

    def unload(self) -> None:
        self._session = None
        super().unload()

    def predict(self, prepared: np.ndarray) -> np.ndarray:
        """Run the graph on a single preprocessed image (CHW ``float32``)."""
        batch = np.ascontiguousarray(prepared[None, ...], dtype=np.float32)
        outputs = self._session.run(None, {self._input_name: batch})
        return np.asarray(outputs[0])
