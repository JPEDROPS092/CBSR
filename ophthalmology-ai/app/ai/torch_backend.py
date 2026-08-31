"""PyTorch adapter base class.

Torch is an optional dependency: the API, the classical models and the test
suite all run without it. Adapters built on this class report themselves as
*unavailable* (with remediation) rather than crashing when torch or their
checkpoint is missing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ai.base import BaseOphthalmologyModel
from app.ai.results import Availability
from app.ai.weights import check_weights, resolve_weight_path, runtime_availability
from app.core.enums import Precision
from app.core.exceptions import ModelUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class TorchModelAdapter(BaseOphthalmologyModel):
    """Runs a PyTorch checkpoint declared by ``metadata.weights``.

    Two checkpoint shapes are accepted:

    * a **TorchScript archive or serialized module** - loaded directly, which
      is the recommended way to deploy a third-party model because the
      architecture travels with the weights;
    * a **state dict** - requires the adapter to override :meth:`build_module`
      with the exact architecture the checkpoint was trained with.

    The adapter never guesses an architecture: an unmatched state dict raises
    :class:`ModelUnavailableError` explaining what to supply.
    """

    def availability(self) -> Availability:
        base = super().availability()
        if not base.available:
            return base
        runtime = runtime_availability(needs_torch=True)
        if not runtime.available:
            return runtime
        if self.metadata.weights is None:
            return Availability(
                available=False,
                reason="Adapter declares no checkpoint.",
                remediation="Set metadata.weights to the checkpoint this model expects.",
            )
        return check_weights(self.metadata.weights)

    def build_module(self) -> Any:
        """Construct the architecture for a state-dict checkpoint.

        Override in adapters whose upstream distributes a bare ``state_dict``.
        """
        raise ModelUnavailableError(
            "This checkpoint is a state_dict, but the adapter does not define an architecture.",
            details={
                "model_id": self.model_id,
                "remediation": (
                    "Provide a TorchScript export of the model "
                    "(torch.jit.script/trace then torch.jit.save), or override build_module() "
                    "with the exact architecture the weights were trained with."
                ),
            },
        )

    def _load(self) -> None:
        import torch

        assert self.metadata.weights is not None  # guarded by availability()
        path = resolve_weight_path(self.metadata.weights)
        device = self._device_manager.profile.torch_device

        try:
            module = torch.jit.load(str(path), map_location=device)
            logger.info("torchscript_loaded", extra={"model_id": self.model_id})
        except Exception:  # noqa: BLE001 - not a TorchScript archive, try a pickle
            checkpoint = torch.load(str(path), map_location=device, weights_only=False)
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("state_dict", checkpoint)
                module = self.build_module()
                missing, unexpected = module.load_state_dict(state_dict, strict=False)
                if missing or unexpected:
                    logger.warning(
                        "state_dict_partial_match",
                        extra={
                            "model_id": self.model_id,
                            "missing_keys": len(missing),
                            "unexpected_keys": len(unexpected),
                        },
                    )
                    if len(missing) > len(state_dict) // 2:
                        raise ModelUnavailableError(
                            "Checkpoint does not match the adapter's architecture.",
                            details={
                                "model_id": self.model_id,
                                "missing_keys": len(missing),
                                "unexpected_keys": len(unexpected),
                            },
                        ) from None
            else:
                module = checkpoint

        module.eval()
        module.to(device)
        precision = self._device_manager.resolve_precision()
        if precision is Precision.FP16:
            module.half()
        elif precision is Precision.BF16:
            module.to(torch.bfloat16)
        self._module = module

    def unload(self) -> None:
        self._module = None
        super().unload()

    def predict(self, prepared: np.ndarray) -> np.ndarray:
        """Forward pass over a single preprocessed image (CHW ``float32``)."""
        import torch

        device = self._device_manager.profile.torch_device
        precision = self._device_manager.resolve_precision()
        tensor = torch.from_numpy(np.ascontiguousarray(prepared)).unsqueeze(0).to(device)
        if precision is Precision.FP16:
            tensor = tensor.half()
        elif precision is Precision.BF16:
            tensor = tensor.to(torch.bfloat16)

        with torch.inference_mode():
            output = self._module(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        return output.detach().float().cpu().numpy()
