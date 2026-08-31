"""Explainability utilities.

Two families are supported:

* **Gradient based** - Grad-CAM over a chosen convolutional layer of a PyTorch
  model, for classification adapters.
* **Structural** - mask and probability-map overlays, which are the honest
  explanation for a segmentation model: the mask *is* the evidence.

An explanation shows where a model looked. It is not a justification, and the
report engine labels it accordingly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ai.postprocessing import heatmap_overlay, mask_overlay, normalize_heatmap
from app.ai.preprocessing import ExamImage, encode_png
from app.ai.results import ArtifactPayload
from app.core.logging import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Grad-CAM for PyTorch convolutional classifiers.

    Hooks a target layer, runs a forward/backward pass for the class of
    interest and weights the layer's activation maps by their mean gradient.

    Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
    Networks via Gradient-based Localization", ICCV 2017.
    """

    def __init__(self, module: Any, target_layer: Any) -> None:
        self.module = module
        self.target_layer = target_layer
        self._activations: Any = None
        self._gradients: Any = None
        self._handles: list[Any] = []

    def __enter__(self) -> GradCAM:
        self._handles = [
            self.target_layer.register_forward_hook(self._save_activation),
            self.target_layer.register_full_backward_hook(self._save_gradient),
        ]
        return self

    def __exit__(self, *exc_info: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _save_activation(self, _module: Any, _inputs: Any, output: Any) -> None:
        self._activations = output.detach()

    def _save_gradient(self, _module: Any, _grad_input: Any, grad_output: Any) -> None:
        self._gradients = grad_output[0].detach()

    def __call__(self, input_tensor: Any, class_index: int | None = None) -> np.ndarray:
        """Return a normalized ``(H, W)`` saliency map for ``class_index``."""
        import torch

        self.module.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.module(input_tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if class_index is None:
                class_index = int(torch.argmax(logits, dim=1)[0])
            logits[:, class_index].sum().backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM target layer produced no activations.")

        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self._activations).sum(dim=1, keepdim=True))
        return normalize_heatmap(cam[0, 0].cpu().numpy())


def gradcam_artifact(
    saliency: np.ndarray,
    image: ExamImage,
    *,
    name: str = "gradcam",
    meta: dict[str, Any] | None = None,
) -> ArtifactPayload:
    """Render a saliency map over the source image as a PNG artifact."""
    overlay = heatmap_overlay(image.to_rgb(), saliency)
    return ArtifactPayload(
        kind="gradcam",
        name=name,
        data=encode_png(overlay),
        meta={"method": "grad-cam", **(meta or {})},
    )


def probability_map_artifact(
    probabilities: np.ndarray, image: ExamImage, *, name: str = "probability_map"
) -> ArtifactPayload:
    """Render a per-pixel probability map as a heat overlay."""
    overlay = heatmap_overlay(image.to_rgb(), normalize_heatmap(probabilities))
    return ArtifactPayload(
        kind="probability_map",
        name=name,
        data=encode_png(overlay),
        meta={"method": "per-pixel-probability"},
    )


def mask_overlay_artifact(
    mask: np.ndarray,
    image: ExamImage,
    *,
    name: str = "overlay",
    color: tuple[int, int, int] = (255, 64, 64),
    meta: dict[str, Any] | None = None,
) -> ArtifactPayload:
    """Render a segmentation mask over the source image."""
    overlay = mask_overlay(image.to_rgb(), mask, color=color)
    return ArtifactPayload(
        kind="overlay",
        name=name,
        data=encode_png(overlay),
        meta=meta or {},
    )


def curves_overlay_artifact(
    image: ExamImage,
    curves: dict[str, np.ndarray],
    *,
    name: str = "layers_overlay",
    palette: tuple[tuple[int, int, int], ...] = ((255, 80, 80), (80, 200, 255), (120, 255, 120)),
    thickness: int = 2,
) -> ArtifactPayload:
    """Draw per-column boundary curves (OCT layers) over the B-scan."""
    canvas = image.to_rgb().copy()
    height, width = canvas.shape[:2]
    for index, (label, ys) in enumerate(curves.items()):
        color = np.asarray(palette[index % len(palette)], dtype=np.uint8)
        for x in range(min(width, ys.shape[0])):
            y = ys[x]
            if not np.isfinite(y):
                continue
            top = max(int(y) - thickness // 2, 0)
            bottom = min(int(y) + thickness // 2 + 1, height)
            canvas[top:bottom, x] = color
        logger.debug("curve_drawn", extra={"label": label})
    return ArtifactPayload(
        kind="overlay",
        name=name,
        data=encode_png(canvas),
        meta={"curves": list(curves)},
    )
