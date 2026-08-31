"""Adapters for externally supplied checkpoints.

These four classes cover the common cases - classification or segmentation,
running on PyTorch or ONNX Runtime - and are configured entirely from a
:class:`~app.ai.manifest.ExternalModelSpec`. Installing a new model is
therefore a deployment action (drop weights + manifest), not a code change.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ai.explainability import mask_overlay_artifact, probability_map_artifact
from app.ai.manifest import ExternalModelSpec
from app.ai.onnx_backend import OnnxModelAdapter
from app.ai.postprocessing import (
    label_mask_to_payloads,
    mask_to_payload,
    scores_to_predictions,
    sigmoid,
    softmax,
)
from app.ai.preprocessing import ExamImage
from app.ai.results import ArtifactPayload, ModelResult
from app.ai.torch_backend import TorchModelAdapter
from app.core.enums import Framework, TaskType
from app.core.exceptions import ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)


class _ManifestConfigured:
    """Mixin holding the manifest that configures an adapter."""

    def __init__(self, spec: ExternalModelSpec, **kwargs: Any) -> None:
        self.spec = spec
        self.metadata = spec.to_metadata()
        super().__init__(**kwargs)


class _ClassificationPostprocess:
    """Turns a logit/probability vector into ranked predictions."""

    spec: ExternalModelSpec

    def postprocess(self, output: np.ndarray, image: ExamImage) -> ModelResult:
        """Apply the declared activation and map scores onto declared labels."""
        scores = np.asarray(output, dtype=np.float32).reshape(-1)
        activation = self.spec.output.activation
        if activation == "softmax":
            scores = softmax(scores)
        elif activation == "sigmoid":
            scores = sigmoid(scores)
        labels = self.spec.output.labels
        if not labels:
            raise ValidationError(
                "Manifest declares no output labels.",
                details={"model_id": self.spec.model_id},
            )
        return ModelResult(
            model_id=self.spec.model_id,
            model_version=self.spec.version,
            task=TaskType.CLASSIFICATION,
            predictions=scores_to_predictions(scores, labels),
        )


class _SegmentationPostprocess:
    """Turns a per-pixel output into masks and area measurements."""

    spec: ExternalModelSpec

    def postprocess(self, output: np.ndarray, image: ExamImage) -> ModelResult:
        """Threshold (binary) or argmax (multi-class) the network output."""
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 4:  # (N, C, H, W) -> drop the batch dimension
            array = array[0]
        if array.ndim == 2:
            array = array[None, ...]

        classes = self.spec.output.segmentation_classes or self.spec.output.labels
        activation = self.spec.output.activation
        result = ModelResult(
            model_id=self.spec.model_id,
            model_version=self.spec.version,
            task=TaskType.SEGMENTATION,
        )
        target_size = (image.height, image.width)

        if array.shape[0] == 1:
            probabilities = sigmoid(array[0]) if activation != "none" else array[0]
            mask = probabilities >= self.spec.output.threshold
            label = classes[0] if classes else "foreground"
            result.segmentations.append(
                mask_to_payload(
                    mask,
                    label,
                    pixel_spacing_um=image.pixel_spacing_um,
                    target_size=target_size,
                )
            )
            self._probabilities = probabilities
        else:
            probabilities = softmax(array, axis=0) if activation == "softmax" else sigmoid(array)
            label_map = np.argmax(probabilities, axis=0)
            names = classes or [f"class_{i}" for i in range(array.shape[0])]
            if len(names) != array.shape[0]:
                raise ValidationError(
                    "Manifest declares a different number of segmentation classes than "
                    "the model outputs.",
                    details={
                        "declared": len(names),
                        "produced": int(array.shape[0]),
                        "model_id": self.spec.model_id,
                    },
                )
            result.segmentations.extend(
                label_mask_to_payloads(label_map, names, pixel_spacing_um=image.pixel_spacing_um)
            )
            self._probabilities = 1.0 - probabilities[0]

        result.measurements = {
            seg.label: {"area_px": seg.area_px, "area_ratio": seg.area_ratio, **seg.measurements}
            for seg in result.segmentations
        }
        return result

    def explain(self, image: ExamImage, output: Any) -> list[ArtifactPayload]:
        """For segmentation the mask is the explanation; add readable overlays."""
        artifacts: list[ArtifactPayload] = []
        probabilities = getattr(self, "_probabilities", None)
        if probabilities is not None:
            artifacts.append(probability_map_artifact(probabilities, image))
            artifacts.append(
                mask_overlay_artifact(
                    probabilities >= self.spec.output.threshold, image, name="segmentation_overlay"
                )
            )
        return artifacts


class TorchClassificationModel(_ManifestConfigured, _ClassificationPostprocess, TorchModelAdapter):
    """Manifest-configured PyTorch classifier."""


class OnnxClassificationModel(_ManifestConfigured, _ClassificationPostprocess, OnnxModelAdapter):
    """Manifest-configured ONNX classifier."""


class TorchSegmentationModel(_ManifestConfigured, _SegmentationPostprocess, TorchModelAdapter):
    """Manifest-configured PyTorch segmentation model."""


class OnnxSegmentationModel(_ManifestConfigured, _SegmentationPostprocess, OnnxModelAdapter):
    """Manifest-configured ONNX segmentation model."""


_BY_FRAMEWORK_AND_TASK = {
    (Framework.PYTORCH, TaskType.CLASSIFICATION): TorchClassificationModel,
    (Framework.ONNX, TaskType.CLASSIFICATION): OnnxClassificationModel,
    (Framework.PYTORCH, TaskType.SEGMENTATION): TorchSegmentationModel,
    (Framework.ONNX, TaskType.SEGMENTATION): OnnxSegmentationModel,
}


def build_external_model(spec: ExternalModelSpec) -> TorchModelAdapter | OnnxModelAdapter:
    """Instantiate the adapter that matches a manifest.

    Raises:
        ValidationError: for a framework/task pair with no generic adapter.
            Such a model needs a purpose-written adapter class.
    """
    adapter = _BY_FRAMEWORK_AND_TASK.get((spec.framework, spec.task))
    if adapter is None:
        raise ValidationError(
            "No generic adapter exists for this framework/task combination.",
            details={
                "framework": str(spec.framework),
                "task": str(spec.task),
                "supported": [f"{f}/{t}" for f, t in _BY_FRAMEWORK_AND_TASK],
            },
        )
    return adapter(spec=spec)  # type: ignore[return-value]
