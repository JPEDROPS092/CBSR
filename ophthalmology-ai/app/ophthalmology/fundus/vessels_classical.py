"""Classical retinal vessel segmentation on fundus photographs.

Green-channel contrast enhancement followed by a morphological black-hat
response, adaptive thresholding and small-component removal - the standard
unsupervised pipeline that predates learned vessel segmentation and still runs
anywhere in milliseconds.

It gives the platform a working fundus segmentation without any checkpoint.
For research or clinical work, register a trained vessel model (U-Net on
DRIVE/CHASE/HRF or similar) through a manifest and use this one only as a
fallback or sanity check.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ai.base import ClassicalModel
from app.ai.explainability import mask_overlay_artifact
from app.ai.postprocessing import mask_to_payload
from app.ai.preprocessing import ExamImage, estimate_fov_mask
from app.ai.results import (
    ArtifactPayload,
    Availability,
    InputSpec,
    ModelLicense,
    ModelMetadata,
    ModelResult,
    OutputSpec,
)
from app.core.enums import EvidenceLevel, Framework, Modality, TaskType

#: Structuring-element diameter for the black-hat response, as a fraction of
#: the image's shorter side. Sized to be wider than the widest retinal vessel.
STRUCTURING_ELEMENT_RATIO = 0.022

#: Response percentile (inside the field of view) used as the vessel threshold.
THRESHOLD_PERCENTILE = 92.0

#: Connected components smaller than this fraction of the field of view are
#: dropped as noise.
MIN_COMPONENT_RATIO = 2e-5


def _require_cv2() -> Any:
    """Import OpenCV, which this model needs for morphology and components."""
    import cv2

    return cv2


class RetinalVesselClassicalModel(ClassicalModel):
    """Unsupervised retinal vessel segmentation."""

    metadata = ModelMetadata(
        model_id="fundus_vessels_classical_v1",
        name="Retinal Vessel Segmentation (classical)",
        version="1.0.0",
        modality=Modality.FUNDUS,
        task=TaskType.SEGMENTATION,
        framework=Framework.CLASSICAL,
        evidence_level=EvidenceLevel.HEURISTIC,
        description=(
            "Green-channel CLAHE plus morphological black-hat vessel enhancement, "
            "percentile thresholding and small-component removal."
        ),
        input_spec=InputSpec(
            channels=3,
            color_space="rgb",
            scale_to_unit_interval=False,
            notes="Native resolution; the green channel carries the vessel contrast.",
        ),
        output_spec=OutputSpec(
            segmentation_classes=["background", "vessels"],
            units={"vessel_area_ratio": "fraction of the field of view"},
        ),
        license=ModelLicense(
            name="Apache-2.0",
            commercial_use="allowed",
            citation=(
                "Morphological vessel enhancement following Zana & Klein, "
                "'Segmentation of vessel-like patterns using mathematical morphology "
                "and curvature evaluation', IEEE TIP 2001."
            ),
        ),
        supports_explainability=True,
        limitations=(
            "Unsupervised baseline: it under-detects thin peripheral vessels, and "
            "bright lesions or strong pathology can produce false positives. Vessel "
            "density from this model is not comparable with published densitometry."
        ),
    )

    def availability(self) -> Availability:
        base = super().availability()
        if not base.available:
            return base
        try:
            _require_cv2()
        except ImportError:
            return Availability(
                available=False,
                reason="OpenCV is required for morphological vessel segmentation.",
                missing=["opencv-python-headless"],
                remediation="pip install opencv-python-headless",
            )
        return base

    def predict(self, prepared: ExamImage) -> dict[str, Any]:
        """Enhance and threshold the vessel response."""
        cv2 = _require_cv2()
        rgb = prepared.to_rgb()
        green = rgb[:, :, 1]

        fov = estimate_fov_mask(rgb)
        if fov.sum() < 0.02 * fov.size:
            fov = np.ones_like(fov)
        # Shrink the field of view so the bright rim of the aperture is not
        # mistaken for a vessel.
        erosion = max(int(min(green.shape) * 0.02), 3)
        fov_eroded = cv2.erode(fov.astype(np.uint8), np.ones((erosion, erosion), np.uint8)) > 0

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(green)

        size = max(int(min(green.shape) * STRUCTURING_ELEMENT_RATIO) | 1, 5)
        element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        # Vessels are dark, thin structures: black-hat isolates them from the
        # smoothly varying retinal background.
        response = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, element)
        response = cv2.GaussianBlur(response, (3, 3), 0)

        inside = response[fov_eroded]
        threshold = float(np.percentile(inside, THRESHOLD_PERCENTILE)) if inside.size else 255.0
        binary = ((response > threshold) & fov_eroded).astype(np.uint8)

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        min_area = max(int(fov_eroded.sum() * MIN_COMPONENT_RATIO), 12)
        keep = np.zeros(count, dtype=bool)
        for index in range(1, count):
            keep[index] = stats[index, cv2.CC_STAT_AREA] >= min_area
        mask = keep[labels]

        return {
            "mask": mask,
            "response": response.astype(np.float32) / 255.0,
            "fov": fov_eroded,
            "component_count": int(keep.sum()),
            "threshold": threshold,
        }

    def postprocess(self, output: dict[str, Any], image: ExamImage) -> ModelResult:
        """Report the vessel mask and its density inside the field of view."""
        mask: np.ndarray = output["mask"]
        fov: np.ndarray = output["fov"]
        fov_area = float(fov.sum()) or 1.0
        density = float(mask.sum()) / fov_area

        result = ModelResult(
            model_id=self.model_id,
            model_version=self.version,
            task=TaskType.SEGMENTATION,
        )
        result.segmentations.append(
            mask_to_payload(
                mask,
                "vessels",
                pixel_spacing_um=image.pixel_spacing_um,
                measurements={
                    "vessel_area_ratio": round(density, 5),
                    "component_count": output["component_count"],
                },
            )
        )
        result.measurements = {
            "vessel_area_ratio": round(density, 5),
            "vessel_area_px": float(mask.sum()),
            "field_of_view_px": fov_area,
            "component_count": output["component_count"],
        }
        if density < 0.01:
            result.warnings.append(
                "Very few vessel pixels were detected; the image may be blurred, "
                "under-exposed or not a fundus photograph."
            )
        return result

    def explain(self, image: ExamImage, output: dict[str, Any]) -> list[ArtifactPayload]:
        """Overlay the detected vessel tree on the photograph."""
        return [
            mask_overlay_artifact(
                output["mask"],
                image,
                name="vessels_overlay",
                color=(80, 255, 140),
                meta={"threshold": round(output["threshold"], 3)},
            )
        ]
