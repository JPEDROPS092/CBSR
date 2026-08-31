"""Fundus image quality control.

A deterministic, no-reference quality model. It runs anywhere - no weights, no
GPU - which is what makes it usable as the platform's default quality gate.

The thresholds below are engineering defaults measured on 8-bit colour fundus
photographs. They are **heuristics**, not a validated quality classifier: a
site should calibrate them against its own cameras (see
``docs/MODEL_REGISTRY.md``) and may replace this model with a trained one by
registering another model with ``task = quality``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ai.base import ClassicalModel
from app.ai.explainability import gradcam_artifact
from app.ai.preprocessing import ExamImage, estimate_fov_mask, laplacian
from app.ai.results import (
    ArtifactPayload,
    InputSpec,
    ModelLicense,
    ModelMetadata,
    ModelResult,
    OutputSpec,
    PredictionItem,
    QualityReport,
)
from app.core.config import get_settings
from app.core.enums import EvidenceLevel, Framework, Modality, TaskType
from app.ophthalmology.quality import metrics

#: Tunable decision thresholds. Values are for luminance in ``[0, 1]``.
THRESHOLDS: dict[str, float] = {
    # Variance of the Laplacian at which an image is considered fully sharp.
    "sharpness_reference": 2.5e-4,
    "sharpness_reject": 6e-5,
    # Acceptable mean luminance inside the field of view.
    "exposure_low": 0.16,
    "exposure_high": 0.72,
    "exposure_ideal_low": 0.25,
    "exposure_ideal_high": 0.60,
    "clipping_reject": 0.12,
    # RMS contrast at which contrast is considered adequate.
    "contrast_reference": 0.11,
    "contrast_reject": 0.045,
    # Fraction of the frame the illuminated field should cover.
    "fov_reference": 0.35,
    "fov_reject": 0.12,
    # Tile-to-tile illumination ratio.
    "uniformity_reference": 0.35,
    "uniformity_reject": 0.12,
    # Colourfulness below this suggests the image is not a colour fundus photo.
    "colorfulness_reject": 6.0,
    # Vertical/horizontal gradient ratio typical of an OCT B-scan.
    "oct_like_band_energy": 1.6,
}

#: Relative weight of each sub-score in the composite quality score.
WEIGHTS: dict[str, float] = {
    "sharpness": 0.30,
    "exposure": 0.22,
    "contrast": 0.18,
    "field_of_view": 0.18,
    "uniformity": 0.12,
}


def _ramp(value: float, reference: float) -> float:
    """Linear ramp saturating at 1.0 once ``value`` reaches ``reference``."""
    if reference <= 0:
        return 1.0
    return float(min(max(value / reference, 0.0), 1.0))


def _exposure_score(mean_luminance: float) -> float:
    """1.0 inside the ideal band, falling off linearly to 0 at the limits."""
    low, high = THRESHOLDS["exposure_ideal_low"], THRESHOLDS["exposure_ideal_high"]
    if low <= mean_luminance <= high:
        return 1.0
    if mean_luminance < low:
        floor = THRESHOLDS["exposure_low"]
        return float(max((mean_luminance - floor) / max(low - floor, 1e-6), 0.0))
    ceiling = THRESHOLDS["exposure_high"]
    return float(max((ceiling - mean_luminance) / max(ceiling - high, 1e-6), 0.0))


def weighted_geometric_mean(scores: dict[str, float], weights: dict[str, float]) -> float:
    """Weighted geometric mean.

    Chosen over an arithmetic mean because one catastrophic factor - a black
    frame, total blur - must drag the composite score down rather than being
    averaged away by the others.
    """
    total_weight = sum(weights[name] for name in scores)
    accumulated = 0.0
    for name, value in scores.items():
        accumulated += weights[name] * np.log(max(value, 1e-4))
    return float(np.exp(accumulated / max(total_weight, 1e-6)))


class FundusQualityModel(ClassicalModel):
    """No-reference quality assessment for colour fundus photographs."""

    metadata = ModelMetadata(
        model_id="fundus_quality_v1",
        name="Fundus Image Quality (heuristic)",
        version="1.0.0",
        modality=Modality.FUNDUS,
        task=TaskType.QUALITY,
        framework=Framework.CLASSICAL,
        evidence_level=EvidenceLevel.HEURISTIC,
        description=(
            "Deterministic no-reference quality control for colour fundus photographs: "
            "blur, exposure, clipping, contrast, field-of-view coverage, illumination "
            "uniformity and modality consistency."
        ),
        input_spec=InputSpec(
            channels=3,
            color_space="rgb",
            scale_to_unit_interval=False,
            notes="Operates on the native-resolution image; no resizing is applied.",
        ),
        output_spec=OutputSpec(
            labels=["acceptable", "unacceptable"],
            units={
                "sharpness": "variance of Laplacian (luminance in [0,1])",
                "mean_luminance": "normalized luminance",
            },
            notes="Also emits a QualityReport consumed by the pipeline's quality gate.",
        ),
        license=ModelLicense(
            name="Apache-2.0",
            commercial_use="allowed",
            citation=(
                "Blur measure: Pech-Pacheco et al., 'Diatom autofocusing in brightfield "
                "microscopy', ICPR 2000. Colourfulness: Hasler & Susstrunk, SPIE 2003."
            ),
        ),
        supports_explainability=True,
        limitations=(
            "Heuristic quality control, not a validated image-quality classifier. "
            "Thresholds are defaults for 8-bit colour fundus photographs and should be "
            "calibrated per camera before clinical or research use."
        ),
    )

    def predict(self, prepared: ExamImage) -> dict[str, Any]:
        """Measure every quality dimension of the image."""
        pixels = prepared.pixels
        gray = metrics.prepare_gray(pixels)
        fov_mask = estimate_fov_mask(pixels)
        if fov_mask.sum() < 0.02 * fov_mask.size:
            fov_mask = np.ones_like(fov_mask)

        measured: dict[str, Any] = {
            "sharpness": metrics.sharpness(gray, fov_mask),
            "edge_density": metrics.edge_density(gray, fov_mask),
            "contrast": metrics.contrast(gray, fov_mask),
            "uniformity": metrics.illumination_uniformity(gray, fov_mask),
            "colorfulness": metrics.colorfulness(pixels),
            "band_energy_ratio": metrics.horizontal_band_energy(gray),
        }
        measured.update(metrics.exposure(gray, fov_mask))
        measured.update(metrics.field_of_view(pixels))
        return measured

    def postprocess(self, output: dict[str, Any], image: ExamImage) -> ModelResult:
        """Turn raw measurements into a quality report and a pass/fail score."""
        settings = get_settings()
        scores = {
            "sharpness": _ramp(output["sharpness"], THRESHOLDS["sharpness_reference"]),
            "exposure": _exposure_score(output["mean_luminance"]),
            "contrast": _ramp(output["contrast"], THRESHOLDS["contrast_reference"]),
            "field_of_view": _ramp(output["fov_ratio"], THRESHOLDS["fov_reference"]),
            "uniformity": _ramp(output["uniformity"], THRESHOLDS["uniformity_reference"]),
        }
        clipping_penalty = 1.0 - min(
            output["clipped_bright_ratio"] + output["clipped_dark_ratio"], 1.0
        )
        quality_score = weighted_geometric_mean(scores, WEIGHTS) * max(clipping_penalty, 0.05)

        issues = self._collect_issues(output)
        # Defects that reject the image regardless of the composite score: a
        # blurred or wrong-modality image cannot be rescued by good exposure.
        blocking = {"blur", "insufficient_field_of_view", "modality_mismatch", "empty_image"}
        is_valid = quality_score >= settings.QUALITY_GATE_MIN_SCORE and not blocking.intersection(
            issues
        )

        report = QualityReport(
            quality_score=round(min(max(quality_score, 0.0), 1.0), 4),
            is_valid=is_valid,
            issues=issues,
            recommendation=(
                "image suitable for analysis"
                if is_valid
                else "reacquire the image; quality is insufficient for reliable analysis"
            ),
            # Raw measurements only; the derived sub-scores are reported
            # separately so a reader can see both what was measured and how it
            # was scored.
            metrics={k: round(float(v), 6) for k, v in output.items()},
        )
        return ModelResult(
            model_id=self.model_id,
            model_version=self.version,
            task=TaskType.QUALITY,
            quality=report,
            predictions=[
                PredictionItem(
                    label="acceptable" if is_valid else "unacceptable",
                    score=report.quality_score if is_valid else 1.0 - report.quality_score,
                )
            ],
            measurements={"sub_scores": {k: round(v, 4) for k, v in scores.items()}},
        )

    def _collect_issues(self, m: dict[str, Any]) -> list[str]:
        """Name the concrete defects found, in the order an operator would act on."""
        issues: list[str] = []
        if m["fov_ratio"] < 0.01:
            issues.append("empty_image")
        if m["colorfulness"] < THRESHOLDS["colorfulness_reject"]:
            if m["band_energy_ratio"] > THRESHOLDS["oct_like_band_energy"]:
                issues.append("modality_mismatch")
            else:
                issues.append("image_not_colour")
        if m["sharpness"] < THRESHOLDS["sharpness_reject"]:
            issues.append("blur")
        if m["mean_luminance"] < THRESHOLDS["exposure_low"]:
            issues.append("underexposed")
        if m["mean_luminance"] > THRESHOLDS["exposure_high"]:
            issues.append("overexposed")
        if m["clipped_bright_ratio"] > THRESHOLDS["clipping_reject"]:
            issues.append("saturated_highlights")
        if m["clipped_dark_ratio"] > THRESHOLDS["clipping_reject"]:
            issues.append("crushed_shadows")
        if m["contrast"] < THRESHOLDS["contrast_reject"]:
            issues.append("low_contrast")
        if m["fov_ratio"] < THRESHOLDS["fov_reject"]:
            issues.append("insufficient_field_of_view")
        if m["uniformity"] < THRESHOLDS["uniformity_reject"]:
            issues.append("uneven_illumination")
        if m["border_contact_ratio"] > 0.6 and m["fov_ratio"] > 0.9:
            issues.append("field_of_view_cropped")
        if m["centroid_offset"] > 0.25:
            issues.append("off_centre_field")
        return issues

    def explain(self, image: ExamImage, output: dict[str, Any]) -> list[ArtifactPayload]:
        """Local sharpness map: shows *where* the image lost detail."""
        gray = metrics.prepare_gray(image.pixels)
        response = np.abs(laplacian(gray))
        # Pool into blocks so the map shows regions, not pixel noise.
        block = max(min(gray.shape) // 48, 1)
        height = (gray.shape[0] // block) * block
        width = (gray.shape[1] // block) * block
        pooled = (
            response[:height, :width]
            .reshape(height // block, block, width // block, block)
            .mean(axis=(1, 3))
        )
        return [
            gradcam_artifact(
                pooled,
                image,
                name="sharpness_map",
                meta={"method": "local-laplacian-energy", "block_size": block},
            )
        ]
