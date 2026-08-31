"""OCT B-scan quality control.

Same contract as the fundus quality model, different physics: an OCT B-scan is
a grayscale depth image whose failure modes are weak signal (poor coupling,
media opacity), speckle noise, truncation of the retina at the top or bottom of
the frame, and tilt.

Like its fundus counterpart this is a **heuristic** gate, not a validated
signal-strength score. Device-reported signal strength (SSI/SQ/Q-score), when
available, is more reliable and can be supplied through
``Image.pixel_spacing_um``-style acquisition metadata and used instead.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ai.base import ClassicalModel
from app.ai.explainability import gradcam_artifact
from app.ai.preprocessing import ExamImage
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
from app.ophthalmology.quality.fundus_quality import weighted_geometric_mean

THRESHOLDS: dict[str, float] = {
    # Signal-to-noise proxy, in dB, at which the scan is considered strong.
    "snr_reference": 18.0,
    "snr_reject": 8.0,
    # Mean intensity of the brightest 5 % of pixels (the RPE complex).
    "signal_reference": 0.55,
    "signal_reject": 0.18,
    # Sharpness of the retinal band boundaries.
    "sharpness_reference": 3.0e-4,
    "sharpness_reject": 5e-5,
    "contrast_reference": 0.14,
    "contrast_reject": 0.05,
    # Fraction of the top/bottom frame edge covered by tissue.
    "truncation_reject": 0.15,
    # Retina tilt across the B-scan, as a fraction of image height.
    "tilt_reject": 0.28,
    # A B-scan should be grayscale; colour suggests the wrong file was sent.
    "colorfulness_reject": 22.0,
}

WEIGHTS: dict[str, float] = {
    "signal": 0.30,
    "snr": 0.26,
    "sharpness": 0.24,
    "contrast": 0.20,
}


def _ramp(value: float, reference: float, floor: float = 0.0) -> float:
    """Linear ramp from ``floor`` to ``reference``, clipped to ``[0, 1]``."""
    span = reference - floor
    if span <= 0:
        return 1.0
    return float(min(max((value - floor) / span, 0.0), 1.0))


def retina_tilt(gray: np.ndarray) -> float:
    """Vertical spread of the retinal band across the scan, normalized to height.

    Computed from the per-column intensity centroid of the brightest tissue.
    A large value means the retina runs diagonally through the frame, which
    degrades layer segmentation and thickness maps.
    """
    height = gray.shape[0]
    weights = np.clip(gray - np.percentile(gray, 60), 0, None)
    column_mass = weights.sum(axis=0)
    valid = column_mass > 1e-6
    if valid.sum() < gray.shape[1] * 0.2:
        return 0.0
    rows = np.arange(height, dtype=np.float32)[:, None]
    centroids = (weights * rows).sum(axis=0)[valid] / column_mass[valid]
    # Robust spread: 5th-95th percentile, so a couple of noisy columns do not
    # dominate the estimate.
    spread = float(np.percentile(centroids, 95) - np.percentile(centroids, 5))
    return spread / height


class OCTQualityModel(ClassicalModel):
    """No-reference quality assessment for OCT B-scans."""

    metadata = ModelMetadata(
        model_id="oct_quality_v1",
        name="OCT B-scan Quality (heuristic)",
        version="1.0.0",
        modality=Modality.OCT,
        task=TaskType.QUALITY,
        framework=Framework.CLASSICAL,
        evidence_level=EvidenceLevel.HEURISTIC,
        description=(
            "Deterministic no-reference quality control for OCT B-scans: signal "
            "strength, SNR, boundary sharpness, contrast, truncation, tilt and "
            "modality consistency."
        ),
        input_spec=InputSpec(
            channels=1,
            color_space="gray",
            scale_to_unit_interval=False,
            notes="Operates on the native-resolution B-scan; no resizing is applied.",
        ),
        output_spec=OutputSpec(
            labels=["acceptable", "unacceptable"],
            units={"snr_db": "dB", "signal": "normalized intensity"},
        ),
        license=ModelLicense(name="Apache-2.0", commercial_use="allowed"),
        supports_explainability=True,
        limitations=(
            "Heuristic quality control, not a device signal-strength index. Where the "
            "OCT device reports its own signal quality metric, prefer that value."
        ),
    )

    def predict(self, prepared: ExamImage) -> dict[str, Any]:
        """Measure every quality dimension of the B-scan."""
        pixels = prepared.pixels
        gray = metrics.prepare_gray(pixels)
        measured: dict[str, Any] = {
            "sharpness": metrics.sharpness(gray),
            "contrast": metrics.contrast(gray),
            "colorfulness": metrics.colorfulness(pixels),
            "band_energy_ratio": metrics.horizontal_band_energy(gray),
            "tilt": retina_tilt(gray),
        }
        measured.update(metrics.signal_to_noise(gray))
        measured.update(metrics.exposure(gray))
        measured.update(metrics.truncation(gray))
        return measured

    def postprocess(self, output: dict[str, Any], image: ExamImage) -> ModelResult:
        """Turn raw measurements into a quality report and a pass/fail score."""
        settings = get_settings()
        scores = {
            "signal": _ramp(output["signal"], THRESHOLDS["signal_reference"]),
            "snr": _ramp(output["snr_db"], THRESHOLDS["snr_reference"], floor=4.0),
            "sharpness": _ramp(output["sharpness"], THRESHOLDS["sharpness_reference"]),
            "contrast": _ramp(output["contrast"], THRESHOLDS["contrast_reference"]),
        }
        quality_score = weighted_geometric_mean(scores, WEIGHTS)

        issues = self._collect_issues(output)
        blocking = {"modality_mismatch", "no_retinal_signal", "truncated_scan"}
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
                else "repeat the scan; signal quality is insufficient for reliable analysis"
            ),
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
        """Name the concrete defects found in the scan."""
        issues: list[str] = []
        if m["colorfulness"] > THRESHOLDS["colorfulness_reject"]:
            # A strongly coloured image is not an OCT B-scan.
            issues.append("modality_mismatch")
        if m["signal"] < THRESHOLDS["signal_reject"]:
            issues.append("no_retinal_signal")
        if m["snr_db"] < THRESHOLDS["snr_reject"]:
            issues.append("low_signal_to_noise")
        if m["sharpness"] < THRESHOLDS["sharpness_reject"]:
            issues.append("blur")
        if m["contrast"] < THRESHOLDS["contrast_reject"]:
            issues.append("low_contrast")
        if m["clipped_bright_ratio"] > 0.10:
            issues.append("saturated_signal")
        if (
            m["top_contact_ratio"] > THRESHOLDS["truncation_reject"]
            or m["bottom_contact_ratio"] > THRESHOLDS["truncation_reject"]
        ):
            issues.append("truncated_scan")
        if m["tilt"] > THRESHOLDS["tilt_reject"]:
            issues.append("excessive_tilt")
        return issues

    def explain(self, image: ExamImage, output: dict[str, Any]) -> list[ArtifactPayload]:
        """Show the depth profile of the signal that drove the score."""
        gray = metrics.prepare_gray(image.pixels)
        profile = metrics.retina_row_profile(gray)
        # Broadcast the row profile across the width to visualize which depths
        # carry signal.
        heatmap = np.repeat(profile[:, None], gray.shape[1], axis=1)
        return [
            gradcam_artifact(
                heatmap,
                image,
                name="signal_profile",
                meta={"method": "row-intensity-profile"},
            )
        ]
