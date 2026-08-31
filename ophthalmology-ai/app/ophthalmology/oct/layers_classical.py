"""Classical ILM/RPE boundary detection on OCT B-scans.

This is the reference segmentation model of the platform: it needs no
checkpoint, so the full pipeline - upload, quality gate, inference,
segmentation mask, quantitative measurement, report - is demonstrable on any
machine.

Method (a standard intensity/gradient approach, not a learned model):

1. denoise the B-scan with a small Gaussian kernel to suppress speckle;
2. per A-scan (image column), find the **ILM** as the first depth where
   intensity rises above a column-adaptive threshold above the vitreous noise
   floor;
3. find the **RPE/photoreceptor complex** as the brightest response below the
   ILM;
4. reject implausible columns, then median-smooth both boundaries across
   columns to enforce the anatomical continuity of the retinal surface;
5. report per-column retinal thickness, converted to micrometres when the
   device's axial scale is known.

It measures a real, well-defined structure, but it is a heuristic: it has no
notion of pathology, it does not separate individual retinal layers, and it
degrades on scans with fluid, strong shadowing or steep tilt. Every result is
labelled ``evidence_level = heuristic`` and carries that caveat into the report.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.ai.base import ClassicalModel
from app.ai.explainability import curves_overlay_artifact, mask_overlay_artifact
from app.ai.postprocessing import mask_to_payload
from app.ai.preprocessing import ExamImage, gaussian_blur, to_grayscale
from app.ai.results import (
    ArtifactPayload,
    InputSpec,
    ModelLicense,
    ModelMetadata,
    ModelResult,
    OutputSpec,
)
from app.core.enums import EvidenceLevel, Framework, Modality, TaskType

#: Plausibility bounds for a retinal thickness measurement, in micrometres.
#: Values outside this range indicate a failed boundary detection rather than
#: an extreme retina, so those columns are dropped.
MIN_PLAUSIBLE_THICKNESS_UM = 80.0
MAX_PLAUSIBLE_THICKNESS_UM = 900.0

#: Fallback axial resolution, in micrometres per pixel, used only to sanity
#: check column detections when the device scale is unknown. Measurements are
#: never *reported* in micrometres without a real device scale.
ASSUMED_AXIAL_UM_PER_PX = 4.0


def _median_filter_1d(values: np.ndarray, size: int) -> np.ndarray:
    """Median filter along a 1-D signal, edges handled by reflection."""
    if size < 3:
        return values
    radius = size // 2
    padded = np.pad(values, radius, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * radius + 1)
    return np.median(windows, axis=-1)


def _interpolate_gaps(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid columns by linear interpolation between valid neighbours."""
    if valid.all():
        return values
    if not valid.any():
        return np.full_like(values, np.nan)
    indices = np.arange(values.size)
    return np.interp(indices, indices[valid], values[valid])


def detect_boundaries(
    gray: np.ndarray, *, smooth_columns: int = 15
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect ILM and RPE depths for every A-scan.

    Args:
        gray: B-scan luminance in ``[0, 1]``, shape ``(depth, width)``.
        smooth_columns: Width of the median filter applied across columns.

    Returns:
        ``(ilm, rpe, valid)`` - two float arrays of row indices per column and
        a boolean mask of the columns where detection succeeded.
    """
    denoised = gaussian_blur(gray, 1.6)
    depth, width = denoised.shape

    # Vitreous noise floor per column: the retina sits well above it.
    noise_floor = np.percentile(denoised, 20, axis=0)
    column_peak = denoised.max(axis=0)
    threshold = noise_floor + 0.35 * np.clip(column_peak - noise_floor, 0.0, None)

    above = denoised > threshold[None, :]
    # Require a few consecutive bright rows so isolated speckle is not mistaken
    # for the retinal surface.
    run = np.cumsum(above, axis=0)
    sustained = np.zeros_like(above)
    sustained[:-3] = (run[3:] - run[:-3]) >= 3
    has_signal = sustained.any(axis=0)
    ilm = np.where(has_signal, np.argmax(sustained, axis=0), -1).astype(np.float64)

    # RPE complex: brightest response below the ILM.
    rows = np.arange(depth)[:, None]
    below_ilm = rows > (ilm[None, :] + 4)
    masked = np.where(below_ilm, denoised, -np.inf)
    rpe = np.argmax(masked, axis=0).astype(np.float64)
    has_rpe = np.isfinite(masked.max(axis=0))

    valid = has_signal & has_rpe & (rpe > ilm + 4) & (ilm >= 0)
    if valid.any():
        thickness_px = rpe - ilm
        median_thickness = float(np.median(thickness_px[valid]))
        # Drop columns whose thickness deviates wildly from the scan's own
        # median - those are shadowing or vessel artefacts, not retina.
        valid &= np.abs(thickness_px - median_thickness) < max(median_thickness * 0.6, 8.0)

    ilm = _interpolate_gaps(ilm, valid)
    rpe = _interpolate_gaps(rpe, valid)
    if valid.sum() > smooth_columns:
        ilm = _median_filter_1d(ilm, smooth_columns)
        rpe = _median_filter_1d(rpe, smooth_columns)
    return np.clip(ilm, 0, depth - 1), np.clip(rpe, 0, depth - 1), valid


def build_retina_mask(ilm: np.ndarray, rpe: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Boolean mask of the tissue between the two boundaries."""
    depth, width = shape
    rows = np.arange(depth)[:, None]
    return (rows >= ilm[None, :]) & (rows <= rpe[None, :])


class OCTLayerBoundaryModel(ClassicalModel):
    """ILM/RPE boundary segmentation and retinal thickness for OCT B-scans."""

    metadata = ModelMetadata(
        model_id="oct_layers_classical_v1",
        name="OCT ILM/RPE Boundary Segmentation (classical)",
        version="1.0.0",
        modality=Modality.OCT,
        task=TaskType.SEGMENTATION,
        framework=Framework.CLASSICAL,
        evidence_level=EvidenceLevel.HEURISTIC,
        description=(
            "Intensity- and gradient-based detection of the inner limiting membrane and "
            "the RPE/photoreceptor complex, with per-A-scan retinal thickness."
        ),
        input_spec=InputSpec(
            channels=1,
            color_space="gray",
            scale_to_unit_interval=False,
            notes="Native-resolution B-scan; no resizing, so measurements stay in device pixels.",
        ),
        output_spec=OutputSpec(
            segmentation_classes=["background", "retina_ilm_to_rpe"],
            units={
                "retinal_thickness_um": "micrometre",
                "retinal_thickness_px": "pixel",
                "central_subfield_thickness_um": "micrometre",
            },
            notes=(
                "Micrometre values are reported only when the exam carries an axial "
                "pixel scale; otherwise thickness is reported in pixels."
            ),
        ),
        license=ModelLicense(
            name="Apache-2.0",
            commercial_use="allowed",
            citation=(
                "Classical intensity/gradient boundary detection in the spirit of "
                "Chiu et al., 'Automatic segmentation of seven retinal layers in SDOCT "
                "images congruent with expert manual segmentation', Optics Express 2010."
            ),
        ),
        supports_explainability=True,
        limitations=(
            "Detects only the outer envelope of the retina (ILM to RPE), not individual "
            "layers. Accuracy degrades with intraretinal fluid, vessel shadowing, steep "
            "tilt or low signal. Not a validated thickness measurement device; do not "
            "compare its values against a device's normative database."
        ),
    )

    def predict(self, prepared: ExamImage) -> dict[str, Any]:
        """Detect boundaries and compute per-column thickness."""
        gray = to_grayscale(prepared.pixels)
        ilm, rpe, valid = detect_boundaries(gray)
        thickness_px = rpe - ilm
        return {
            "gray_shape": gray.shape,
            "ilm": ilm,
            "rpe": rpe,
            "valid": valid,
            "thickness_px": thickness_px,
        }

    def postprocess(self, output: dict[str, Any], image: ExamImage) -> ModelResult:
        """Build the retina mask and the quantitative thickness summary."""
        shape: tuple[int, int] = output["gray_shape"]
        ilm, rpe, valid = output["ilm"], output["rpe"], output["valid"]
        thickness_px = output["thickness_px"]
        result = ModelResult(
            model_id=self.model_id,
            model_version=self.version,
            task=TaskType.SEGMENTATION,
        )

        coverage = float(valid.mean()) if valid.size else 0.0
        if coverage < 0.25:
            result.warnings.append(
                "Retinal boundaries could not be detected in most A-scans; "
                "measurements are unreliable."
            )

        mask = build_retina_mask(ilm, rpe, shape)
        result.segmentations.append(
            mask_to_payload(
                mask,
                "retina_ilm_to_rpe",
                pixel_spacing_um=image.pixel_spacing_um,
                measurements={"detected_column_ratio": round(coverage, 4)},
            )
        )
        result.measurements = self._measurements(thickness_px, valid, image, coverage, result)
        self._mask = mask
        return result

    def _measurements(
        self,
        thickness_px: np.ndarray,
        valid: np.ndarray,
        image: ExamImage,
        coverage: float,
        result: ModelResult,
    ) -> dict[str, Any]:
        """Summarize thickness, in micrometres when the device scale is known."""
        usable = thickness_px[valid] if valid.any() else thickness_px
        measurements: dict[str, Any] = {
            "detected_column_ratio": round(coverage, 4),
            "retinal_thickness_px": {
                "mean": round(float(np.mean(usable)), 2),
                "min": round(float(np.min(usable)), 2),
                "max": round(float(np.max(usable)), 2),
                "std": round(float(np.std(usable)), 2),
            },
        }

        axial = image.pixel_spacing_um.get("axial") or image.pixel_spacing_um.get("y")
        if not axial:
            result.warnings.append(
                "Exam carries no axial pixel scale, so retinal thickness is reported in "
                "pixels only. Provide pixel_spacing_um.axial to obtain micrometres."
            )
            return measurements

        thickness_um = usable * float(axial)
        plausible = thickness_um[
            (thickness_um >= MIN_PLAUSIBLE_THICKNESS_UM)
            & (thickness_um <= MAX_PLAUSIBLE_THICKNESS_UM)
        ]
        if plausible.size == 0:
            result.warnings.append(
                "Detected thickness values fall outside a physiologically plausible "
                "range; boundary detection likely failed on this scan."
            )
            return measurements

        centre = self._central_subfield(thickness_px, valid, image, float(axial))
        measurements["retinal_thickness_um"] = {
            "mean": round(float(np.mean(plausible)), 1),
            "min": round(float(np.min(plausible)), 1),
            "max": round(float(np.max(plausible)), 1),
            "std": round(float(np.std(plausible)), 1),
            "plausible_column_ratio": round(float(plausible.size / max(usable.size, 1)), 4),
        }
        if centre is not None:
            measurements["central_subfield_thickness_um"] = round(centre, 1)
        return measurements

    def _central_subfield(
        self, thickness_px: np.ndarray, valid: np.ndarray, image: ExamImage, axial: float
    ) -> float | None:
        """Mean thickness over the central 1 mm of the scan, when scale allows.

        Falls back to the central 10 % of A-scans when no lateral scale is
        known - reported as a central-region average, not an ETDRS subfield.
        """
        width = thickness_px.size
        lateral = image.pixel_spacing_um.get("lateral") or image.pixel_spacing_um.get("x")
        if lateral:
            half_width_px = int(1000.0 / (2.0 * float(lateral)))
        else:
            half_width_px = max(int(width * 0.05), 1)
        centre = width // 2
        lo, hi = max(centre - half_width_px, 0), min(centre + half_width_px + 1, width)
        window = thickness_px[lo:hi]
        window_valid = valid[lo:hi]
        usable = window[window_valid] if window_valid.any() else window
        if usable.size == 0:
            return None
        return float(np.mean(usable) * axial)

    def explain(self, image: ExamImage, output: dict[str, Any]) -> list[ArtifactPayload]:
        """Draw the detected boundaries and the segmented retina."""
        artifacts = [
            curves_overlay_artifact(
                image,
                {"ilm": output["ilm"], "rpe": output["rpe"]},
                name="ilm_rpe_boundaries",
            )
        ]
        mask = getattr(self, "_mask", None)
        if mask is not None:
            artifacts.append(
                mask_overlay_artifact(mask, image, name="retina_overlay", color=(80, 200, 255))
            )
        return artifacts
