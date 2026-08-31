"""Quality-control models.

Assertions are on *behaviour under a known defect* - a blurred phantom must
score lower than a sharp one and be flagged - rather than on absolute scores,
which are heuristic and camera-dependent.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ai.preprocessing import ExamImage, gaussian_blur
from app.core.enums import Modality
from app.ophthalmology.quality import FundusQualityModel, OCTQualityModel
from tests.factories import fundus_phantom, noise_image, oct_phantom


def _blur(image: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    channels = [
        gaussian_blur(image[..., c].astype(np.float32) / 255.0, sigma) * 255.0
        for c in range(image.shape[2])
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


@pytest.fixture
def fundus_model() -> FundusQualityModel:
    return FundusQualityModel()


@pytest.fixture
def oct_model() -> OCTQualityModel:
    return OCTQualityModel()


def test_good_fundus_image_passes(fundus_model: FundusQualityModel) -> None:
    result = fundus_model.run(ExamImage(pixels=fundus_phantom(), modality=Modality.FUNDUS))
    assert result.quality is not None
    assert result.quality.is_valid
    assert result.quality.issues == []
    assert result.quality.quality_score > 0.6
    assert result.processing_time_ms > 0


def test_blurred_fundus_image_is_flagged_and_scores_lower(
    fundus_model: FundusQualityModel,
) -> None:
    sharp = fundus_model.run(ExamImage(pixels=fundus_phantom()))
    blurred = fundus_model.run(ExamImage(pixels=_blur(fundus_phantom())))
    assert "blur" in blurred.quality.issues
    assert blurred.quality.is_valid is False
    assert blurred.quality.quality_score < sharp.quality.quality_score


def test_dark_fundus_image_is_rejected(fundus_model: FundusQualityModel) -> None:
    dark = (fundus_phantom() * 0.12).astype(np.uint8)
    result = fundus_model.run(ExamImage(pixels=dark))
    assert result.quality.is_valid is False
    assert "underexposed" in result.quality.issues


def test_fundus_model_detects_a_modality_mismatch(fundus_model: FundusQualityModel) -> None:
    """An OCT B-scan submitted as a fundus photograph must be refused."""
    scan, _ = oct_phantom()
    result = fundus_model.run(ExamImage(pixels=scan, modality=Modality.FUNDUS))
    assert result.quality.is_valid is False
    assert "modality_mismatch" in result.quality.issues


def test_good_oct_scan_passes(oct_model: OCTQualityModel) -> None:
    scan, _ = oct_phantom()
    result = oct_model.run(ExamImage(pixels=scan, modality=Modality.OCT))
    assert result.quality.is_valid
    assert result.quality.metrics["snr_db"] > 8.0


def test_weak_oct_signal_is_rejected(oct_model: OCTQualityModel) -> None:
    scan, _ = oct_phantom()
    weak = (scan * 0.18).astype(np.uint8)
    result = oct_model.run(ExamImage(pixels=weak, modality=Modality.OCT))
    assert result.quality.is_valid is False
    assert "no_retinal_signal" in result.quality.issues


def test_truncated_oct_scan_is_rejected(oct_model: OCTQualityModel) -> None:
    """A retina running off the frame invalidates thickness measurements."""
    scan, _ = oct_phantom()
    # Crop above the RPE: the retina now runs off the top of the frame, which is
    # exactly the acquisition error that invalidates a thickness measurement.
    truncated = scan[250:, :]
    result = oct_model.run(ExamImage(pixels=truncated, modality=Modality.OCT))
    assert result.quality.is_valid is False
    assert "truncated_scan" in result.quality.issues


def test_colour_image_is_not_accepted_as_oct(oct_model: OCTQualityModel) -> None:
    result = oct_model.run(ExamImage(pixels=fundus_phantom(), modality=Modality.OCT))
    assert result.quality.is_valid is False
    assert "modality_mismatch" in result.quality.issues


def test_noise_fails_both_modalities(
    fundus_model: FundusQualityModel, oct_model: OCTQualityModel
) -> None:
    noise = noise_image()
    assert fundus_model.run(ExamImage(pixels=noise)).quality.is_valid is False
    assert oct_model.run(ExamImage(pixels=noise)).quality.is_valid is False


def test_quality_models_emit_a_diagnostic_artifact(fundus_model: FundusQualityModel) -> None:
    result = fundus_model.run(ExamImage(pixels=fundus_phantom()), explain=True)
    assert [artifact.name for artifact in result.artifacts] == ["sharpness_map"]
    assert result.artifacts[0].data[:8] == b"\x89PNG\r\n\x1a\n"
