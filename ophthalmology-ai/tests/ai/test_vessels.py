"""Classical vessel segmentation, measured against a synthetic vessel tree."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image as PILImage

from app.ai.preprocessing import ExamImage
from app.core.enums import Modality
from app.ophthalmology.fundus import RetinalVesselClassicalModel


def _vessel_phantom(size: int = 384) -> tuple[np.ndarray, np.ndarray]:
    """A fundus-like disc with dark radial vessels; returns image and truth mask."""
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (0.46 * size) ** 2
    base = np.where(disc, 150.0, 0.0)

    vessels = np.zeros((size, size), dtype=bool)
    for angle in np.linspace(0.2, 3.0, 9):
        steps = np.linspace(0, size * 0.55, 700)
        ys = (size / 2 + steps * np.sin(angle) + 14 * np.sin(steps / 28.0)).astype(int)
        xs = (size / 2 + steps * np.cos(angle) + 14 * np.cos(steps / 33.0)).astype(int)
        inside = (ys > 1) & (ys < size - 2) & (xs > 1) & (xs < size - 2)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                vessels[ys[inside] + dy, xs[inside] + dx] = True

    green = np.where(vessels, base * 0.45, base) + rng.normal(0, 4, (size, size))
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = np.clip(green * 2.0, 0, 255)
    image[..., 1] = np.clip(green, 0, 255)
    image[..., 2] = np.clip(green * 0.6, 0, 255)
    return image, vessels & disc


@pytest.fixture
def model() -> RetinalVesselClassicalModel:
    return RetinalVesselClassicalModel()


def test_model_is_available_when_opencv_is_installed(
    model: RetinalVesselClassicalModel,
) -> None:
    availability = model.availability()
    assert availability.available, availability.reason


def test_vessels_are_recovered_from_the_phantom(model: RetinalVesselClassicalModel) -> None:
    """A baseline must actually find the vessels, not just produce a mask."""
    image, truth = _vessel_phantom()
    result = model.run(ExamImage(pixels=image, modality=Modality.FUNDUS))
    predicted = np.array(PILImage.open(io.BytesIO(result.segmentations[0].data))) > 127
    recall = (predicted & truth).sum() / truth.sum()
    precision = (predicted & truth).sum() / max(predicted.sum(), 1)
    assert recall > 0.7
    assert precision > 0.5
    assert 0.01 < result.measurements["vessel_area_ratio"] < 0.25


def test_blank_image_produces_a_warning_not_a_finding(
    model: RetinalVesselClassicalModel,
) -> None:
    """With nothing to find, the model says so instead of inventing vessels."""
    flat = np.full((256, 256, 3), 140, dtype=np.uint8)
    result = model.run(ExamImage(pixels=flat, modality=Modality.FUNDUS))
    assert result.measurements["vessel_area_ratio"] < 0.02
    assert any("few vessel pixels" in warning for warning in result.warnings)


def test_overlay_artifact_is_produced(model: RetinalVesselClassicalModel) -> None:
    image, _ = _vessel_phantom(size=256)
    result = model.run(ExamImage(pixels=image), explain=True)
    assert [artifact.name for artifact in result.artifacts] == ["vessels_overlay"]
