"""Classical OCT boundary detection, measured against a phantom's ground truth."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image as PILImage

from app.ai.preprocessing import ExamImage
from app.core.enums import Modality
from app.ophthalmology.oct import OCTLayerBoundaryModel
from app.ophthalmology.oct.layers_classical import build_retina_mask, detect_boundaries
from tests.factories import oct_phantom

AXIAL_UM = 3.87


@pytest.fixture
def model() -> OCTLayerBoundaryModel:
    return OCTLayerBoundaryModel()


def _image(thickness: int = 80) -> tuple[ExamImage, int]:
    pixels, truth = oct_phantom(retina_thickness_px=thickness)
    return (
        ExamImage(
            pixels=pixels,
            modality=Modality.OCT,
            pixel_spacing_um={"axial": AXIAL_UM, "lateral": 11.7},
        ),
        truth,
    )


def test_boundaries_are_detected_on_every_column() -> None:
    pixels, _ = oct_phantom()
    ilm, rpe, valid = detect_boundaries(pixels.astype(np.float32) / 255.0)
    assert valid.mean() > 0.95
    assert np.all(rpe > ilm)


def test_measured_thickness_matches_the_phantom() -> None:
    """The reported thickness must track the phantom's real separation."""
    for thickness in (60, 80, 110):
        image, truth = _image(thickness)
        result = OCTLayerBoundaryModel().run(image)
        measured = result.measurements["retinal_thickness_um"]["mean"]
        assert measured == pytest.approx(truth * AXIAL_UM, abs=25.0)


def test_thicker_retina_measures_thicker(model: OCTLayerBoundaryModel) -> None:
    thin = model.run(_image(60)[0]).measurements["retinal_thickness_um"]["mean"]
    thick = model.run(_image(110)[0]).measurements["retinal_thickness_um"]["mean"]
    assert thick > thin + 100


def test_mask_is_produced_and_covers_the_retina(model: OCTLayerBoundaryModel) -> None:
    image, _ = _image()
    result = model.run(image)
    segmentation = result.segmentations[0]
    assert segmentation.label == "retina_ilm_to_rpe"
    decoded = np.array(PILImage.open(io.BytesIO(segmentation.data)))
    assert decoded.shape == (image.height, image.width)
    assert 0.05 < (decoded > 127).mean() < 0.5
    assert segmentation.measurements["area_mm2"] > 0


def test_micrometre_values_require_a_device_scale(model: OCTLayerBoundaryModel) -> None:
    """Without an axial scale the model reports pixels and says why."""
    pixels, _ = oct_phantom()
    result = model.run(ExamImage(pixels=pixels, modality=Modality.OCT))
    assert "retinal_thickness_px" in result.measurements
    assert "retinal_thickness_um" not in result.measurements
    assert any("axial pixel scale" in warning for warning in result.warnings)


def test_central_subfield_is_reported(model: OCTLayerBoundaryModel) -> None:
    image, _ = _image()
    result = model.run(image)
    assert result.measurements["central_subfield_thickness_um"] > 0


def test_explainability_draws_boundaries_and_overlay(model: OCTLayerBoundaryModel) -> None:
    image, _ = _image()
    result = model.run(image, explain=True)
    names = {artifact.name for artifact in result.artifacts}
    assert names == {"ilm_rpe_boundaries", "retina_overlay"}


def test_limitations_are_attached_to_every_result(model: OCTLayerBoundaryModel) -> None:
    """A heuristic result must carry its caveat downstream."""
    result = model.run(_image()[0])
    assert any("not a validated thickness measurement" in w.lower() for w in result.warnings)


def test_retina_mask_geometry() -> None:
    ilm = np.array([2.0, 2.0, 2.0])
    rpe = np.array([5.0, 5.0, 5.0])
    mask = build_retina_mask(ilm, rpe, (8, 3))
    assert mask[:, 0].sum() == 4  # rows 2..5 inclusive
    assert not mask[0, 0] and not mask[7, 0]
