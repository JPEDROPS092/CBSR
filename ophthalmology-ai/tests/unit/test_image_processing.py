"""Preprocessing and postprocessing primitives."""

from __future__ import annotations

import numpy as np
import pytest

from app.ai.postprocessing import (
    binary_mask_stats,
    label_mask_to_payloads,
    mask_to_payload,
    normalize_heatmap,
    scores_to_predictions,
    sigmoid,
    softmax,
)
from app.ai.preprocessing import (
    crop_to_fov,
    decode_image,
    decode_series,
    encode_png,
    estimate_fov_mask,
    gaussian_blur,
    normalize,
    prepare_for_model,
    resize,
    to_grayscale,
)
from app.ai.results import InputSpec
from app.core.exceptions import ValidationError
from tests.factories import encode, fundus_phantom


def test_decode_round_trip_preserves_pixels() -> None:
    original = fundus_phantom(size=64)
    decoded = decode_image(encode(original))
    assert decoded.pixels.shape == original.shape
    np.testing.assert_array_equal(decoded.pixels, original)
    assert decoded.is_color


def test_decode_rejects_non_image_bytes() -> None:
    with pytest.raises(ValidationError):
        decode_image(b"this is not an image")


def test_decode_series_reads_every_frame() -> None:
    """A multi-frame TIFF is read as an ordered B-scan series."""
    import io

    from PIL import Image as PILImage

    frames = [
        PILImage.fromarray((np.ones((16, 16)) * value).astype(np.uint8)) for value in (10, 20, 30)
    ]
    buffer = io.BytesIO()
    frames[0].save(buffer, format="TIFF", save_all=True, append_images=frames[1:])
    series = decode_series(buffer.getvalue())
    assert [frame.frame_index for frame in series] == [0, 1, 2]
    assert [int(frame.pixels.mean()) for frame in series] == [10, 20, 30]
    assert all(frame.frame_count == 3 for frame in series)


def test_input_fingerprint_is_stable_and_content_sensitive() -> None:
    """The fingerprint identifies exactly the pixels a model consumed."""
    image = decode_image(encode(fundus_phantom(size=32)))
    other = decode_image(encode(fundus_phantom(size=32, seed=1)))
    assert image.input_fingerprint() == image.input_fingerprint()
    assert image.input_fingerprint() != other.input_fingerprint()


def test_normalization_follows_the_declared_spec() -> None:
    """A model's declared mean/std must be applied exactly."""
    pixels = np.full((4, 4, 3), 255, dtype=np.uint8)
    spec = InputSpec(normalization_mean=(0.5, 0.5, 0.5), normalization_std=(0.5, 0.5, 0.5))
    chw = normalize(pixels, spec)
    assert chw.shape == (3, 4, 4)
    np.testing.assert_allclose(chw, 1.0)


def test_grayscale_spec_collapses_channels() -> None:
    chw = normalize(fundus_phantom(size=8), InputSpec(channels=1, color_space="gray"))
    assert chw.shape == (1, 8, 8)


def test_prepare_for_model_resizes_to_declared_size() -> None:
    image = decode_image(encode(fundus_phantom(size=64)))
    chw = prepare_for_model(image, InputSpec(image_size=(24, 32)))
    assert chw.shape == (3, 24, 32)


def test_resize_nearest_preserves_label_values() -> None:
    """Masks must never be interpolated into invalid class indices."""
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 255
    resized = resize(mask, (16, 16), nearest=True)
    assert set(np.unique(resized)) == {0, 255}


def test_fov_mask_and_crop() -> None:
    image = fundus_phantom(size=128)
    mask = estimate_fov_mask(image)
    assert 0.5 < mask.mean() < 0.85  # a centred circular field
    cropped = crop_to_fov(image)
    assert cropped.shape[0] < image.shape[0]


def test_gaussian_blur_reduces_high_frequency_energy() -> None:
    gray = to_grayscale(fundus_phantom(size=96))
    assert gaussian_blur(gray, 2.0).var() < gray.var()


def test_encode_png_is_a_valid_png() -> None:
    assert encode_png(np.zeros((4, 4), np.uint8))[:8] == b"\x89PNG\r\n\x1a\n"


def test_softmax_and_sigmoid_are_numerically_stable() -> None:
    np.testing.assert_allclose(softmax(np.array([1000.0, 1000.0])), [0.5, 0.5])
    assert np.all(np.isfinite(sigmoid(np.array([-1e6, 0.0, 1e6]))))


def test_scores_to_predictions_ranks_and_validates() -> None:
    predictions = scores_to_predictions(np.array([0.1, 0.7, 0.2]), ["a", "b", "c"])
    assert [p.label for p in predictions] == ["b", "c", "a"]
    assert [p.rank for p in predictions] == [0, 1, 2]
    with pytest.raises(ValueError):
        scores_to_predictions(np.array([0.5, 0.5]), ["only-one-label"])


def test_mask_statistics_use_the_device_scale() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[0:5, 0:4] = True
    stats = binary_mask_stats(mask, {"lateral": 10.0, "axial": 10.0})
    assert stats["area_px"] == 20
    assert stats["area_ratio"] == pytest.approx(0.2)
    assert stats["area_mm2"] == pytest.approx(20 * 100 / 1e6)
    # Without a scale, no physical area is invented.
    assert "area_mm2" not in binary_mask_stats(mask)


def test_mask_payload_encodes_png_and_area() -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[1:4, 1:4] = True
    payload = mask_to_payload(mask, "lesion")
    assert payload.label == "lesion"
    assert payload.area_px == 9
    assert payload.data[:8] == b"\x89PNG\r\n\x1a\n"


def test_label_map_splits_into_per_class_masks_skipping_background() -> None:
    label_map = np.array([[0, 1], [2, 2]])
    payloads = label_mask_to_payloads(label_map, ["background", "irf", "srf"])
    assert [p.label for p in payloads] == ["irf", "srf"]
    assert [p.area_px for p in payloads] == [1.0, 2.0]


def test_normalize_heatmap_handles_a_flat_map() -> None:
    np.testing.assert_array_equal(normalize_heatmap(np.ones((3, 3))), np.zeros((3, 3)))
