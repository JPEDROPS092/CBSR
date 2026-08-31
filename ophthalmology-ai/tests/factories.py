"""Synthetic image builders for tests.

These are *phantoms*, not clinical data: geometric approximations of a fundus
photograph and an OCT B-scan with known ground truth, which is what lets the
tests assert on measured values.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image as PILImage


def encode(array: np.ndarray, fmt: str = "PNG") -> bytes:
    """Encode an array as image bytes."""
    mode = "L" if array.ndim == 2 else "RGB"
    buffer = io.BytesIO()
    PILImage.fromarray(array.astype(np.uint8), mode=mode).save(buffer, format=fmt)
    return buffer.getvalue()


def fundus_phantom(size: int = 384, *, seed: int = 0, blur: bool = False) -> np.ndarray:
    """A colour disc with retinal texture on a black background."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    disc = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (0.45 * size) ** 2
    texture = 120 + 40 * np.sin(xx / 7.0) + 30 * rng.standard_normal((size, size))
    if blur:
        texture = 120 + 40 * np.sin(xx / 60.0)
    texture = np.clip(texture, 0, 255)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[..., 0] = np.where(disc, texture, 0)
    image[..., 1] = np.where(disc, texture * 0.45, 0)
    image[..., 2] = np.where(disc, texture * 0.25, 0)
    return image


def oct_phantom(
    height: int = 496,
    width: int = 512,
    *,
    seed: int = 3,
    retina_thickness_px: int = 80,
    noise: float = 7.0,
) -> tuple[np.ndarray, int]:
    """An OCT-like B-scan with a known ILM/RPE separation.

    Returns the image and the ground-truth ILM-to-RPE-peak distance in pixels,
    which the layer model is expected to recover.
    """
    rng = np.random.default_rng(seed)
    image = np.zeros((height, width), dtype=np.float32)
    inner = retina_thickness_px - 9
    for x in range(width):
        top = int(height * 0.36 + 12 * np.sin(x / 60.0))
        image[top : top + 5, x] = 130
        image[top + 5 : top + inner, x] = 55
        image[top + inner : top + inner + 9, x] = 210
    image = np.clip(image + rng.normal(0, noise, (height, width)), 0, 255)
    # The RPE is detected at the centre of the bright band.
    truth = inner + 4
    return image.astype(np.uint8), truth


def noise_image(height: int = 256, width: int = 256, *, seed: int = 9) -> np.ndarray:
    """A dark noise field: fails quality control for either modality."""
    rng = np.random.default_rng(seed)
    return (rng.random((height, width)) * 40).astype(np.uint8)
