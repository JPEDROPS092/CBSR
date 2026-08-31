"""Image-quality metrics.

Deterministic, dependency-light measurements shared by the fundus and OCT
quality models. Every function returns a raw physical quantity; mapping those
quantities to a 0-1 score happens in the models, where the thresholds are
declared and documented as heuristics.
"""

from __future__ import annotations

import numpy as np

from app.ai.preprocessing import (
    estimate_fov_mask,
    gaussian_blur,
    laplacian,
    sobel_gradients,
    to_grayscale,
)


def _masked(values: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Flatten ``values`` restricted to ``mask`` (falling back to everything)."""
    if mask is None:
        return values.reshape(-1)
    selected = values[mask]
    return selected if selected.size else values.reshape(-1)


def sharpness(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Variance of the Laplacian - the standard no-reference blur measure.

    Higher means more high-frequency detail. Values are comparable only within
    a modality and roughly within a resolution, which is why the models
    normalize them against modality-specific references.
    """
    response = laplacian(gray)
    return float(np.var(_masked(response, mask)))


def edge_density(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean Sobel gradient magnitude; a blur measure less sensitive to noise."""
    gy, gx = sobel_gradients(gray)
    magnitude = np.hypot(gy, gx)
    return float(np.mean(_masked(magnitude, mask)))


def exposure(gray: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float]:
    """Brightness distribution and clipping ratios."""
    values = _masked(gray, mask)
    return {
        "mean_luminance": float(np.mean(values)),
        "median_luminance": float(np.median(values)),
        "p01": float(np.percentile(values, 1)),
        "p99": float(np.percentile(values, 99)),
        "clipped_dark_ratio": float(np.mean(values < 0.02)),
        "clipped_bright_ratio": float(np.mean(values > 0.98)),
    }


def contrast(gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Root-mean-square contrast (standard deviation of luminance)."""
    return float(np.std(_masked(gray, mask)))


def illumination_uniformity(
    gray: np.ndarray, mask: np.ndarray | None = None, grid: int = 4
) -> float:
    """Uniformity of illumination across a ``grid x grid`` tiling.

    Returns ``min(tile mean) / max(tile mean)`` in ``[0, 1]``: 1 is perfectly
    even, low values indicate vignetting or a partially illuminated field.
    """
    height, width = gray.shape[:2]
    tile_means: list[float] = []
    for row in range(grid):
        for col in range(grid):
            y0, y1 = row * height // grid, (row + 1) * height // grid
            x0, x1 = col * width // grid, (col + 1) * width // grid
            tile = gray[y0:y1, x0:x1]
            tile_mask = None if mask is None else mask[y0:y1, x0:x1]
            if tile_mask is not None and tile_mask.sum() < tile.size * 0.2:
                continue  # Mostly outside the field of view.
            tile_means.append(float(np.mean(_masked(tile, tile_mask))))
    if len(tile_means) < 2:
        return 1.0
    lowest, highest = min(tile_means), max(tile_means)
    return float(lowest / highest) if highest > 1e-6 else 0.0


def field_of_view(pixels: np.ndarray) -> dict[str, float]:
    """Geometry of the illuminated field in a fundus photograph.

    Returns the fraction of the frame that is illuminated, how far the field's
    centroid sits from the frame centre, and how much of the field touches the
    frame border (a proxy for a cropped/incomplete acquisition).
    """
    mask = estimate_fov_mask(pixels)
    height, width = mask.shape
    total = float(mask.size)
    covered = float(mask.sum())
    if covered == 0:
        return {
            "fov_ratio": 0.0,
            "centroid_offset": 1.0,
            "border_contact_ratio": 0.0,
        }
    ys, xs = np.nonzero(mask)
    centroid_y, centroid_x = float(ys.mean()), float(xs.mean())
    offset = np.hypot(centroid_y - height / 2, centroid_x - width / 2)
    max_offset = np.hypot(height / 2, width / 2)
    border = (
        int(mask[0, :].sum())
        + int(mask[-1, :].sum())
        + int(mask[:, 0].sum())
        + int(mask[:, -1].sum())
    )
    perimeter = 2 * (height + width)
    return {
        "fov_ratio": covered / total,
        "centroid_offset": float(offset / max_offset),
        "border_contact_ratio": float(border / perimeter),
    }


def colorfulness(pixels: np.ndarray) -> float:
    """Hasler & Süsstrunk (2003) colourfulness metric.

    Used to detect a modality mismatch: fundus photographs are strongly
    coloured (typically > 15), OCT B-scans are grayscale (near 0).
    """
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        return 0.0
    array = pixels.astype(np.float32)
    rg = array[:, :, 0] - array[:, :, 1]
    yb = 0.5 * (array[:, :, 0] + array[:, :, 1]) - array[:, :, 2]
    std = np.hypot(float(np.std(rg)), float(np.std(yb)))
    mean = np.hypot(float(np.mean(rg)), float(np.mean(yb)))
    return float(std + 0.3 * mean)


def signal_to_noise(gray: np.ndarray, *, background_percentile: float = 25.0) -> dict[str, float]:
    """Crude SNR proxy for OCT B-scans.

    Signal is the mean of the brightest 5 % of pixels (the RPE/photoreceptor
    complex); noise is the standard deviation of the darkest quartile (the
    vitreous, which should be near-empty). Reported in decibels.
    """
    values = gray.reshape(-1)
    signal = float(np.mean(values[values >= np.percentile(values, 95.0)]))
    background = values[values <= np.percentile(values, background_percentile)]
    noise = float(np.std(background))
    if noise < 1e-6:
        return {"signal": signal, "noise": noise, "snr_db": 60.0}
    return {
        "signal": signal,
        "noise": noise,
        "snr_db": float(20.0 * np.log10(max(signal, 1e-6) / noise)),
    }


def horizontal_band_energy(gray: np.ndarray) -> float:
    """Strength of horizontal layering in an image.

    OCT B-scans are dominated by horizontal retinal bands, fundus photographs
    are not. Computed as the ratio of vertical- to horizontal-gradient energy.
    """
    gy, gx = sobel_gradients(gaussian_blur(gray, 1.2))
    vertical = float(np.mean(np.abs(gy)))
    horizontal = float(np.mean(np.abs(gx)))
    return float(vertical / (horizontal + 1e-6))


def retina_row_profile(gray: np.ndarray) -> np.ndarray:
    """Mean intensity per row - the retinal signal profile of a B-scan."""
    return gray.mean(axis=1)


def truncation(gray: np.ndarray, *, threshold: float = 0.25) -> dict[str, float]:
    """How much bright tissue touches the top/bottom edge of a B-scan.

    A retina running off the edge of the frame means the scan is truncated and
    thickness measurements would be wrong.
    """
    top = float(np.mean(gray[0, :] > threshold))
    bottom = float(np.mean(gray[-1, :] > threshold))
    return {"top_contact_ratio": top, "bottom_contact_ratio": bottom}


def prepare_gray(pixels: np.ndarray, *, denoise_sigma: float = 0.0) -> np.ndarray:
    """Grayscale ``float32`` view, optionally denoised."""
    gray = to_grayscale(pixels)
    return gaussian_blur(gray, denoise_sigma) if denoise_sigma > 0 else gray
