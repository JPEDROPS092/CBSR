"""Turning raw model outputs into the platform's standard result objects."""

from __future__ import annotations

import numpy as np

from app.ai.preprocessing import encode_png, resize
from app.ai.results import BoundingBox, DetectionItem, MaskPayload, PredictionItem


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60))),
        np.exp(np.clip(logits, -60, 60)) / (1.0 + np.exp(np.clip(logits, -60, 60))),
    )


def scores_to_predictions(
    scores: np.ndarray, labels: list[str], *, top_k: int | None = None
) -> list[PredictionItem]:
    """Convert a score vector into ranked :class:`PredictionItem` objects.

    Raises:
        ValueError: if the score vector and the label list disagree in length,
            which almost always means the checkpoint does not match the
            adapter's declared ``OutputSpec``.
    """
    flat = np.asarray(scores, dtype=np.float32).reshape(-1)
    if flat.size != len(labels):
        raise ValueError(
            f"Model produced {flat.size} scores but {len(labels)} labels are declared."
        )
    order = np.argsort(flat)[::-1]
    if top_k is not None:
        order = order[:top_k]
    return [
        PredictionItem(
            label=labels[int(i)], score=float(np.clip(flat[int(i)], 0.0, 1.0)), rank=rank
        )
        for rank, i in enumerate(order)
    ]


def binary_mask_stats(mask: np.ndarray, pixel_spacing_um: dict[str, float] | None = None) -> dict:
    """Area statistics for a boolean mask, in pixels and (when known) mm²."""
    boolean = mask.astype(bool)
    area_px = float(boolean.sum())
    total = float(boolean.size) or 1.0
    stats: dict[str, float] = {"area_px": area_px, "area_ratio": area_px / total}
    spacing = pixel_spacing_um or {}
    lateral = spacing.get("lateral") or spacing.get("x")
    axial = spacing.get("axial") or spacing.get("y")
    if lateral and axial:
        # µm² -> mm²
        stats["area_mm2"] = area_px * float(lateral) * float(axial) / 1e6
    return stats


def mask_to_payload(
    mask: np.ndarray,
    label: str,
    *,
    pixel_spacing_um: dict[str, float] | None = None,
    measurements: dict | None = None,
    target_size: tuple[int, int] | None = None,
) -> MaskPayload:
    """Encode a boolean/label mask as a PNG payload with area statistics.

    Masks are stored as 8-bit PNGs where pixel value ``255`` marks the
    structure (or the class index for multi-class masks), so any image viewer
    can display them and any downstream tool can threshold them.
    """
    boolean = mask.astype(bool)
    stats = binary_mask_stats(boolean, pixel_spacing_um)
    encoded = (boolean.astype(np.uint8)) * 255
    if target_size is not None and encoded.shape != target_size:
        encoded = resize(encoded, target_size, nearest=True)
    combined = dict(measurements or {})
    if "area_mm2" in stats:
        combined.setdefault("area_mm2", stats["area_mm2"])
    return MaskPayload(
        label=label,
        data=encode_png(encoded),
        area_px=stats["area_px"],
        area_ratio=stats["area_ratio"],
        measurements=combined,
    )


def label_mask_to_payloads(
    label_map: np.ndarray,
    class_names: list[str],
    *,
    pixel_spacing_um: dict[str, float] | None = None,
    skip_background: bool = True,
) -> list[MaskPayload]:
    """Split a multi-class label map into one payload per class."""
    payloads: list[MaskPayload] = []
    for index, name in enumerate(class_names):
        if skip_background and index == 0:
            continue
        mask = label_map == index
        if not mask.any():
            continue
        payloads.append(mask_to_payload(mask, name, pixel_spacing_um=pixel_spacing_um))
    return payloads


def boxes_to_detections(
    boxes: np.ndarray, scores: np.ndarray, labels: list[str]
) -> list[DetectionItem]:
    """Convert ``(N, 4)`` xyxy boxes into :class:`DetectionItem` objects."""
    detections: list[DetectionItem] = []
    for box, score, label in zip(boxes, scores, labels, strict=True):
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        detections.append(
            DetectionItem(
                label=label,
                score=float(np.clip(score, 0.0, 1.0)),
                box=BoundingBox(x=x1, y=y1, width=max(x2 - x1, 0), height=max(y2 - y1, 0)),
            )
        )
    return detections


# --------------------------------------------------------------------------- #
# Visual overlays
# --------------------------------------------------------------------------- #
def _turbo_like_colormap(values: np.ndarray) -> np.ndarray:
    """Map ``[0, 1]`` values to an RGB heat colormap (blue -> red).

    Implemented in NumPy so overlays do not require matplotlib on GPU workers.
    """
    v = np.clip(values, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0.0, 1.0)
    green = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0.0, 1.0)
    blue = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0.0, 1.0)
    return (np.stack([red, green, blue], axis=-1) * 255.0).astype(np.uint8)


def heatmap_overlay(
    base_rgb: np.ndarray, heatmap: np.ndarray, *, alpha: float = 0.45
) -> np.ndarray:
    """Blend a normalized heatmap over an RGB image."""
    if heatmap.shape != base_rgb.shape[:2]:
        heatmap = (
            resize((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), base_rgb.shape[:2]).astype(
                np.float32
            )
            / 255.0
        )
    colored = _turbo_like_colormap(heatmap).astype(np.float32)
    blended = (1.0 - alpha) * base_rgb.astype(np.float32) + alpha * colored
    return np.clip(blended, 0, 255).astype(np.uint8)


def mask_overlay(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a boolean mask over an RGB image for visual review."""
    boolean = mask.astype(bool)
    if boolean.shape != base_rgb.shape[:2]:
        boolean = resize(boolean.astype(np.uint8) * 255, base_rgb.shape[:2], nearest=True) > 127
    overlay = base_rgb.astype(np.float32).copy()
    tint = np.asarray(color, dtype=np.float32)
    overlay[boolean] = (1.0 - alpha) * overlay[boolean] + alpha * tint
    return np.clip(overlay, 0, 255).astype(np.uint8)


def normalize_heatmap(raw: np.ndarray) -> np.ndarray:
    """Rescale an arbitrary activation map to ``[0, 1]``."""
    array = np.asarray(raw, dtype=np.float32)
    low, high = float(array.min()), float(array.max())
    if high - low < 1e-12:
        return np.zeros_like(array)
    return (array - low) / (high - low)
