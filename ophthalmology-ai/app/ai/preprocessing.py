"""Image decoding and preprocessing primitives.

Everything here works on plain NumPy arrays and Pillow, so the API process can
validate and inspect images without a deep-learning runtime installed.

Preprocessing is deliberately *explicit*: a model adapter states its
:class:`~app.ai.results.InputSpec` (size, colour space, normalization) and this
module applies exactly that. Silently guessing a normalization is the fastest
way to make a published checkpoint produce plausible-looking nonsense.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps, ImageSequence

from app.ai.results import InputSpec
from app.core.enums import Modality
from app.core.exceptions import ValidationError

# Pillow refuses very large images by default (decompression-bomb guard). Raise
# it to a value that still covers wide-field fundus and OCT volumes.
PILImage.MAX_IMAGE_PIXELS = 300_000_000


@dataclass(slots=True)
class ExamImage:
    """A decoded exam image handed to models.

    Attributes:
        pixels: ``uint8`` array, ``(H, W)`` for grayscale or ``(H, W, 3)`` RGB.
        modality: Which modality the image belongs to.
        image_id: Database identifier, used for artifact naming and logging.
        pixel_spacing_um: Physical scale reported by the device, e.g.
            ``{"axial": 3.87, "lateral": 11.7}`` for OCT. Absent means
            measurements can only be reported in pixels.
        checksum_sha256: Hash of the *source bytes*, recorded per inference for
            reproducibility.
        frame_index: Position inside a multi-frame series (OCT B-scans).
    """

    pixels: np.ndarray
    modality: Modality = Modality.OTHER
    image_id: str | None = None
    pixel_spacing_um: dict[str, float] = field(default_factory=dict)
    content_type: str = "image/png"
    checksum_sha256: str = ""
    frame_index: int = 0
    frame_count: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def is_color(self) -> bool:
        return self.pixels.ndim == 3 and self.pixels.shape[2] == 3

    def to_gray(self) -> np.ndarray:
        """Luminance as ``float32`` in ``[0, 1]``."""
        return to_grayscale(self.pixels)

    def to_rgb(self) -> np.ndarray:
        """``uint8`` RGB view, replicating channels for grayscale sources."""
        if self.is_color:
            return self.pixels
        return np.repeat(self.pixels[:, :, None], 3, axis=2)

    def input_fingerprint(self) -> str:
        """Stable hash of the exact pixels fed to a model.

        Recorded on every :class:`~app.database.models.ModelRun` so a result can
        be reproduced later even if the stored file is re-encoded.
        """
        digest = hashlib.sha256()
        digest.update(str(self.pixels.shape).encode())
        digest.update(np.ascontiguousarray(self.pixels).tobytes())
        return digest.hexdigest()


def decode_image(
    data: bytes,
    *,
    modality: Modality = Modality.OTHER,
    image_id: str | None = None,
    content_type: str = "image/png",
    pixel_spacing_um: dict[str, float] | None = None,
    frame_index: int = 0,
) -> ExamImage:
    """Decode raw bytes into an :class:`ExamImage`.

    Applies EXIF transposition and converts palette/16-bit sources to 8-bit
    RGB or grayscale.

    Raises:
        ValidationError: when the payload is not a decodable image.
    """
    try:
        with PILImage.open(io.BytesIO(data)) as handle:
            frame_count = getattr(handle, "n_frames", 1)
            if frame_index:
                if frame_index >= frame_count:
                    raise ValidationError(
                        "Requested frame does not exist in this image.",
                        details={"frame_index": frame_index, "frame_count": frame_count},
                    )
                handle.seek(frame_index)
            pixels = _pil_to_array(handle)
    except ValidationError:
        raise
    except Exception as exc:  # noqa: BLE001 - Pillow raises many decoder errors
        raise ValidationError("File could not be decoded as an image.") from exc

    return ExamImage(
        pixels=pixels,
        modality=modality,
        image_id=image_id,
        pixel_spacing_um=pixel_spacing_um or {},
        content_type=content_type,
        checksum_sha256=hashlib.sha256(data).hexdigest(),
        frame_index=frame_index,
        frame_count=int(frame_count),
    )


def decode_series(
    data: bytes,
    *,
    modality: Modality = Modality.OCT,
    image_id: str | None = None,
    content_type: str = "image/tiff",
    pixel_spacing_um: dict[str, float] | None = None,
    max_frames: int = 128,
) -> list[ExamImage]:
    """Decode a multi-frame file (an OCT B-scan series) into frames."""
    checksum = hashlib.sha256(data).hexdigest()
    frames: list[ExamImage] = []
    try:
        with PILImage.open(io.BytesIO(data)) as handle:
            total = getattr(handle, "n_frames", 1)
            for index, frame in enumerate(ImageSequence.Iterator(handle)):
                if index >= max_frames:
                    break
                frames.append(
                    ExamImage(
                        pixels=_pil_to_array(frame),
                        modality=modality,
                        image_id=image_id,
                        pixel_spacing_um=pixel_spacing_um or {},
                        content_type=content_type,
                        checksum_sha256=checksum,
                        frame_index=index,
                        frame_count=int(total),
                    )
                )
    except Exception as exc:  # noqa: BLE001
        raise ValidationError("File could not be decoded as an image series.") from exc
    if not frames:
        raise ValidationError("Image series contains no frames.")
    return frames


def _pil_to_array(handle: PILImage.Image) -> np.ndarray:
    """Convert a Pillow image to an 8-bit NumPy array."""
    image = ImageOps.exif_transpose(handle) or handle
    if image.mode in ("I;16", "I", "F"):
        # 16-bit OCT exports: rescale to 8 bits using the actual dynamic range
        # rather than clipping, which would flatten the retina.
        raw = np.asarray(image, dtype=np.float32)
        lo, hi = float(raw.min()), float(raw.max())
        scaled = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
        return (scaled * 255.0).astype(np.uint8)
    if image.mode == "L":
        return np.asarray(image, dtype=np.uint8)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def encode_png(array: np.ndarray) -> bytes:
    """Encode a ``uint8`` array as PNG bytes."""
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    mode = "L" if array.ndim == 2 else "RGB"
    buffer = io.BytesIO()
    PILImage.fromarray(array, mode=mode).save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def to_grayscale(pixels: np.ndarray) -> np.ndarray:
    """Rec. 601 luminance as ``float32`` in ``[0, 1]``."""
    array = pixels.astype(np.float32)
    if array.ndim == 3:
        array = array @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return array / 255.0


def resize(pixels: np.ndarray, size: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    """Resize to ``(height, width)``.

    Uses nearest-neighbour for label masks (``nearest=True``) so class indices
    are never blended, and bilinear for intensity images.
    """
    height, width = size
    mode = "L" if pixels.ndim == 2 else "RGB"
    source = pixels if pixels.dtype == np.uint8 else np.clip(pixels, 0, 255).astype(np.uint8)
    resample = PILImage.Resampling.NEAREST if nearest else PILImage.Resampling.BILINEAR
    resized = PILImage.fromarray(source, mode=mode).resize((width, height), resample)
    return np.asarray(resized, dtype=np.uint8)


def normalize(pixels: np.ndarray, spec: InputSpec) -> np.ndarray:
    """Apply a model's declared normalization, returning ``float32`` CHW."""
    array = pixels.astype(np.float32)
    if spec.color_space == "gray" and array.ndim == 3:
        array = (to_grayscale(pixels) * 255.0)[:, :, None]
    elif spec.color_space == "bgr" and array.ndim == 3:
        array = array[:, :, ::-1]
    elif array.ndim == 2:
        array = array[:, :, None]
    if spec.channels == 3 and array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)

    if spec.scale_to_unit_interval:
        array = array / 255.0
    if spec.normalization_mean is not None:
        mean = np.asarray(spec.normalization_mean, dtype=np.float32)
        array = array - mean.reshape(1, 1, -1)
    if spec.normalization_std is not None:
        std = np.asarray(spec.normalization_std, dtype=np.float32)
        array = array / std.reshape(1, 1, -1)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def prepare_for_model(image: ExamImage, spec: InputSpec) -> np.ndarray:
    """Full preprocessing chain: resize (if declared) then normalize to CHW."""
    pixels = image.pixels
    if spec.image_size is not None:
        pixels = resize(pixels, spec.image_size)
    return normalize(pixels, spec)


def estimate_fov_mask(pixels: np.ndarray, *, threshold: float = 0.06) -> np.ndarray:
    """Boolean mask of the illuminated field of view of a fundus photograph.

    Fundus cameras produce a bright circular field on a black background; the
    dark border is not retina and must be excluded from exposure and contrast
    statistics, otherwise every image looks under-exposed.
    """
    gray = to_grayscale(pixels)
    return gray > threshold


def crop_to_fov(pixels: np.ndarray, *, threshold: float = 0.06, padding: int = 4) -> np.ndarray:
    """Crop away the black border around a fundus field of view."""
    mask = estimate_fov_mask(pixels, threshold=threshold)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return pixels
    top = max(int(rows[0]) - padding, 0)
    bottom = min(int(rows[-1]) + 1 + padding, pixels.shape[0])
    left = max(int(cols[0]) - padding, 0)
    right = min(int(cols[-1]) + 1 + padding, pixels.shape[1])
    return pixels[top:bottom, left:right]


def convolve_1d(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """Convolve a 2-D image with a 1-D kernel along ``axis``.

    Vectorized with a sliding window view: a per-row Python loop is an order of
    magnitude slower and this runs on every quality check.
    """
    radius = kernel.size // 2
    padding = [(0, 0), (0, 0)]
    padding[axis] = (radius, radius)
    padded = np.pad(image, padding, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=axis)
    return windows @ kernel.astype(np.float32)


def gaussian_blur(gray: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Separable Gaussian blur on a 2-D ``float32`` image (NumPy only)."""
    radius = max(int(3 * sigma), 1)
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return convolve_1d(convolve_1d(gray.astype(np.float32), kernel, axis=1), kernel, axis=0)


def laplacian(gray: np.ndarray) -> np.ndarray:
    """4-neighbour Laplacian, the classic sharpness operator."""
    padded = np.pad(gray, 1, mode="reflect")
    return (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )


def sobel_gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sobel gradients ``(gy, gx)`` of a 2-D ``float32`` image."""
    padded = np.pad(gray, 1, mode="reflect")
    gx = (
        -padded[:-2, :-2]
        + padded[:-2, 2:]
        - 2.0 * padded[1:-1, :-2]
        + 2.0 * padded[1:-1, 2:]
        - padded[2:, :-2]
        + padded[2:, 2:]
    )
    gy = (
        -padded[:-2, :-2]
        - 2.0 * padded[:-2, 1:-1]
        - padded[:-2, 2:]
        + padded[2:, :-2]
        + 2.0 * padded[2:, 1:-1]
        + padded[2:, 2:]
    )
    return gy, gx
