"""Checkpoint resolution.

The platform ships **no** model weights. Each adapter declares a
:class:`~app.ai.results.WeightSpec`; this module locates the file under
``MODEL_DIR``, optionally verifies its SHA-256, and - when it is missing -
produces an :class:`~app.ai.results.Availability` that tells the operator
exactly which file to place where.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from app.ai.results import Availability, WeightSpec
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def resolve_weight_path(spec: WeightSpec, *, model_dir: Path | None = None) -> Path:
    """Absolute path where ``spec``'s checkpoint is expected."""
    root = Path(model_dir or get_settings().MODEL_DIR)
    return (root / spec.subdir / spec.filename) if spec.subdir else (root / spec.filename)


def file_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Stream a file's SHA-256 (checkpoints are too large to read at once)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def check_weights(spec: WeightSpec, *, model_dir: Path | None = None) -> Availability:
    """Report whether a declared checkpoint is present and intact."""
    path = resolve_weight_path(spec, model_dir=model_dir)
    if not path.is_file():
        return Availability(
            available=False,
            reason="Model weights are not present in this deployment.",
            missing=[str(path)],
            remediation=(
                f"Place the checkpoint at {path}. Expected format: {spec.format}. "
                f"{spec.expects}".strip()
            ),
        )
    if spec.sha256:
        actual = file_sha256(path)
        if actual != spec.sha256:
            return Availability(
                available=False,
                reason="Checkpoint checksum does not match the declared value.",
                missing=[str(path)],
                remediation=(
                    "The file at this path is not the expected checkpoint "
                    f"(expected sha256 {spec.sha256}, found {actual}). Replace it or "
                    "update the adapter's WeightSpec if you intentionally changed weights."
                ),
            )
    return Availability(available=True)


def runtime_availability(*, needs_torch: bool = False, needs_onnx: bool = False) -> Availability:
    """Check that the runtime a model needs is installed."""
    from app.ai.devices import onnxruntime_available, torch_available

    missing: list[str] = []
    if needs_torch and not torch_available():
        missing.append("torch")
    if needs_onnx and not onnxruntime_available():
        missing.append("onnxruntime")
    if missing:
        return Availability(
            available=False,
            reason="Required inference runtime is not installed.",
            missing=missing,
            remediation="Install the ML extra: pip install -e '.[ml]'",
        )
    return Availability(available=True)
