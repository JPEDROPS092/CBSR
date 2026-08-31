"""Placeholders for catalogued models whose weights are not installed.

The platform advertises a catalogue of ophthalmology tasks. For the ones that
require third-party weights, a placeholder is registered so that:

* ``GET /api/v1/models`` shows the task exists and is *not* runnable here;
* the response states precisely which files to install and where;
* an analysis that requests the model fails fast with the same explanation
  instead of silently returning nothing.

A placeholder never produces a prediction. Fabricating one - even a neutral
one - would put an invented number in a medical result.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NoReturn

from app.ai.base import BaseOphthalmologyModel
from app.ai.preprocessing import ExamImage
from app.ai.results import Availability, ModelMetadata, ModelResult
from app.core.config import get_settings
from app.core.enums import ModelStatus
from app.core.exceptions import ModelUnavailableError


class PlaceholderModel(BaseOphthalmologyModel):
    """A catalogued model whose checkpoint and manifest are not present."""

    def __init__(self, metadata: ModelMetadata, *, subdir: str, manifest_name: str) -> None:
        super().__init__()
        self.metadata = metadata.model_copy(update={"status": ModelStatus.UNAVAILABLE})
        self.subdir = subdir
        self.manifest_name = manifest_name

    @property
    def expected_manifest_path(self) -> Path:
        """Where the operator must place the sidecar manifest."""
        root = Path(get_settings().MODEL_DIR)
        return root / self.subdir / self.manifest_name if self.subdir else root / self.manifest_name

    def availability(self) -> Availability:
        """Always unavailable, with actionable remediation."""
        return Availability(
            available=False,
            reason="No weights or manifest are installed for this catalogued model.",
            missing=[str(self.expected_manifest_path)],
            remediation=(
                f"Install a model for this task by placing its checkpoint and a manifest "
                f"at {self.expected_manifest_path} (see docs/MODEL_REGISTRY.md for the "
                "manifest schema). The manifest must state the checkpoint's input size, "
                "colour space, normalization, output labels and license."
            ),
        )

    def _unavailable(self) -> NoReturn:
        availability = self.availability()
        raise ModelUnavailableError(
            availability.reason or "Model is unavailable.",
            details=availability.model_dump(exclude_none=True),
        )

    def predict(self, prepared: Any) -> NoReturn:
        self._unavailable()

    def postprocess(self, output: Any, image: ExamImage) -> ModelResult:
        self._unavailable()
