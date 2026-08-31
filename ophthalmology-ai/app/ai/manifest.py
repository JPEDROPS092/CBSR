"""Sidecar manifests for externally supplied models.

The platform ships no weights. To install a third-party model an operator
drops two files under ``MODEL_DIR``:

1. the checkpoint (``.pt``/``.pth``/``.safetensors``/``.onnx``);
2. a JSON manifest next to it describing **exactly** how that checkpoint must
   be fed and how its output is interpreted.

The manifest is mandatory on purpose. Input size, colour space and
normalization are properties of the trained weights, not of this codebase;
guessing them produces confident, wrong numbers. If a model's documentation
does not state them, the model must not be run in this platform.

Example (``models/fundus/dr_grading_v1.json``)::

    {
      "model_id": "dr_grading_v1",
      "name": "Diabetic Retinopathy Grading",
      "version": "1.0.0",
      "modality": "fundus",
      "task": "classification",
      "framework": "onnx",
      "weights_file": "dr_grading_v1.onnx",
      "input": {
        "image_size": [512, 512],
        "channels": 3,
        "color_space": "rgb",
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
        "scale_to_unit_interval": true
      },
      "output": {
        "labels": ["no_dr", "mild_npdr", "moderate_npdr", "severe_npdr", "pdr"],
        "activation": "softmax"
      },
      "license": {"name": "CC BY-NC 4.0", "commercial_use": "prohibited"},
      "limitations": "Trained on 45-degree macula-centred images only."
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from app.ai.results import (
    InputSpec,
    ModelLicense,
    ModelMetadata,
    OutputSpec,
    WeightSpec,
)
from app.core.config import get_settings
from app.core.enums import EvidenceLevel, Framework, Modality, ModelStatus, TaskType
from app.core.exceptions import ValidationError
from app.core.logging import get_logger

logger = get_logger(__name__)

MANIFEST_SUFFIX = ".json"


class ManifestInput(BaseModel):
    """Preprocessing declared by the model's author."""

    image_size: tuple[int, int] | None = None
    channels: int = 3
    color_space: Literal["rgb", "bgr", "gray"] = "rgb"
    normalization_mean: tuple[float, ...] | None = None
    normalization_std: tuple[float, ...] | None = None
    scale_to_unit_interval: bool = True
    notes: str = ""

    def to_spec(self) -> InputSpec:
        return InputSpec(**self.model_dump())


class ManifestOutput(BaseModel):
    """Output interpretation declared by the model's author."""

    labels: list[str] = Field(default_factory=list)
    segmentation_classes: list[str] = Field(default_factory=list)
    activation: Literal["softmax", "sigmoid", "none"] = "softmax"
    threshold: float = 0.5
    units: dict[str, str] = Field(default_factory=dict)
    notes: str = ""

    def to_spec(self) -> OutputSpec:
        return OutputSpec(
            labels=self.labels,
            segmentation_classes=self.segmentation_classes,
            units=self.units,
            notes=self.notes,
        )


class ManifestLicense(BaseModel):
    """License and provenance, copied from the model's distribution."""

    name: str = "unknown"
    url: str | None = None
    source_url: str | None = None
    dataset: str | None = None
    dataset_license: str | None = None
    commercial_use: Literal["allowed", "restricted", "prohibited", "unknown"] = "unknown"
    citation: str | None = None
    restrictions: str | None = None

    def to_license(self) -> ModelLicense:
        return ModelLicense(**self.model_dump())


class ExternalModelSpec(BaseModel):
    """A complete description of an externally supplied model."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    name: str
    version: str = "1.0.0"
    modality: Modality
    task: TaskType
    framework: Framework
    weights_file: str
    weights_sha256: str | None = None
    description: str = ""
    evidence_level: EvidenceLevel = EvidenceLevel.RESEARCH
    input: ManifestInput = Field(default_factory=ManifestInput)
    output: ManifestOutput = Field(default_factory=ManifestOutput)
    license: ManifestLicense = Field(default_factory=ManifestLicense)
    reported_metrics: dict[str, Any] = Field(default_factory=dict)
    limitations: str = ""
    supports_explainability: bool = False
    #: Directory of the manifest relative to MODEL_DIR; filled in on load.
    subdir: str = ""

    def to_metadata(self) -> ModelMetadata:
        """Build registry metadata from this manifest."""
        return ModelMetadata(
            model_id=self.model_id,
            name=self.name,
            version=self.version,
            modality=self.modality,
            task=self.task,
            framework=self.framework,
            evidence_level=self.evidence_level,
            description=self.description,
            status=ModelStatus.ACTIVE,
            input_spec=self.input.to_spec(),
            output_spec=self.output.to_spec(),
            weights=WeightSpec(
                filename=self.weights_file,
                subdir=self.subdir,
                sha256=self.weights_sha256,
                format=_weight_format(self.weights_file),
                expects=f"Checkpoint for {self.name} ({self.framework}).",
            ),
            license=self.license.to_license(),
            supports_explainability=self.supports_explainability,
            reported_metrics=self.reported_metrics,
            limitations=self.limitations,
        )


def _weight_format(filename: str) -> str:
    suffix = Path(filename).suffix.lower().lstrip(".")
    return suffix if suffix in {"pt", "pth", "safetensors", "onnx"} else "pt"


def load_manifest(path: Path, *, model_dir: Path | None = None) -> ExternalModelSpec:
    """Parse a manifest file.

    Raises:
        ValidationError: when the file is not valid JSON or does not match the
            manifest schema.
    """
    root = Path(model_dir or get_settings().MODEL_DIR).resolve()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(
            "Model manifest could not be read.", details={"path": str(path)}
        ) from exc
    try:
        spec = ExternalModelSpec.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValidationError(
            "Model manifest does not match the expected schema.",
            details={"path": str(path), "errors": exc.errors(include_url=False)},
        ) from exc
    try:
        relative = path.resolve().parent.relative_to(root)
        spec.subdir = "" if str(relative) == "." else str(relative)
    except ValueError:
        spec.subdir = ""
    return spec


def discover_manifests(model_dir: Path | None = None) -> list[ExternalModelSpec]:
    """Find every valid manifest under ``MODEL_DIR``.

    Invalid manifests are logged and skipped: one malformed file must not stop
    the platform from starting.
    """
    root = Path(model_dir or get_settings().MODEL_DIR)
    if not root.is_dir():
        return []
    specs: list[ExternalModelSpec] = []
    for path in sorted(root.rglob(f"*{MANIFEST_SUFFIX}")):
        try:
            specs.append(load_manifest(path, model_dir=root))
        except ValidationError as exc:
            logger.warning(
                "model_manifest_invalid",
                extra={"path": str(path), "reason": exc.message},
            )
    return specs


def find_manifest_for(model_id: str, *, model_dir: Path | None = None) -> ExternalModelSpec | None:
    """Locate the sidecar manifest of a named adapter, if the operator installed one."""
    for spec in discover_manifests(model_dir):
        if spec.model_id == model_id:
            return spec
    return None
