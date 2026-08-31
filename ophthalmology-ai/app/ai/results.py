"""Standardized result schema shared by every model.

The contract in this module is what decouples the API, the database and the
report engine from any individual model: adding a model never changes these
shapes, so routes, schemas and clients keep working.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import (
    EvidenceLevel,
    Framework,
    Modality,
    ModelStatus,
    Precision,
    RunStatus,
    TaskType,
)


class PredictionItem(BaseModel):
    """One scored label.

    ``score`` is the model's output for the label - a probability when the
    model is calibrated for it, otherwise a relative score. It is never a
    diagnosis.
    """

    label: str
    score: float = Field(ge=0.0, le=1.0)
    rank: int = 0
    extra: dict[str, Any] | None = None


class BoundingBox(BaseModel):
    """Axis-aligned detection box in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int


class DetectionItem(BaseModel):
    """A detected object with its box."""

    label: str
    score: float = Field(ge=0.0, le=1.0)
    box: BoundingBox


class MaskPayload(BaseModel):
    """A segmentation mask produced in memory, before it is persisted.

    ``data`` holds PNG bytes; the inference engine writes it to object storage
    and replaces it with a key/URL in the API response.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    label: str
    data: bytes = Field(repr=False)
    content_type: str = "image/png"
    area_px: float | None = None
    area_ratio: float | None = None
    measurements: dict[str, Any] = Field(default_factory=dict)


class ArtifactPayload(BaseModel):
    """A derived visual output (Grad-CAM, overlay, probability map)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["gradcam", "overlay", "probability_map", "attention", "diagnostic"]
    name: str
    data: bytes = Field(repr=False)
    content_type: str = "image/png"
    meta: dict[str, Any] = Field(default_factory=dict)


class QualityReport(BaseModel):
    """Output of a quality-control model, used by the quality gate."""

    quality_score: float = Field(ge=0.0, le=1.0)
    is_valid: bool
    issues: list[str] = Field(default_factory=list)
    recommendation: str = ""
    metrics: dict[str, float] = Field(default_factory=dict)


class DeviceInfo(BaseModel):
    """Where and how an inference ran."""

    device: str = "cpu"
    device_name: str | None = None
    precision: Precision = Precision.FP32
    batch_size: int = 1
    vram_total_mb: float | None = None
    vram_used_mb: float | None = None


class ModelResult(BaseModel):
    """The single result shape every model returns.

    Only fields relevant to a model's task are populated: a classifier fills
    ``predictions``, a segmenter fills ``segmentations`` and ``measurements``,
    a quality model fills ``quality``.
    """

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    model_version: str
    task: TaskType
    status: RunStatus = RunStatus.COMPLETED
    predictions: list[PredictionItem] = Field(default_factory=list)
    detections: list[DetectionItem] = Field(default_factory=list)
    segmentations: list[MaskPayload] = Field(default_factory=list)
    artifacts: list[ArtifactPayload] = Field(default_factory=list)
    measurements: dict[str, Any] = Field(default_factory=dict)
    quality: QualityReport | None = None
    warnings: list[str] = Field(default_factory=list)
    processing_time_ms: float = 0.0
    device_info: DeviceInfo = Field(default_factory=DeviceInfo)
    input_hash: str | None = None
    error_message: str | None = None

    @property
    def top_prediction(self) -> PredictionItem | None:
        """Highest-scoring prediction, if any."""
        return max(self.predictions, key=lambda p: p.score, default=None)


class WeightSpec(BaseModel):
    """Declares the checkpoint a model adapter needs.

    The platform never ships or fabricates weights. An adapter states the file
    it expects under ``MODEL_DIR``; when the file is absent the model is
    registered with status ``unavailable`` and the API reports exactly what to
    provide (see ``docs/MODEL_REGISTRY.md``).
    """

    filename: str
    #: Path relative to MODEL_DIR, e.g. ``oct/layers``.
    subdir: str = ""
    sha256: str | None = None
    format: Literal["pt", "pth", "safetensors", "onnx"] = "pt"
    #: Free-text description of the checkpoint's expected state dict / graph.
    expects: str = ""


class InputSpec(BaseModel):
    """Preprocessing contract of a model.

    These values come from the original model's documentation and must not be
    guessed: feeding a checkpoint the wrong input size or normalization
    silently degrades its output.
    """

    image_size: tuple[int, int] | None = None
    channels: int = 3
    color_space: Literal["rgb", "bgr", "gray"] = "rgb"
    normalization_mean: tuple[float, ...] | None = None
    normalization_std: tuple[float, ...] | None = None
    scale_to_unit_interval: bool = True
    notes: str = ""


class OutputSpec(BaseModel):
    """What a model emits: class labels or segmentation classes."""

    labels: list[str] = Field(default_factory=list)
    #: For segmentation: index -> class name, index 0 conventionally background.
    segmentation_classes: list[str] = Field(default_factory=list)
    units: dict[str, str] = Field(default_factory=dict)
    notes: str = ""


class ModelLicense(BaseModel):
    """License and provenance of a model and its training data."""

    name: str = "unknown"
    url: str | None = None
    source_url: str | None = None
    dataset: str | None = None
    dataset_license: str | None = None
    commercial_use: Literal["allowed", "restricted", "prohibited", "unknown"] = "unknown"
    citation: str | None = None
    restrictions: str | None = None


class ModelMetadata(BaseModel):
    """Everything the registry knows about a model version."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    name: str
    version: str
    modality: Modality
    task: TaskType
    framework: Framework
    evidence_level: EvidenceLevel
    description: str = ""
    status: ModelStatus = ModelStatus.ACTIVE
    input_spec: InputSpec = Field(default_factory=InputSpec)
    output_spec: OutputSpec = Field(default_factory=OutputSpec)
    weights: WeightSpec | None = None
    license: ModelLicense = Field(default_factory=ModelLicense)
    supports_explainability: bool = False
    #: Metrics reported by the model's authors, copied verbatim from their
    #: documentation. The platform never invents or back-fills these.
    reported_metrics: dict[str, Any] = Field(default_factory=dict)
    #: Documented limitations that a reader of the results must know about.
    limitations: str = ""


class Availability(BaseModel):
    """Whether a model can actually run in this deployment."""

    available: bool
    reason: str | None = None
    missing: list[str] = Field(default_factory=list)
    remediation: str | None = None
