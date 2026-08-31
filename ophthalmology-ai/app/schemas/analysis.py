"""Analysis request and result schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import AnalysisStatus, RunStatus, TaskType


class AnalysisCreate(BaseModel):
    """Request body of ``POST /analysis``."""

    exam_id: uuid.UUID
    image_id: uuid.UUID | None = Field(
        default=None, description="Defaults to the exam's most recent image."
    )
    models: list[str] = Field(
        default_factory=list,
        description=(
            'Model ids, optionally version-pinned as "model_id@1.2.0". Empty runs the '
            "default pipeline for the exam's modality."
        ),
        examples=[["oct_quality_v1", "oct_layers_classical_v1"]],
    )
    quality_gate: bool | None = Field(
        default=None, description="Override QUALITY_GATE_ENABLED for this analysis."
    )
    explainability: bool | None = Field(
        default=None, description="Override EXPLAINABILITY_ENABLED for this analysis."
    )
    frame_selection: str = Field(
        default="middle",
        description="Which frame of a multi-frame series to analyse: first, middle or last.",
    )


class AnalysisAccepted(BaseModel):
    """Response of ``POST /analysis``: the job has been queued."""

    analysis_id: uuid.UUID
    status: AnalysisStatus
    task_id: str | None = None


class PredictionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    label: str
    score: float
    rank: int
    extra: dict[str, Any] | None = None


class SegmentationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    label: str
    mask_url: str
    content_type: str
    area_px: float | None
    area_ratio: float | None
    measurements: dict[str, Any] | None


class ArtifactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    kind: str
    artifact_url: str
    content_type: str
    meta: dict[str, Any] | None


class ModelRunRead(BaseModel):
    """One model's contribution to an analysis."""

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    id: uuid.UUID
    model_id: str
    model_version: str
    task: TaskType
    status: RunStatus
    device: str
    device_name: str | None
    precision: str
    batch_size: int
    processing_time_ms: float | None
    input_hash: str | None
    software_version: str | None
    predictions: list[PredictionRead] = Field(default_factory=list)
    segmentations: list[SegmentationRead] = Field(default_factory=list)
    artifacts: list[ArtifactRead] = Field(default_factory=list)
    measurements: dict[str, Any] | None = None
    quality: dict[str, Any] | None = None
    warnings: list[str] | None = None
    error_message: str | None = None


class AnalysisRead(BaseModel):
    """Full analysis document returned by ``GET /analysis/{id}``."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    exam_id: uuid.UUID
    image_id: uuid.UUID | None
    status: AnalysisStatus
    requested_models: list[str]
    quality_summary: dict[str, Any] | None
    queued_at: datetime | None
    started_at: datetime | None
    finished_at: datetime | None
    error_code: str | None
    error_message: str | None
    software_version: str | None
    created_at: datetime
    models: list[ModelRunRead] = Field(
        default_factory=list, description="One entry per model executed."
    )
    disclaimer: str
