"""Model-registry schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.ai.results import Availability, ModelLicense
from app.core.enums import EvidenceLevel, Framework, Modality, ModelStatus, TaskType


class ModelSummary(BaseModel):
    """One entry of ``GET /models``."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    name: str
    version: str
    modality: Modality
    task: TaskType
    framework: Framework
    evidence_level: EvidenceLevel
    status: ModelStatus
    available: bool
    description: str = ""


class ModelDetail(ModelSummary):
    """Full metadata of one model, including how to install it if missing."""

    availability: Availability
    input_spec: dict[str, Any]
    output_spec: dict[str, Any]
    license: ModelLicense
    supports_explainability: bool
    reported_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Metrics as published by the model's authors. Empty means no metric has "
            "been supplied - the platform never estimates one."
        ),
    )
    limitations: str = ""
