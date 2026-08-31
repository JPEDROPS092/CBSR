"""Exam and image schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import ImageStatus, Laterality, Modality


class ExamCreate(BaseModel):
    patient_id: uuid.UUID
    modality: Modality
    laterality: Laterality = Laterality.UNKNOWN
    acquired_at: datetime | None = None
    device_manufacturer: str | None = Field(default=None, max_length=120)
    device_model: str | None = Field(default=None, max_length=120)
    acquisition_metadata: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Device acquisition parameters. Include pixel_spacing_um "
            '(e.g. {"axial": 3.87, "lateral": 11.7}) to obtain measurements in micrometres.'
        ),
    )


class ExamRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    patient_id: uuid.UUID
    modality: Modality
    laterality: Laterality
    acquired_at: datetime | None
    device_manufacturer: str | None
    device_model: str | None
    acquisition_metadata: dict[str, Any] | None
    created_at: datetime


class ImageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    exam_id: uuid.UUID
    content_type: str
    byte_size: int
    checksum_sha256: str
    width: int | None
    height: int | None
    num_frames: int
    pixel_spacing_um: dict[str, Any] | None
    status: ImageStatus
    created_at: datetime


class ImageUploadResponse(BaseModel):
    """Upload result, including the quality screening performed inline."""

    image: ImageRead
    quality: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Result of the quality model for the exam's modality, run at upload time so "
            "an operator can re-acquire immediately. Advisory only - it does not block "
            "the upload."
        ),
    )
