"""Report schemas."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.core.enums import ReportFormat


class ReportCreate(BaseModel):
    format: ReportFormat = ReportFormat.JSON
    include_explainability: bool = True


class ReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    analysis_id: uuid.UUID
    format: ReportFormat
    payload: dict[str, Any] | None
    document_url: str | None = Field(default=None, description="Set for rendered HTML/PDF reports.")
    disclaimer_version: str
    created_at: datetime
