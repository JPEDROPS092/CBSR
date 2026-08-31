"""Shared API schemas."""

from __future__ import annotations

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

ItemT = TypeVar("ItemT")


class Page(BaseModel, Generic[ItemT]):
    """A page of results."""

    items: list[ItemT]
    total: int = Field(description="Total number of matching records.")
    limit: int
    offset: int


class ErrorDetail(BaseModel):
    """The platform's standard error envelope."""

    code: str
    message: str
    details: dict | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


class MessageResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    """Liveness/readiness payload."""

    status: str
    environment: str
    software_version: str
    database: str
    storage: str
    models_registered: int
    models_available: int
    device: dict
