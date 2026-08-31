"""Declarative base and shared column mixins."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeVar

from sqlalchemy import DateTime, MetaData, String, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import JSON, TypeDecorator

from app.core.enums import (
    AnalysisStatus,
    EvidenceLevel,
    Framework,
    ImageStatus,
    Laterality,
    Modality,
    ModelStatus,
    ReportFormat,
    RunStatus,
    TaskType,
    UserRole,
)

# Explicit naming convention so Alembic autogenerates stable constraint names.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

# JSONB on PostgreSQL, portable JSON everywhere else (SQLite in tests).
JSONType = JSON().with_variant(JSONB, "postgresql")

EnumT = TypeVar("EnumT", bound=StrEnum)


class StrEnumType(TypeDecorator[EnumT]):
    """Stores a :class:`StrEnum` as its value and loads it back as the member.

    Without this, a column declared ``Mapped[Modality]`` would come back from
    the database as a bare ``str``: comparisons such as
    ``exam.modality is Modality.OCT`` would silently be false, and a model
    would quietly not run. Native database enums are avoided on purpose -
    adding a value to one requires a migration on PostgreSQL.
    """

    impl = String
    cache_ok = True

    def __init__(self, enum_class: type[EnumT], length: int = 32) -> None:
        self.enum_class = enum_class
        super().__init__(length=length)

    def process_bind_param(self, value: EnumT | str | None, dialect: object) -> str | None:
        if value is None:
            return None
        return str(self.enum_class(value))

    def process_result_value(self, value: str | None, dialect: object) -> EnumT | None:
        if value is None:
            return None
        return self.enum_class(value)


#: Every domain enum is persisted through :class:`StrEnumType`, so ORM
#: attributes always hold real enum members.
ENUM_TYPE_MAP: dict[type, StrEnumType] = {
    UserRole: StrEnumType(UserRole),
    Modality: StrEnumType(Modality, 16),
    Laterality: StrEnumType(Laterality, 16),
    TaskType: StrEnumType(TaskType, 24),
    Framework: StrEnumType(Framework, 24),
    EvidenceLevel: StrEnumType(EvidenceLevel),
    ModelStatus: StrEnumType(ModelStatus, 16),
    AnalysisStatus: StrEnumType(AnalysisStatus, 16),
    RunStatus: StrEnumType(RunStatus, 24),
    ImageStatus: StrEnumType(ImageStatus, 16),
    ReportFormat: StrEnumType(ReportFormat, 8),
}


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {dict[str, Any]: JSONType, list[str]: JSONType, **ENUM_TYPE_MAP}


def utcnow() -> datetime:
    """Timezone-aware current time (UTC)."""
    return datetime.now(UTC)


class UUIDPrimaryKeyMixin:
    """UUID primary key.

    Sequential integer ids would leak how many patients or exams a deployment
    holds; UUIDs are also what the API exposes, per ``docs/SECURITY.md``.
    """

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)


class TimestampMixin:
    """``created_at`` / ``updated_at`` maintained by the database."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=utcnow,
        nullable=False,
    )
