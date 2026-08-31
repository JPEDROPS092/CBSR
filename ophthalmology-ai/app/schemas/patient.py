"""Patient schemas.

The API surface carries no direct identifiers: a patient is a UUID plus a
site-assigned pseudonymous reference. Names and national identifiers are
rejected by design, not merely discouraged - see ``docs/SECURITY.md``.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Digit-only strings this long look like a CPF or a national id and are
#: refused as a pseudonymous reference.
_ID_LIKE = re.compile(r"^\d{9,}$")


class PatientBase(BaseModel):
    external_ref: str = Field(
        min_length=1,
        max_length=64,
        description=(
            "Site-assigned pseudonymous identifier. Must not be a name, a national "
            "identifier (CPF) or any other directly identifying value."
        ),
        examples=["P-2024-0001"],
    )
    birth_year: int | None = Field(default=None, ge=1850, le=2100)
    sex: Literal["female", "male", "other", "unknown"] | None = None
    consent_research: bool = False
    clinical_context: dict[str, Any] | None = Field(
        default=None,
        description="Non-identifying clinical context (e.g. diabetes duration, HbA1c band).",
    )

    @field_validator("external_ref")
    @classmethod
    def _reject_identifier_like_refs(cls, value: str) -> str:
        candidate = value.strip()
        if _ID_LIKE.match(candidate.replace(".", "").replace("-", "")):
            raise ValueError(
                "external_ref looks like a national identifier; use a pseudonymous "
                "reference instead."
            )
        return candidate


class PatientCreate(PatientBase):
    pass


class PatientUpdate(BaseModel):
    birth_year: int | None = Field(default=None, ge=1850, le=2100)
    sex: Literal["female", "male", "other", "unknown"] | None = None
    consent_research: bool | None = None
    clinical_context: dict[str, Any] | None = None


class PatientRead(PatientBase):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    created_at: datetime
    updated_at: datetime
