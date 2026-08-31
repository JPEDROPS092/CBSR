"""Patient routes."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import (
    CurrentUser,
    get_exam_service,
    get_patient_service,
    require_permission,
)
from app.core.security import Permission
from app.schemas.common import Page
from app.schemas.exam import ExamRead
from app.schemas.patient import PatientCreate, PatientRead, PatientUpdate
from app.services.exam_service import ExamService
from app.services.patient_service import PatientService

router = APIRouter(prefix="/patients", tags=["patients"])

PatientDep = Annotated[PatientService, Depends(get_patient_service)]


@router.post(
    "",
    response_model=PatientRead,
    status_code=status.HTTP_201_CREATED,
    summary="Register a pseudonymous patient",
    dependencies=[Depends(require_permission(Permission.PATIENT_WRITE))],
)
def create_patient(payload: PatientCreate, service: PatientDep, user: CurrentUser) -> PatientRead:
    """Create a patient record identified only by a site-assigned reference."""
    patient = service.create(payload, actor=user)
    service.session.commit()
    return PatientRead.model_validate(patient)


@router.get(
    "",
    response_model=Page[PatientRead],
    summary="List patients",
    dependencies=[Depends(require_permission(Permission.PATIENT_READ))],
)
def list_patients(
    service: PatientDep,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
    external_ref: str | None = None,
) -> Page[PatientRead]:
    """List patients, optionally filtered by external reference prefix."""
    items, total = service.list(limit=limit, offset=offset, external_ref=external_ref)
    return Page[PatientRead](
        items=[PatientRead.model_validate(item) for item in items],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/{patient_id}",
    response_model=PatientRead,
    summary="Fetch a patient",
    dependencies=[Depends(require_permission(Permission.PATIENT_READ))],
)
def get_patient(patient_id: uuid.UUID, service: PatientDep, user: CurrentUser) -> PatientRead:
    """Fetch one patient by id."""
    patient = service.get(patient_id, actor=user)
    service.session.commit()
    return PatientRead.model_validate(patient)


@router.patch(
    "/{patient_id}",
    response_model=PatientRead,
    summary="Update a patient",
    dependencies=[Depends(require_permission(Permission.PATIENT_WRITE))],
)
def update_patient(
    patient_id: uuid.UUID, payload: PatientUpdate, service: PatientDep, user: CurrentUser
) -> PatientRead:
    """Apply a partial update to a patient record."""
    patient = service.update(patient_id, payload, actor=user)
    service.session.commit()
    return PatientRead.model_validate(patient)


@router.get(
    "/{patient_id}/exams",
    response_model=list[ExamRead],
    summary="List a patient's exams",
    dependencies=[Depends(require_permission(Permission.EXAM_READ))],
)
def list_patient_exams(
    patient_id: uuid.UUID,
    exams: Annotated[ExamService, Depends(get_exam_service)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ExamRead]:
    """List the exams recorded for a patient, newest first."""
    return [
        ExamRead.model_validate(exam)
        for exam in exams.list_for_patient(patient_id, limit=limit, offset=offset)
    ]
