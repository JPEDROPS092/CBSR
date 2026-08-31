"""Patient service."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy.orm import Session

from app.core.enums import AuditAction
from app.core.exceptions import ConflictError, NotFoundError
from app.database.models import Patient, User
from app.database.repositories import PatientRepository
from app.schemas.patient import PatientCreate, PatientUpdate
from app.services.audit_service import AuditService


class PatientService:
    """Creates and reads pseudonymous patient records."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.patients = PatientRepository(session)
        self.audit = AuditService(session)

    def create(self, payload: PatientCreate, *, actor: User | None = None) -> Patient:
        """Register a patient.

        Raises:
            ConflictError: when ``external_ref`` is already in use.
        """
        if self.patients.get_by_external_ref(payload.external_ref):
            raise ConflictError(
                "A patient with this external reference already exists.",
                details={"external_ref": payload.external_ref},
            )
        patient = Patient(
            external_ref=payload.external_ref,
            birth_year=payload.birth_year,
            sex=payload.sex,
            consent_research=payload.consent_research,
            clinical_context=payload.clinical_context,
            created_by_id=actor.id if actor else None,
        )
        self.patients.add(patient)
        self.audit.record(
            AuditAction.PATIENT_CREATE,
            actor=actor,
            resource_type="patient",
            resource_id=patient.id,
        )
        return patient

    def get(self, patient_id: uuid.UUID, *, actor: User | None = None) -> Patient:
        """Fetch a patient.

        Raises:
            NotFoundError: when no such patient exists.
        """
        patient = self.patients.get(patient_id)
        if patient is None:
            raise NotFoundError("Patient not found.")
        self.audit.record(
            AuditAction.PATIENT_READ,
            actor=actor,
            resource_type="patient",
            resource_id=patient.id,
        )
        return patient

    def list(
        self, *, limit: int = 50, offset: int = 0, external_ref: str | None = None
    ) -> tuple[Sequence[Patient], int]:
        """List patients and the total count."""
        items = self.patients.search(limit=limit, offset=offset, external_ref=external_ref)
        return items, self.patients.count()

    def update(
        self, patient_id: uuid.UUID, payload: PatientUpdate, *, actor: User | None = None
    ) -> Patient:
        """Apply a partial update to a patient record."""
        patient = self.patients.get(patient_id)
        if patient is None:
            raise NotFoundError("Patient not found.")
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(patient, field, value)
        self.session.flush()
        self.audit.record(
            AuditAction.PATIENT_UPDATE,
            actor=actor,
            resource_type="patient",
            resource_id=patient.id,
            meta={"fields": list(payload.model_dump(exclude_unset=True))},
        )
        return patient
