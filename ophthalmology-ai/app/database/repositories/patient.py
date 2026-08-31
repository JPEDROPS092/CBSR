"""Patient repository."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select

from app.database.models import Patient
from app.database.repositories.base import BaseRepository


class PatientRepository(BaseRepository[Patient]):
    model = Patient

    def get_by_external_ref(self, external_ref: str) -> Patient | None:
        """Look up a patient by the site's pseudonymous identifier."""
        stmt = select(Patient).where(Patient.external_ref == external_ref)
        return self.session.execute(stmt).scalar_one_or_none()

    def search(
        self, *, limit: int, offset: int, external_ref: str | None = None
    ) -> Sequence[Patient]:
        """List patients, optionally filtering by external reference prefix."""
        stmt = select(Patient).order_by(Patient.created_at.desc()).limit(limit).offset(offset)
        if external_ref:
            stmt = stmt.where(Patient.external_ref.startswith(external_ref))
        return self.session.execute(stmt).scalars().all()
