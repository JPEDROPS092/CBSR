"""Exam and image repositories."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select

from app.database.models import Exam, Image
from app.database.repositories.base import BaseRepository


class ExamRepository(BaseRepository[Exam]):
    model = Exam

    def list_for_patient(
        self, patient_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> Sequence[Exam]:
        """List a patient's exams, newest first."""
        stmt = (
            select(Exam)
            .where(Exam.patient_id == patient_id)
            .order_by(Exam.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return self.session.execute(stmt).scalars().all()


class ImageRepository(BaseRepository[Image]):
    model = Image

    def list_for_exam(self, exam_id: uuid.UUID) -> Sequence[Image]:
        """List an exam's images in upload order."""
        stmt = select(Image).where(Image.exam_id == exam_id).order_by(Image.created_at)
        return self.session.execute(stmt).scalars().all()

    def get_by_checksum(self, exam_id: uuid.UUID, checksum: str) -> Image | None:
        """Find an already-uploaded identical image inside the same exam."""
        stmt = select(Image).where(Image.exam_id == exam_id, Image.checksum_sha256 == checksum)
        return self.session.execute(stmt).scalars().first()
