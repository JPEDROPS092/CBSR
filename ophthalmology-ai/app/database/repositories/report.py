"""Report repository."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select

from app.database.models import Report
from app.database.repositories.base import BaseRepository


class ReportRepository(BaseRepository[Report]):
    model = Report

    def list_for_analysis(self, analysis_id: uuid.UUID) -> Sequence[Report]:
        """List reports generated for an analysis, newest first."""
        stmt = (
            select(Report)
            .where(Report.analysis_id == analysis_id)
            .order_by(Report.created_at.desc())
        )
        return self.session.execute(stmt).scalars().all()
