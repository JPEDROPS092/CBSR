"""Audit-log repository."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select

from app.database.models import AuditLog
from app.database.repositories.base import BaseRepository


class AuditLogRepository(BaseRepository[AuditLog]):
    model = AuditLog

    def recent(self, *, limit: int = 100, offset: int = 0) -> Sequence[AuditLog]:
        """Most recent audit entries first."""
        stmt = select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit).offset(offset)
        return self.session.execute(stmt).scalars().all()
