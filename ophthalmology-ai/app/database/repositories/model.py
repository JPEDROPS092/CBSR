"""Model-registry repository."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select

from app.database.models import ModelRecord
from app.database.repositories.base import BaseRepository


class ModelRecordRepository(BaseRepository[ModelRecord]):
    model = ModelRecord

    def get_by_model_id(self, model_id: str, version: str | None = None) -> ModelRecord | None:
        """Fetch a model record, defaulting to the most recently registered version."""
        stmt = select(ModelRecord).where(ModelRecord.model_id == model_id)
        if version:
            stmt = stmt.where(ModelRecord.version == version)
        stmt = stmt.order_by(ModelRecord.created_at.desc())
        return self.session.execute(stmt).scalars().first()

    def list_all(self) -> Sequence[ModelRecord]:
        """List every registered model version."""
        stmt = select(ModelRecord).order_by(ModelRecord.model_id, ModelRecord.version)
        return self.session.execute(stmt).scalars().all()
