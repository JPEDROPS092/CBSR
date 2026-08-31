"""Generic repository.

Repositories keep SQLAlchemy queries out of the service layer, which makes
services testable against fakes and keeps the domain independent of the ORM.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database.base import Base

ModelT = TypeVar("ModelT", bound=Base)


class BaseRepository(Generic[ModelT]):
    """CRUD operations shared by all repositories."""

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def add(self, entity: ModelT) -> ModelT:
        """Stage a new entity and flush so its primary key is populated."""
        self.session.add(entity)
        self.session.flush()
        return entity

    def get(self, entity_id: uuid.UUID) -> ModelT | None:
        """Fetch by primary key, or ``None``."""
        return self.session.get(self.model, entity_id)

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        order_by: Any = None,
        **filters: Any,
    ) -> Sequence[ModelT]:
        """List entities matching equality ``filters``."""
        stmt = select(self.model).filter_by(**filters).limit(limit).offset(offset)
        if order_by is not None:
            stmt = stmt.order_by(order_by)
        return self.session.execute(stmt).scalars().all()

    def count(self, **filters: Any) -> int:
        """Count entities matching equality ``filters``."""
        stmt = select(func.count()).select_from(self.model).filter_by(**filters)
        return int(self.session.execute(stmt).scalar_one())

    def delete(self, entity: ModelT) -> None:
        """Delete an entity."""
        self.session.delete(entity)
        self.session.flush()
