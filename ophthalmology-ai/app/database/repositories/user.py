"""User repository."""

from __future__ import annotations

from sqlalchemy import select

from app.database.models import User
from app.database.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    model = User

    def get_by_email(self, email: str) -> User | None:
        """Look up a user by (case-insensitive) email."""
        stmt = select(User).where(User.email == email.strip().lower())
        return self.session.execute(stmt).scalar_one_or_none()
