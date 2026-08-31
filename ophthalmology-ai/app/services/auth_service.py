"""Authentication service."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from app.core.enums import AuditAction, UserRole
from app.core.exceptions import AuthenticationError, ConflictError
from app.core.security import (
    REFRESH_TOKEN_TYPE,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    needs_rehash,
    validate_password_strength,
    verify_password,
)
from app.database.base import utcnow
from app.database.models import User
from app.database.repositories import UserRepository
from app.services.audit_service import AuditService


class AuthService:
    """Password authentication and JWT issuance."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.users = UserRepository(session)
        self.audit = AuditService(session)

    def create_user(self, *, email: str, full_name: str, password: str, role: UserRole) -> User:
        """Create a user after enforcing the password policy.

        Raises:
            ConflictError: when the email is already registered.
            ValidationError: when the password is too weak.
        """
        normalized = email.strip().lower()
        if self.users.get_by_email(normalized):
            raise ConflictError("A user with this email already exists.")
        validate_password_strength(password)
        user = User(
            email=normalized,
            full_name=full_name.strip(),
            hashed_password=hash_password(password),
            role=role,
        )
        self.users.add(user)
        self.audit.record(
            AuditAction.USER_CREATE,
            resource_type="user",
            resource_id=user.id,
            meta={"role": str(role)},
        )
        return user

    def authenticate(self, email: str, password: str, *, client_ip: str | None = None) -> User:
        """Verify credentials.

        The same error is returned for an unknown email, a wrong password and a
        deactivated account, so the endpoint cannot be used to enumerate users.

        Raises:
            AuthenticationError: on any failure.
        """
        user = self.users.get_by_email(email.strip().lower())
        # Always run a verification so timing does not reveal whether the email
        # exists.
        stored = user.hashed_password if user else hash_password("invalid-placeholder")
        password_ok = verify_password(password, stored)

        if user is None or not password_ok or not user.is_active:
            self.audit.record(
                AuditAction.LOGIN_FAILURE,
                actor=user if user and password_ok else None,
                resource_type="user",
                resource_id=user.id if user else None,
                outcome="failure",
                client_ip=client_ip,
            )
            raise AuthenticationError("Invalid email or password.")

        if needs_rehash(user.hashed_password):
            user.hashed_password = hash_password(password)
        user.last_login_at = utcnow()
        self.audit.record(
            AuditAction.LOGIN_SUCCESS,
            actor=user,
            resource_type="user",
            resource_id=user.id,
            client_ip=client_ip,
        )
        return user

    def issue_tokens(self, user: User) -> tuple[str, str, datetime]:
        """Return ``(access_token, refresh_token, access_expires_at)``."""
        access_token, expires_at = create_access_token(str(user.id), role=user.role)
        refresh_token, _ = create_refresh_token(str(user.id))
        return access_token, refresh_token, expires_at

    def refresh(self, refresh_token: str) -> tuple[User, str, str, datetime]:
        """Exchange a refresh token for a new token pair.

        Raises:
            AuthenticationError: if the token is invalid, expired, of the wrong
                type, or its user is gone or deactivated.
        """
        payload = decode_token(refresh_token, expected_type=REFRESH_TOKEN_TYPE)
        try:
            subject = uuid.UUID(str(payload.get("sub")))
        except ValueError as exc:
            raise AuthenticationError("Token subject is malformed.") from exc
        user = self.users.get(subject)
        if user is None or not user.is_active:
            raise AuthenticationError("Token subject is no longer active.")
        access_token, new_refresh, expires_at = self.issue_tokens(user)
        self.audit.record(
            AuditAction.TOKEN_REFRESH, actor=user, resource_type="user", resource_id=user.id
        )
        return user, access_token, new_refresh, expires_at
