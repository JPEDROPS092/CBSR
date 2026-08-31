"""FastAPI dependencies: authentication, authorization, services.

Everything a route needs is injected here, so routes stay thin and every
service can be swapped in tests via ``app.dependency_overrides``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.ai.registry import ModelRegistry
from app.ai.registry import registry as global_registry
from app.api.rate_limit import get_rate_limiter
from app.core.config import Settings, get_settings
from app.core.enums import UserRole
from app.core.exceptions import AuthenticationError, PermissionDeniedError, RateLimitError
from app.core.logging import user_id_var
from app.core.security import Permission, decode_token, has_permission
from app.database.models import User
from app.database.repositories import UserRepository
from app.database.session import get_db
from app.services.analysis_service import AnalysisService
from app.services.audit_service import AuditService
from app.services.auth_service import AuthService
from app.services.exam_service import ExamService
from app.services.patient_service import PatientService
from app.services.report_service import ReportService
from app.storage import ObjectStorage, get_storage

bearer_scheme = HTTPBearer(auto_error=False, description="JWT access token")

DbSession = Annotated[Session, Depends(get_db)]


def get_settings_dep() -> Settings:
    """Inject application settings."""
    return get_settings()


def get_registry_dep() -> ModelRegistry:
    """Inject the model registry."""
    return global_registry


def get_storage_dep() -> ObjectStorage:
    """Inject the object-storage backend."""
    return get_storage()


def client_ip(request: Request) -> str | None:
    """Best-effort client address.

    ``X-Forwarded-For`` is honoured because the API normally sits behind a
    reverse proxy; only the first hop is used.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else None


def enforce_rate_limit(request: Request) -> None:
    """Reject a caller that exceeded its quota.

    Raises:
        RateLimitError: when the quota for this identity is exhausted.
    """
    settings = get_settings()
    if not settings.RATE_LIMIT_ENABLED:
        return
    identity = request.headers.get("authorization") or client_ip(request) or "anonymous"
    # Never key the limiter on the raw token; hash it down to a short digest.
    key = str(hash(identity))
    if not get_rate_limiter().allow(key):
        raise RateLimitError(
            "Too many requests.",
            details={
                "limit": settings.RATE_LIMIT_REQUESTS,
                "window_seconds": settings.RATE_LIMIT_WINDOW_SECONDS,
            },
        )


def get_current_user(
    session: DbSession,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer_scheme)],
) -> User:
    """Resolve the authenticated user from the bearer token.

    Raises:
        AuthenticationError: missing/invalid token, or inactive user.
    """
    if credentials is None or not credentials.credentials:
        raise AuthenticationError("Authorization header is missing.")
    payload = decode_token(credentials.credentials)
    try:
        user_id = uuid.UUID(str(payload.get("sub")))
    except ValueError as exc:
        raise AuthenticationError("Token subject is malformed.") from exc

    user = UserRepository(session).get(user_id)
    if user is None or not user.is_active:
        raise AuthenticationError("User is not active.")
    user_id_var.set(str(user.id))
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def require_permission(permission: Permission) -> Callable[[User], User]:
    """Build a dependency that enforces one permission.

    Usage::

        @router.post("/", dependencies=[Depends(require_permission(Permission.PATIENT_WRITE))])
    """

    def dependency(user: CurrentUser) -> User:
        if not has_permission(user.role, permission):
            raise PermissionDeniedError(
                "Your role does not allow this action.",
                details={"required_permission": str(permission), "role": str(user.role)},
            )
        return user

    return dependency


def require_role(*roles: UserRole) -> Callable[[User], User]:
    """Build a dependency that restricts a route to specific roles."""

    def dependency(user: CurrentUser) -> User:
        if user.role not in roles:
            raise PermissionDeniedError(
                "Your role does not allow this action.",
                details={"allowed_roles": [str(role) for role in roles]},
            )
        return user

    return dependency


# --------------------------------------------------------------------------- #
# Service factories
# --------------------------------------------------------------------------- #
def get_auth_service(session: DbSession) -> AuthService:
    return AuthService(session)


def get_audit_service(session: DbSession) -> AuditService:
    return AuditService(session)


def get_patient_service(session: DbSession) -> PatientService:
    return PatientService(session)


def get_exam_service(
    session: DbSession,
    storage: Annotated[ObjectStorage, Depends(get_storage_dep)],
    model_registry: Annotated[ModelRegistry, Depends(get_registry_dep)],
) -> ExamService:
    return ExamService(session, storage=storage, model_registry=model_registry)


def get_analysis_service(
    session: DbSession,
    storage: Annotated[ObjectStorage, Depends(get_storage_dep)],
    model_registry: Annotated[ModelRegistry, Depends(get_registry_dep)],
) -> AnalysisService:
    return AnalysisService(session, storage=storage, model_registry=model_registry)


def get_report_service(
    session: DbSession,
    storage: Annotated[ObjectStorage, Depends(get_storage_dep)],
    model_registry: Annotated[ModelRegistry, Depends(get_registry_dep)],
) -> ReportService:
    return ReportService(session, storage=storage, model_registry=model_registry)
