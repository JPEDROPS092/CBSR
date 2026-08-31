"""Authentication routes."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status

from app.api.dependencies import (
    CurrentUser,
    client_ip,
    get_auth_service,
    require_permission,
)
from app.core.security import Permission
from app.schemas.auth import LoginRequest, RefreshRequest, TokenResponse, UserCreate, UserRead
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])

AuthDep = Annotated[AuthService, Depends(get_auth_service)]


@router.post("/login", response_model=TokenResponse, summary="Exchange credentials for tokens")
def login(payload: LoginRequest, request: Request, service: AuthDep) -> TokenResponse:
    """Authenticate a user and issue an access/refresh token pair."""
    user = service.authenticate(payload.email, payload.password, client_ip=client_ip(request))
    access_token, refresh_token, expires_at = service.issue_tokens(user)
    service.session.commit()
    return TokenResponse(
        access_token=access_token, refresh_token=refresh_token, expires_at=expires_at
    )


@router.post("/refresh", response_model=TokenResponse, summary="Rotate an access token")
def refresh(payload: RefreshRequest, service: AuthDep) -> TokenResponse:
    """Exchange a valid refresh token for a new token pair."""
    _, access_token, refresh_token, expires_at = service.refresh(payload.refresh_token)
    service.session.commit()
    return TokenResponse(
        access_token=access_token, refresh_token=refresh_token, expires_at=expires_at
    )


@router.get("/me", response_model=UserRead, summary="Current user")
def me(user: CurrentUser) -> UserRead:
    """Return the authenticated user's profile."""
    return UserRead.model_validate(user)


@router.post(
    "/users",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user (admin only)",
    dependencies=[Depends(require_permission(Permission.USER_MANAGE))],
)
def create_user(payload: UserCreate, service: AuthDep) -> UserRead:
    """Create a platform user with an explicit role."""
    user = service.create_user(
        email=payload.email,
        full_name=payload.full_name,
        password=payload.password,
        role=payload.role,
    )
    service.session.commit()
    return UserRead.model_validate(user)
