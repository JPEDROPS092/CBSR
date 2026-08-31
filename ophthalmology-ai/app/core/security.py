"""Authentication and authorization primitives.

* Password hashing: PBKDF2-HMAC-SHA256 from the standard library, with a
  per-password random salt and a configurable iteration count. Hashes are
  stored in the PHC-like string ``pbkdf2_sha256$<iterations>$<salt>$<hash>``
  so the iteration count can be raised over time and old hashes upgraded on
  next login.
* Tokens: short-lived JWT access tokens plus long-lived refresh tokens. Both
  carry ``jti`` and ``typ`` claims; a refresh token is never accepted where an
  access token is required.
* RBAC: :func:`has_permission` maps roles to a fixed permission set.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, Final

import jwt

from app.core.config import Settings, get_settings
from app.core.enums import UserRole
from app.core.exceptions import AuthenticationError, ValidationError

_ALGORITHM_TAG: Final = "pbkdf2_sha256"
ACCESS_TOKEN_TYPE: Final = "access"
REFRESH_TOKEN_TYPE: Final = "refresh"


# --------------------------------------------------------------------------- #
# Passwords
# --------------------------------------------------------------------------- #
def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def hash_password(password: str, *, iterations: int | None = None) -> str:
    """Hash a plaintext password for storage."""
    settings = get_settings()
    iterations = iterations or settings.PASSWORD_HASH_ITERATIONS
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"{_ALGORITHM_TAG}${iterations}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Constant-time verification of a password against a stored hash."""
    try:
        tag, iterations_raw, salt_raw, digest_raw = stored_hash.split("$")
        if tag != _ALGORITHM_TAG:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), _unb64(salt_raw), int(iterations_raw)
        )
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(digest, _unb64(digest_raw))


def needs_rehash(stored_hash: str, *, iterations: int | None = None) -> bool:
    """True when a stored hash uses fewer iterations than currently configured."""
    target = iterations or get_settings().PASSWORD_HASH_ITERATIONS
    try:
        _, iterations_raw, _, _ = stored_hash.split("$")
        return int(iterations_raw) < target
    except ValueError:
        return True


def validate_password_strength(password: str, *, settings: Settings | None = None) -> None:
    """Reject passwords that are trivially weak.

    Raises:
        ValidationError: when the password fails the policy.
    """
    settings = settings or get_settings()
    problems: list[str] = []
    if len(password) < settings.PASSWORD_MIN_LENGTH:
        problems.append(f"must be at least {settings.PASSWORD_MIN_LENGTH} characters")
    if not any(c.isalpha() for c in password):
        problems.append("must contain a letter")
    if not any(c.isdigit() for c in password):
        problems.append("must contain a digit")
    if problems:
        raise ValidationError("Password does not meet the policy.", details={"problems": problems})


# --------------------------------------------------------------------------- #
# Tokens
# --------------------------------------------------------------------------- #
def _encode(
    subject: str, token_type: str, expires_delta: timedelta, extra: dict[str, Any]
) -> tuple[str, datetime]:
    settings = get_settings()
    now = datetime.now(UTC)
    expires_at = now + expires_delta
    payload: dict[str, Any] = {
        "sub": subject,
        "typ": token_type,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "jti": uuid.uuid4().hex,
        **extra,
    }
    token = jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token, expires_at


def create_access_token(
    subject: str, *, role: UserRole, expires_delta: timedelta | None = None
) -> tuple[str, datetime]:
    """Create a signed access token. Returns ``(token, expires_at)``."""
    settings = get_settings()
    delta = expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_TTL_MINUTES)
    return _encode(subject, ACCESS_TOKEN_TYPE, delta, {"role": str(role)})


def create_refresh_token(
    subject: str, *, expires_delta: timedelta | None = None
) -> tuple[str, datetime]:
    """Create a signed refresh token. Returns ``(token, expires_at)``."""
    settings = get_settings()
    delta = expires_delta or timedelta(days=settings.REFRESH_TOKEN_TTL_DAYS)
    return _encode(subject, REFRESH_TOKEN_TYPE, delta, {})


def decode_token(token: str, *, expected_type: str = ACCESS_TOKEN_TYPE) -> dict[str, Any]:
    """Decode and validate a JWT.

    Raises:
        AuthenticationError: if the token is expired, malformed, or of the
            wrong type.
    """
    settings = get_settings()
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    except jwt.ExpiredSignatureError as exc:
        raise AuthenticationError("Token has expired.") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthenticationError("Token is invalid.") from exc
    if payload.get("typ") != expected_type:
        raise AuthenticationError("Token type is not valid for this operation.")
    return payload


# --------------------------------------------------------------------------- #
# RBAC
# --------------------------------------------------------------------------- #
class Permission(StrEnum):
    """Fine-grained permissions granted to roles."""

    PATIENT_READ = "patient:read"
    PATIENT_WRITE = "patient:write"
    EXAM_READ = "exam:read"
    EXAM_WRITE = "exam:write"
    IMAGE_UPLOAD = "image:upload"
    ANALYSIS_READ = "analysis:read"
    ANALYSIS_RUN = "analysis:run"
    REPORT_READ = "report:read"
    REPORT_WRITE = "report:write"
    MODEL_READ = "model:read"
    MODEL_MANAGE = "model:manage"
    USER_MANAGE = "user:manage"
    AUDIT_READ = "audit:read"


ROLE_PERMISSIONS: Final[dict[UserRole, frozenset[Permission]]] = {
    UserRole.ADMIN: frozenset(Permission),
    UserRole.DOCTOR: frozenset(
        {
            Permission.PATIENT_READ,
            Permission.PATIENT_WRITE,
            Permission.EXAM_READ,
            Permission.EXAM_WRITE,
            Permission.IMAGE_UPLOAD,
            Permission.ANALYSIS_READ,
            Permission.ANALYSIS_RUN,
            Permission.REPORT_READ,
            Permission.REPORT_WRITE,
            Permission.MODEL_READ,
        }
    ),
    # Researchers work on de-identified cohorts: they may run models and read
    # results, but not create or edit patient records.
    UserRole.RESEARCHER: frozenset(
        {
            Permission.PATIENT_READ,
            Permission.EXAM_READ,
            Permission.ANALYSIS_READ,
            Permission.ANALYSIS_RUN,
            Permission.REPORT_READ,
            Permission.MODEL_READ,
        }
    ),
    # Operators acquire and upload images; they do not interpret results.
    UserRole.OPERATOR: frozenset(
        {
            Permission.PATIENT_READ,
            Permission.PATIENT_WRITE,
            Permission.EXAM_READ,
            Permission.EXAM_WRITE,
            Permission.IMAGE_UPLOAD,
            Permission.ANALYSIS_READ,
            Permission.MODEL_READ,
        }
    ),
    UserRole.VIEWER: frozenset(
        {
            Permission.PATIENT_READ,
            Permission.EXAM_READ,
            Permission.ANALYSIS_READ,
            Permission.REPORT_READ,
            Permission.MODEL_READ,
        }
    ),
}


def has_permission(role: UserRole, permission: Permission) -> bool:
    """Return whether ``role`` grants ``permission``."""
    return permission in ROLE_PERMISSIONS.get(role, frozenset())
