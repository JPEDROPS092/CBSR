"""Password hashing, tokens and the RBAC matrix."""

from __future__ import annotations

import pytest

from app.core.enums import UserRole
from app.core.exceptions import AuthenticationError, ValidationError
from app.core.security import (
    ACCESS_TOKEN_TYPE,
    REFRESH_TOKEN_TYPE,
    Permission,
    create_access_token,
    create_refresh_token,
    decode_token,
    has_permission,
    hash_password,
    needs_rehash,
    validate_password_strength,
    verify_password,
)


def test_password_hash_round_trip() -> None:
    hashed = hash_password("correct-horse-battery-9", iterations=1000)
    assert hashed.startswith("pbkdf2_sha256$1000$")
    assert verify_password("correct-horse-battery-9", hashed)
    assert not verify_password("wrong-password", hashed)


def test_password_hashes_are_salted() -> None:
    """Two hashes of the same password must differ."""
    assert hash_password("same-password-1", iterations=1000) != hash_password(
        "same-password-1", iterations=1000
    )


def test_malformed_hash_is_rejected_not_crashed() -> None:
    assert verify_password("anything", "not-a-valid-hash") is False


def test_needs_rehash_detects_outdated_cost() -> None:
    assert needs_rehash(hash_password("password-12345", iterations=1000), iterations=600_000)
    assert not needs_rehash(hash_password("password-12345", iterations=1000), iterations=500)


@pytest.mark.parametrize("password", ["short1", "nodigitshereatall", "123456789012"])
def test_weak_passwords_are_rejected(password: str) -> None:
    with pytest.raises(ValidationError):
        validate_password_strength(password)


def test_access_token_carries_role_and_type() -> None:
    token, _ = create_access_token("user-1", role=UserRole.DOCTOR)
    payload = decode_token(token, expected_type=ACCESS_TOKEN_TYPE)
    assert payload["sub"] == "user-1"
    assert payload["role"] == "doctor"
    assert payload["jti"]


def test_refresh_token_is_not_accepted_as_access_token() -> None:
    """A refresh token must never authorize a request."""
    token, _ = create_refresh_token("user-1")
    with pytest.raises(AuthenticationError):
        decode_token(token, expected_type=ACCESS_TOKEN_TYPE)
    assert decode_token(token, expected_type=REFRESH_TOKEN_TYPE)["sub"] == "user-1"


def test_tampered_token_is_rejected() -> None:
    token, _ = create_access_token("user-1", role=UserRole.ADMIN)
    head, payload, signature = token.split(".")
    with pytest.raises(AuthenticationError):
        decode_token(f"{head}.{payload}.{signature[:-2]}xx")


@pytest.mark.parametrize(
    ("role", "permission", "expected"),
    [
        (UserRole.ADMIN, Permission.USER_MANAGE, True),
        (UserRole.DOCTOR, Permission.ANALYSIS_RUN, True),
        (UserRole.DOCTOR, Permission.USER_MANAGE, False),
        (UserRole.RESEARCHER, Permission.ANALYSIS_RUN, True),
        (UserRole.RESEARCHER, Permission.PATIENT_WRITE, False),
        (UserRole.OPERATOR, Permission.IMAGE_UPLOAD, True),
        (UserRole.OPERATOR, Permission.ANALYSIS_RUN, False),
        (UserRole.VIEWER, Permission.ANALYSIS_READ, True),
        (UserRole.VIEWER, Permission.ANALYSIS_RUN, False),
        (UserRole.VIEWER, Permission.PATIENT_WRITE, False),
    ],
)
def test_rbac_matrix(role: UserRole, permission: Permission, expected: bool) -> None:
    assert has_permission(role, permission) is expected
