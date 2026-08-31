"""Authentication, authorization and the audit trail, through the API."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import AuditLog


def test_unauthenticated_requests_are_refused(client: TestClient) -> None:
    response = client.get("/api/v1/patients")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_failed"


def test_login_failure_does_not_reveal_whether_the_user_exists(
    client: TestClient, doctor_auth: dict[str, str]
) -> None:
    """Wrong password and unknown email must be indistinguishable."""
    known = client.post(
        "/api/v1/auth/login",
        json={"email": "doctor@example.com", "password": "wrong-password-value"},
    )
    unknown = client.post(
        "/api/v1/auth/login",
        json={"email": "nobody@example.com", "password": "wrong-password-value"},
    )
    assert known.status_code == unknown.status_code == 401
    assert known.json() == unknown.json()


def test_me_returns_the_authenticated_user(client: TestClient, doctor_auth: dict[str, str]) -> None:
    response = client.get("/api/v1/auth/me", headers=doctor_auth)
    assert response.status_code == 200
    assert response.json()["role"] == "doctor"


def test_refresh_rotates_tokens(client: TestClient) -> None:
    from app.core.enums import UserRole
    from tests.conftest import _create_user

    user, password = _create_user(UserRole.DOCTOR)
    tokens = client.post(
        "/api/v1/auth/login", json={"email": user.email, "password": password}
    ).json()

    refreshed = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert refreshed.status_code == 200
    assert refreshed.json()["access_token"] != tokens["access_token"]

    # An access token is not a refresh token.
    assert (
        client.post(
            "/api/v1/auth/refresh", json={"refresh_token": tokens["access_token"]}
        ).status_code
        == 401
    )


def test_viewer_cannot_create_patients_or_run_analyses(
    client: TestClient, viewer_auth: dict[str, str]
) -> None:
    denied = client.post("/api/v1/patients", json={"external_ref": "P-RBAC-1"}, headers=viewer_auth)
    assert denied.status_code == 403
    assert denied.json()["error"]["code"] == "permission_denied"

    run = client.post(
        "/api/v1/analysis",
        json={"exam_id": "00000000-0000-0000-0000-000000000000"},
        headers=viewer_auth,
    )
    assert run.status_code == 403


def test_only_admins_can_create_users(
    client: TestClient, admin_auth: dict[str, str], doctor_auth: dict[str, str]
) -> None:
    payload = {
        "email": "new.user@example.com",
        "full_name": "New User",
        "password": "A-strong-password-1",
        "role": "operator",
    }
    assert client.post("/api/v1/auth/users", json=payload, headers=doctor_auth).status_code == 403
    created = client.post("/api/v1/auth/users", json=payload, headers=admin_auth)
    assert created.status_code == 201
    assert created.json()["role"] == "operator"


def test_weak_passwords_are_refused(client: TestClient, admin_auth: dict[str, str]) -> None:
    response = client.post(
        "/api/v1/auth/users",
        json={
            "email": "weak@example.com",
            "full_name": "Weak",
            "password": "short",
            "role": "viewer",
        },
        headers=admin_auth,
    )
    assert response.status_code == 422


def test_audit_trail_records_access_without_clinical_content(
    client: TestClient, doctor_auth: dict[str, str], session: Session
) -> None:
    """Every patient access is auditable; the trail holds ids, not data."""
    patient = client.post(
        "/api/v1/patients", json={"external_ref": "P-AUDIT-1"}, headers=doctor_auth
    ).json()
    client.get(f"/api/v1/patients/{patient['id']}", headers=doctor_auth)

    entries = session.execute(select(AuditLog)).scalars().all()
    actions = {entry.action for entry in entries}
    assert {"login.success", "patient.create", "patient.read"} <= actions

    created = next(e for e in entries if e.action == "patient.create")
    assert created.resource_id == patient["id"]
    assert created.actor_user_id is not None
    assert created.request_id
    assert created.actor_role == "doctor"
    # No clinical or directly identifying content is stored in the trail.
    assert created.meta is None or "external_ref" not in str(created.meta)


def test_patient_endpoint_rejects_identifier_like_references(
    client: TestClient, doctor_auth: dict[str, str]
) -> None:
    """A CPF-shaped reference is refused: the platform stores pseudonyms only."""
    response = client.post(
        "/api/v1/patients", json={"external_ref": "12345678901"}, headers=doctor_auth
    )
    assert response.status_code == 422


def test_request_id_is_echoed(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "trace-me-123"})
    assert response.headers["X-Request-ID"] == "trace-me-123"
