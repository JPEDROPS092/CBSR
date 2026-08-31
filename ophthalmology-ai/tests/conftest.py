"""Shared test fixtures.

The environment is configured **before** any application module is imported,
because settings are read once into a cached singleton. Tests run against a
file-backed SQLite database, local object storage in a temporary directory and
the inline task queue, so no external service is needed.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

_TMP_ROOT = Path(tempfile.mkdtemp(prefix="ophthalmology-tests-"))

os.environ.update(
    {
        "ENVIRONMENT": "test",
        "DATABASE_URL": f"sqlite+pysqlite:///{_TMP_ROOT / 'test.db'}",
        "STORAGE_BACKEND": "local",
        "STORAGE_LOCAL_ROOT": str(_TMP_ROOT / "storage"),
        "MODEL_DIR": str(_TMP_ROOT / "models"),
        "TASK_QUEUE_BACKEND": "inline",
        "JWT_SECRET": "test-secret-value-that-is-long-enough-1234",
        "LOG_LEVEL": "WARNING",
        "LOG_FORMAT": "console",
        "RATE_LIMIT_ENABLED": "false",
        "OBJECT_STORAGE_BUCKET": "test-bucket",
    }
)
(_TMP_ROOT / "models").mkdir(parents=True, exist_ok=True)

from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.ai.models import bootstrap_registry  # noqa: E402
from app.ai.registry import registry  # noqa: E402
from app.core.enums import UserRole  # noqa: E402
from app.core.security import hash_password  # noqa: E402
from app.database.models import Base, User  # noqa: E402
from app.database.session import SessionLocal, engine  # noqa: E402
from app.main import create_app  # noqa: E402
from app.storage import get_storage  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _database() -> Iterator[None]:
    """Create the schema once for the whole test session."""
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="session", autouse=True)
def _registry() -> Iterator[None]:
    """Populate the model registry once."""
    bootstrap_registry()
    yield
    registry.clear()


@pytest.fixture
def session() -> Iterator[Session]:
    """A database session, rolled back and cleaned up after each test."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.rollback()
        db.close()


@pytest.fixture(autouse=True)
def _clean_tables() -> Iterator[None]:
    """Truncate every table between tests to keep them independent."""
    yield
    with engine.begin() as connection:
        for table in reversed(Base.metadata.sorted_tables):
            connection.execute(table.delete())


@pytest.fixture
def client() -> Iterator[TestClient]:
    """A TestClient with the application's lifespan executed."""
    with TestClient(create_app()) as test_client:
        yield test_client


def _create_user(role: UserRole) -> tuple[User, str]:
    """Create a user with a known password and return it with that password."""
    password = "Test-Password-2024"
    session = SessionLocal()
    try:
        user = User(
            email=f"{role}-{uuid.uuid4().hex[:8]}@example.com",
            full_name=f"{role.title()} User",
            hashed_password=hash_password(password, iterations=1000),
            role=role,
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user, password
    finally:
        session.close()


@pytest.fixture
def admin_auth(client: TestClient) -> dict[str, str]:
    """Authorization header for an admin user."""
    return _auth_header(client, UserRole.ADMIN)


@pytest.fixture
def doctor_auth(client: TestClient) -> dict[str, str]:
    """Authorization header for a doctor user."""
    return _auth_header(client, UserRole.DOCTOR)


@pytest.fixture
def viewer_auth(client: TestClient) -> dict[str, str]:
    """Authorization header for a read-only user."""
    return _auth_header(client, UserRole.VIEWER)


def _auth_header(client: TestClient, role: UserRole) -> dict[str, str]:
    user, password = _create_user(role)
    response = client.post("/api/v1/auth/login", json={"email": user.email, "password": password})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


@pytest.fixture(autouse=True)
def _clean_storage() -> Iterator[None]:
    """Empty the object store between tests."""
    yield
    storage = get_storage()
    if hasattr(storage, "clear"):
        storage.clear()


@pytest.fixture
def model_dir() -> Path:
    """The temporary ``MODEL_DIR`` used by tests."""
    return _TMP_ROOT / "models"
