"""Database engine and session management.

A single synchronous engine is shared by the API and the Celery workers.
Synchronous SQLAlchemy is a deliberate choice: inference tasks are CPU/GPU
bound and run in worker processes, so an async driver would add complexity
without throughput gains.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import Settings, get_settings


def _engine_kwargs(settings: Settings) -> dict[str, object]:
    if settings.is_sqlite:
        # In-memory SQLite needs a single shared connection to survive across
        # sessions; file-backed SQLite just needs the thread check relaxed.
        kwargs: dict[str, object] = {"connect_args": {"check_same_thread": False}}
        if ":memory:" in settings.DATABASE_URL:
            kwargs["poolclass"] = StaticPool
        return kwargs
    return {
        "pool_size": settings.DATABASE_POOL_SIZE,
        "max_overflow": settings.DATABASE_MAX_OVERFLOW,
        "pool_pre_ping": True,
    }


def create_db_engine(settings: Settings | None = None) -> Engine:
    """Build a configured SQLAlchemy engine."""
    settings = settings or get_settings()
    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.DATABASE_ECHO,
        future=True,
        **_engine_kwargs(settings),  # type: ignore[arg-type]
    )
    if settings.is_sqlite:

        @event.listens_for(engine, "connect")
        def _enable_sqlite_fks(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
            """SQLite ignores foreign keys unless explicitly told not to."""
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


engine: Engine = create_db_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency yielding a request-scoped session."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for workers and scripts."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
