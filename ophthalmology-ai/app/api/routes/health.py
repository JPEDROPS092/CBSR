"""Health and readiness endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.ai.devices import get_device_manager
from app.ai.registry import ModelRegistry
from app.api.dependencies import DbSession, get_registry_dep, get_storage_dep
from app.core.config import get_settings
from app.schemas.common import HealthResponse
from app.storage import ObjectStorage

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness and dependency status")
def health(
    session: DbSession,
    storage: Annotated[ObjectStorage, Depends(get_storage_dep)],
    model_registry: Annotated[ModelRegistry, Depends(get_registry_dep)],
) -> HealthResponse:
    """Report the status of the database, object storage and model registry.

    Returns ``degraded`` rather than failing when a dependency is down, so a
    load balancer can distinguish "process alive" from "fully serving".
    """
    settings = get_settings()
    try:
        session.execute(text("SELECT 1"))
        database = "ok"
    except Exception:  # noqa: BLE001 - health checks report, they do not raise
        database = "unavailable"
    try:
        storage.exists("healthcheck-probe")
        storage_status = "ok"
    except Exception:  # noqa: BLE001
        storage_status = "unavailable"

    available = model_registry.list(available_only=True)
    return HealthResponse(
        status="ok" if database == "ok" and storage_status == "ok" else "degraded",
        environment=settings.ENVIRONMENT,
        software_version=settings.SOFTWARE_VERSION,
        database=database,
        storage=storage_status,
        models_registered=len(model_registry),
        models_available=len(available),
        device=get_device_manager().describe(),
    )
