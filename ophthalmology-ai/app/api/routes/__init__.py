"""API route modules."""

from fastapi import APIRouter

from app.api.routes import analysis, auth, exams, models, objects, patients, reports


def build_api_router(prefix: str) -> APIRouter:
    """Assemble the versioned API router."""
    router = APIRouter(prefix=prefix)
    router.include_router(auth.router)
    router.include_router(patients.router)
    router.include_router(exams.router)
    router.include_router(analysis.router)
    router.include_router(models.router)
    router.include_router(reports.router)
    router.include_router(objects.router)
    return router


__all__ = ["build_api_router"]
