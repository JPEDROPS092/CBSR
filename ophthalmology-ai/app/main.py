"""FastAPI application factory.

``main`` wires the layers together and owns nothing else: middleware, error
translation, router mounting and startup. Business logic lives in
``app/services``, model execution in ``app/ai``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.dependencies import enforce_rate_limit
from app.api.routes import build_api_router
from app.api.routes.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.disclaimer import DISCLAIMER_EN, DISCLAIMER_PT
from app.core.exceptions import AppError
from app.core.logging import configure_logging, get_logger, log_context, new_request_id

logger = get_logger(__name__)

DESCRIPTION = f"""
Modular platform that orchestrates deep-learning models over ophthalmology
exams (fundus photography and OCT).

**{DISCLAIMER_EN}**

*{DISCLAIMER_PT}*

Pipeline: upload → validation → quality control → preprocessing → model
selection → inference → postprocessing → explainability → results → report.

Adding a model is a deployment action, not a code change: install a checkpoint
plus its manifest under `MODEL_DIR` and it appears in `GET /models`. See
`docs/MODEL_REGISTRY.md`.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Populate the model registry at startup and release GPU memory at shutdown."""
    from app.ai.devices import get_device_manager
    from app.ai.models import bootstrap_registry

    settings = get_settings()
    bootstrap_registry()
    _sync_model_catalogue()
    logger.info(
        "api_started",
        extra={
            "environment": settings.ENVIRONMENT,
            "queue_backend": settings.TASK_QUEUE_BACKEND,
            "storage_backend": settings.STORAGE_BACKEND,
            "device": get_device_manager().profile.torch_device,
        },
    )
    yield
    get_device_manager().empty_cache()
    logger.info("api_stopped")


def _sync_model_catalogue() -> None:
    """Persist registry metadata at startup, best effort.

    Model runs reference a durable ``ModelRecord``; writing it here means a
    fresh deployment has the catalogue without an operator remembering to call
    ``/models/refresh``. A database that is not migrated yet must not stop the
    API from starting, so failures are logged and swallowed.
    """
    from app.ai.registry import registry
    from app.api.routes.models import sync_registry_to_database
    from app.database.session import session_scope

    try:
        with session_scope() as session:
            sync_registry_to_database(session, registry)
    except Exception:  # noqa: BLE001 - startup must survive an unmigrated database
        logger.warning("model_catalogue_sync_skipped", exc_info=True)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application."""
    settings = settings or get_settings()
    configure_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.SOFTWARE_VERSION,
        description=DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
        max_age=600,
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Attach a request id, time the request and log its outcome.

        The request id is echoed in ``X-Request-ID`` and written into every log
        line produced while handling the request, which is what makes an
        inference traceable from an API call to a model run.
        """
        request_id = request.headers.get("x-request-id") or new_request_id()
        started = time.perf_counter()
        with log_context(request_id=request_id):
            try:
                enforce_rate_limit(request)
                response = await call_next(request)
            except AppError as exc:
                response = _error_response(exc, request_id)
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            response.headers["X-Request-ID"] = request_id
            logger.info(
                "http_request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_ms": duration_ms,
                },
            )
            return response

    _install_exception_handlers(app)

    app.include_router(health_router)
    app.include_router(build_api_router(settings.API_V1_PREFIX))
    return app


def _error_response(exc: AppError, request_id: str | None = None) -> JSONResponse:
    """Render an :class:`AppError` in the platform's error envelope."""
    headers = {"X-Request-ID": request_id} if request_id else None
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.to_dict()}, headers=headers
    )


def _install_exception_handlers(app: FastAPI) -> None:
    """Translate exceptions into the platform's error envelope.

    Unexpected exceptions are logged with a stack trace but answered with a
    generic message: internal details must never leak to an API client, and in
    a medical system an error string can carry patient data.
    """

    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("app_error", extra={"code": exc.code, "path": request.url.path})
        return _error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Report which field failed and why, but never echo the submitted
        # value back: a rejected payload may contain patient data, and it
        # would end up in the client's logs.
        errors = [
            {
                "type": error.get("type"),
                "field": ".".join(str(part) for part in error.get("loc", ())),
                "message": error.get("msg"),
            }
            for error in exc.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "Request payload is invalid.",
                    "details": {"errors": errors},
                }
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_exception", extra={"path": request.url.path})
        return JSONResponse(
            status_code=500,
            content={"error": {"code": "internal_error", "message": "Internal server error."}},
        )


app = create_app()
