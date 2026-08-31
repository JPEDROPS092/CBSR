"""Celery application for GPU/CPU inference workers.

Run with::

    celery -A app.workers.celery_app:celery_app worker -Q inference -c 1

Concurrency is 1 per GPU by design: two processes sharing one device serialize
on it anyway and multiply VRAM use.
"""

from __future__ import annotations

from typing import Any

from celery import Celery
from celery.signals import worker_process_init

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger

settings = get_settings()
logger = get_logger(__name__)

celery_app = Celery(
    "ophthalmology_ai",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    # A GPU worker holds one model set in memory; prefetching more than one job
    # only delays other workers.
    worker_prefetch_multiplier=1,
    task_default_queue="inference",
    task_time_limit=1800,
    task_soft_time_limit=1500,
    result_expires=86400,
)


@worker_process_init.connect
def _on_worker_start(**_kwargs: Any) -> None:
    """Configure logging and populate the model registry in each worker process."""
    configure_logging(settings.LOG_LEVEL, settings.LOG_FORMAT)
    from app.ai.models import bootstrap_registry

    bootstrap_registry()
    logger.info("worker_ready", extra={"queue": "inference"})


@celery_app.task(name="analysis.execute", bind=True, max_retries=2)
def execute_analysis_task(self: Any, analysis_id: str) -> dict[str, str]:
    """Celery wrapper around :func:`app.workers.tasks.execute_analysis`."""
    from app.workers.tasks import execute_analysis

    return execute_analysis(analysis_id)


@celery_app.task(name="models.warm_up")
def warm_up_models_task() -> dict[str, int]:
    """Celery wrapper around :func:`app.workers.tasks.warm_up_models`."""
    from app.workers.tasks import warm_up_models

    return warm_up_models()
