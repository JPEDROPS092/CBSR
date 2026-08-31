"""Worker entry points.

These are plain functions with no Celery import, so the inline queue (and the
test suite) can call them directly. ``celery_app`` registers thin task wrappers
around them.
"""

from __future__ import annotations

import uuid

from app.core.logging import get_logger, log_context
from app.database.session import SessionLocal
from app.services.analysis_service import AnalysisService

logger = get_logger(__name__)


def execute_analysis(analysis_id: str) -> dict[str, str]:
    """Run one analysis to completion in its own database session.

    Returns:
        ``{"analysis_id": ..., "status": ...}`` - the terminal state. Failures
        are recorded on the analysis row, so this only raises if the row itself
        cannot be read.
    """
    identifier = uuid.UUID(analysis_id)
    with log_context(analysis_id=analysis_id):
        session = SessionLocal()
        try:
            service = AnalysisService(session)
            analysis = service.execute(identifier)
            return {"analysis_id": analysis_id, "status": str(analysis.status)}
        finally:
            session.close()


def warm_up_models() -> dict[str, int]:
    """Load every available model once, so the first real request is not slow.

    Intended as a worker startup hook on GPU deployments, where loading a
    checkpoint can take tens of seconds.
    """
    from app.ai.models import bootstrap_registry
    from app.ai.registry import registry

    bootstrap_registry()
    loaded = 0
    for model in registry.list(available_only=True):
        try:
            model.load()
            loaded += 1
        except Exception:  # noqa: BLE001 - a broken model must not stop the worker
            logger.warning("model_warmup_failed", extra={"model_id": model.model_id})
    return {"loaded": loaded, "registered": len(registry)}
