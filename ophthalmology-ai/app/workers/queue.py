"""Task-queue abstraction.

Deep-learning inference must not block an HTTP request, so ``POST /analysis``
only enqueues work. Two backends implement the same interface:

* ``inline`` - runs the analysis synchronously in the calling process. Used by
  tests and single-process development, where a broker would be overhead.
* ``celery`` - hands the job to a Redis-backed worker (the GPU worker in
  Docker Compose).

The choice is configuration (``TASK_QUEUE_BACKEND``); no call site changes.
"""

from __future__ import annotations

import functools
import uuid
from abc import ABC, abstractmethod

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class TaskQueue(ABC):
    """Dispatches analysis jobs."""

    @abstractmethod
    def enqueue_analysis(self, analysis_id: uuid.UUID) -> str | None:
        """Schedule an analysis. Returns a task id when the backend has one."""


class InlineTaskQueue(TaskQueue):
    """Runs the analysis immediately, in-process."""

    def enqueue_analysis(self, analysis_id: uuid.UUID) -> str | None:
        from app.workers.tasks import execute_analysis

        logger.info("analysis_executed_inline", extra={"analysis_id": str(analysis_id)})
        execute_analysis(str(analysis_id))
        return None


class CeleryTaskQueue(TaskQueue):
    """Sends the analysis to a Celery worker."""

    def enqueue_analysis(self, analysis_id: uuid.UUID) -> str | None:
        from app.workers.celery_app import celery_app

        async_result = celery_app.send_task(
            "analysis.execute", args=[str(analysis_id)], queue="inference"
        )
        logger.info(
            "analysis_enqueued",
            extra={"analysis_id": str(analysis_id), "task_id": async_result.id},
        )
        return str(async_result.id)


def build_queue() -> TaskQueue:
    """Instantiate the configured queue backend."""
    if get_settings().TASK_QUEUE_BACKEND == "celery":
        return CeleryTaskQueue()
    return InlineTaskQueue()


@functools.lru_cache
def get_queue() -> TaskQueue:
    """Process-wide queue singleton."""
    return build_queue()
