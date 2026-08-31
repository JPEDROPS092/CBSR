"""Request rate limiting.

Two backends:

* ``memory`` - a per-process sliding window. Correct for a single API process
  and for development; with N processes the effective limit is N times the
  configured one.
* ``redis`` - a shared counter, which is what a multi-replica deployment needs.

The limiter fails **open**: if Redis is unreachable the request is allowed and
a warning is logged, because losing the rate limiter must not take down the
API.
"""

from __future__ import annotations

import functools
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict, deque

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RateLimiter(ABC):
    """Fixed quota per identity per window."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        self.limit = limit
        self.window_seconds = window_seconds

    @abstractmethod
    def allow(self, identity: str) -> bool:
        """Whether a request from ``identity`` may proceed now."""


class InMemoryRateLimiter(RateLimiter):
    """Sliding-window limiter held in process memory."""

    def __init__(self, limit: int, window_seconds: int) -> None:
        super().__init__(limit, window_seconds)
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            hits = self._hits[identity]
            while hits and hits[0] < cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            # Keep the table from growing without bound on a long-lived process.
            if len(self._hits) > 10_000:
                for key in [k for k, v in self._hits.items() if not v]:
                    del self._hits[key]
        return True


class RedisRateLimiter(RateLimiter):
    """Shared fixed-window limiter backed by Redis."""

    def __init__(self, limit: int, window_seconds: int, url: str) -> None:
        super().__init__(limit, window_seconds)
        import redis

        self._client = redis.Redis.from_url(url)

    def allow(self, identity: str) -> bool:
        window = int(time.time() // self.window_seconds)
        key = f"ratelimit:{identity}:{window}"
        try:
            pipeline = self._client.pipeline()
            pipeline.incr(key)
            pipeline.expire(key, self.window_seconds * 2)
            count, _ = pipeline.execute()
        except Exception:  # noqa: BLE001 - never fail closed on a cache outage
            logger.warning("rate_limiter_unavailable", exc_info=True)
            return True
        return int(count) <= self.limit


@functools.lru_cache
def get_rate_limiter() -> RateLimiter:
    """Build the configured limiter (process-wide)."""
    settings = get_settings()
    if settings.TASK_QUEUE_BACKEND == "celery":
        # A Celery deployment already runs Redis, and it is almost certainly
        # multi-replica, so use the shared counter.
        try:
            return RedisRateLimiter(
                settings.RATE_LIMIT_REQUESTS, settings.RATE_LIMIT_WINDOW_SECONDS, settings.REDIS_URL
            )
        except Exception:  # noqa: BLE001
            logger.warning("redis_rate_limiter_unavailable_falling_back_to_memory")
    return InMemoryRateLimiter(settings.RATE_LIMIT_REQUESTS, settings.RATE_LIMIT_WINDOW_SECONDS)
