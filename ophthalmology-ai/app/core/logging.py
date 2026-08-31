"""Structured logging.

Emits one JSON object per line so logs can be shipped to Loki/ELK/Cloud
Logging without a parser. Correlation identifiers (request, analysis, model)
are carried in :mod:`contextvars` and injected into every record, which is what
makes an inference traceable end to end.

Privacy: this module never logs request bodies or patient attributes. Only
opaque UUIDs are correlated.
"""

from __future__ import annotations

import json
import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
analysis_id_var: ContextVar[str | None] = ContextVar("analysis_id", default=None)
model_id_var: ContextVar[str | None] = ContextVar("model_id", default=None)
user_id_var: ContextVar[str | None] = ContextVar("user_id", default=None)

_RESERVED = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
    | {"asctime", "message", "taskName"}
)


class JsonFormatter(logging.Formatter):
    """Format records as single-line JSON documents."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in (
            ("request_id", request_id_var.get()),
            ("analysis_id", analysis_id_var.get()),
            ("model_id", model_id_var.get()),
            ("user_id", user_id_var.get()),
        ):
            if value:
                payload[key] = value
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """Human-readable formatter for local development."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<8} {record.name}: "
        base += record.getMessage()
        extras = {
            k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")
        }
        rid = request_id_var.get()
        if rid:
            extras["request_id"] = rid
        if extras:
            base += " | " + " ".join(f"{k}={v}" for k, v in extras.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """Install the root logging handler. Idempotent."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else ConsoleFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # Uvicorn installs its own handlers; route them through ours instead.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)


def new_request_id() -> str:
    """Generate a correlation id for an inbound request."""
    return uuid.uuid4().hex


@contextmanager
def log_context(
    *,
    request_id: str | None = None,
    analysis_id: str | None = None,
    model_id: str | None = None,
    user_id: str | None = None,
) -> Iterator[None]:
    """Bind correlation identifiers for the duration of the block."""
    tokens = []
    for var, value in (
        (request_id_var, request_id),
        (analysis_id_var, analysis_id),
        (model_id_var, model_id),
        (user_id_var, user_id),
    ):
        if value is not None:
            tokens.append((var, var.set(value)))
    try:
        yield
    finally:
        for var, token in reversed(tokens):
            var.reset(token)
