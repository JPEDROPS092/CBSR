#!/usr/bin/env python3
"""Re-scan ``MODEL_DIR`` and persist the model catalogue to the database.

Run after installing new weights, or as a deployment step, so that model runs
can reference a durable, versioned model record.
"""

from __future__ import annotations

from app.ai.models import bootstrap_registry
from app.ai.registry import registry
from app.api.routes.models import sync_registry_to_database
from app.core.logging import configure_logging
from app.database.session import session_scope


def main() -> int:
    """Synchronize registry -> database; returns a process exit code."""
    configure_logging("INFO", "console")
    bootstrap_registry()
    with session_scope() as session:
        sync_registry_to_database(session, registry)

    for model in registry.list():
        availability = model.availability()
        state = "available" if availability.available else "UNAVAILABLE"
        print(f"{model.model_id:38s} {model.version:8s} {state}")
        if not availability.available and availability.remediation:
            print(f"    -> {availability.remediation}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
