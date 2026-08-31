"""Audit trail.

Every action that touches patient-linked data is recorded: who, what, which
resource, which request. Clinical content is never written to the audit trail -
it holds identifiers and outcomes only.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.enums import AuditAction
from app.core.logging import get_logger, request_id_var
from app.database.base import utcnow
from app.database.models import AuditLog, User
from app.database.repositories import AuditLogRepository

logger = get_logger(__name__)


def hash_client_ip(ip: str | None) -> str | None:
    """Salted hash of a client IP.

    Keeps abuse correlation possible without storing an address that, combined
    with clinic records, could re-identify a person.
    """
    if not ip:
        return None
    salt = get_settings().JWT_SECRET.encode("utf-8")
    return hashlib.sha256(salt + ip.encode("utf-8")).hexdigest()[:32]


class AuditService:
    """Writes audit entries."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.repository = AuditLogRepository(session)

    def record(
        self,
        action: AuditAction,
        *,
        actor: User | None = None,
        resource_type: str | None = None,
        resource_id: uuid.UUID | str | None = None,
        outcome: str = "success",
        client_ip: str | None = None,
        meta: dict[str, Any] | None = None,
    ) -> AuditLog:
        """Append one entry to the audit trail."""
        entry = AuditLog(
            created_at=utcnow(),
            actor_user_id=actor.id if actor else None,
            actor_role=str(actor.role) if actor else None,
            action=str(action),
            resource_type=resource_type,
            resource_id=str(resource_id) if resource_id else None,
            request_id=request_id_var.get(),
            client_ip_hash=hash_client_ip(client_ip),
            outcome=outcome,
            meta=meta,
        )
        self.repository.add(entry)
        logger.info(
            "audit",
            extra={
                "action": str(action),
                "resource_type": resource_type,
                "resource_id": str(resource_id) if resource_id else None,
                "outcome": outcome,
            },
        )
        return entry
