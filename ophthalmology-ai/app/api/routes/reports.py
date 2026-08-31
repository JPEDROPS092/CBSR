"""Report routes."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Response, status

from app.api.dependencies import CurrentUser, get_report_service, require_permission
from app.core.enums import ReportFormat
from app.core.exceptions import NotFoundError
from app.core.security import Permission
from app.schemas.report import ReportCreate, ReportRead
from app.services.report_service import ReportService

router = APIRouter(prefix="/reports", tags=["reports"])

ReportDep = Annotated[ReportService, Depends(get_report_service)]


def _to_schema(service: ReportService, report: object) -> ReportRead:
    return ReportRead(
        id=report.id,  # type: ignore[attr-defined]
        analysis_id=report.analysis_id,  # type: ignore[attr-defined]
        format=report.format,  # type: ignore[attr-defined]
        payload=report.payload,  # type: ignore[attr-defined]
        document_url=service.document_url(report),  # type: ignore[arg-type]
        disclaimer_version=report.disclaimer_version,  # type: ignore[attr-defined]
        created_at=report.created_at,  # type: ignore[attr-defined]
    )


@router.post(
    "/{analysis_id}",
    response_model=ReportRead,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a report for an analysis",
    dependencies=[Depends(require_permission(Permission.REPORT_WRITE))],
)
def create_report(
    analysis_id: uuid.UUID, payload: ReportCreate, service: ReportDep, user: CurrentUser
) -> ReportRead:
    """Render a report from a completed analysis.

    ``json`` returns the canonical document inline; ``html`` and ``pdf`` also
    store a rendered file in object storage and return its URL.
    """
    report = service.create(
        analysis_id,
        report_format=payload.format,
        include_explainability=payload.include_explainability,
        actor=user,
    )
    return _to_schema(service, report)


@router.get(
    "/{report_id}",
    response_model=ReportRead,
    summary="Fetch a report",
    dependencies=[Depends(require_permission(Permission.REPORT_READ))],
)
def get_report(report_id: uuid.UUID, service: ReportDep, user: CurrentUser) -> ReportRead:
    """Fetch a previously generated report."""
    report = service.get(report_id, actor=user)
    service.session.commit()
    return _to_schema(service, report)


@router.get(
    "/{report_id}/document",
    summary="Download a rendered report document",
    dependencies=[Depends(require_permission(Permission.REPORT_READ))],
    response_class=Response,
)
def download_report_document(
    report_id: uuid.UUID, service: ReportDep, user: CurrentUser
) -> Response:
    """Stream the rendered HTML or PDF document of a report."""
    report = service.get(report_id, actor=user)
    service.session.commit()
    if not report.storage_key:
        raise NotFoundError(
            "This report has no rendered document.",
            details={"format": str(report.format), "hint": "Generate it as html or pdf."},
        )
    body = service.storage.get(report.storage_key)
    media_type = "text/html" if report.format is ReportFormat.HTML else "application/pdf"
    return Response(content=body, media_type=media_type)
