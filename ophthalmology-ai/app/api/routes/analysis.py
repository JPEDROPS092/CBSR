"""Analysis routes."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, status

from app.api.dependencies import CurrentUser, get_analysis_service, require_permission
from app.core.security import Permission
from app.schemas.analysis import AnalysisAccepted, AnalysisCreate, AnalysisRead
from app.services.analysis_service import AnalysisService

router = APIRouter(tags=["analysis"])

AnalysisDep = Annotated[AnalysisService, Depends(get_analysis_service)]


@router.post(
    "/analysis",
    response_model=AnalysisAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue an analysis",
    dependencies=[Depends(require_permission(Permission.ANALYSIS_RUN))],
)
def create_analysis(
    payload: AnalysisCreate, service: AnalysisDep, user: CurrentUser
) -> AnalysisAccepted:
    """Queue one or more models against an exam image.

    Returns immediately with ``status = queued``; poll
    ``GET /analysis/{analysis_id}`` for results. An unknown model id is
    rejected here rather than failing later inside the worker.
    """
    analysis = service.create(
        exam_id=payload.exam_id,
        image_id=payload.image_id,
        models=payload.models,
        quality_gate=payload.quality_gate,
        explainability=payload.explainability,
        frame_selection=payload.frame_selection,
        actor=user,
    )
    return AnalysisAccepted(
        analysis_id=analysis.id, status=analysis.status, task_id=analysis.task_id
    )


@router.get(
    "/analysis/{analysis_id}",
    response_model=AnalysisRead,
    summary="Fetch an analysis and its results",
    dependencies=[Depends(require_permission(Permission.ANALYSIS_READ))],
)
def get_analysis(analysis_id: uuid.UUID, service: AnalysisDep, user: CurrentUser) -> AnalysisRead:
    """Fetch an analysis with every model run, prediction, mask and artifact."""
    analysis = service.get(analysis_id, actor=user)
    payload = service.to_schema(analysis)
    service.session.commit()
    return payload


@router.post(
    "/analysis/{analysis_id}/cancel",
    response_model=AnalysisRead,
    summary="Cancel a queued analysis",
    dependencies=[Depends(require_permission(Permission.ANALYSIS_RUN))],
)
def cancel_analysis(
    analysis_id: uuid.UUID, service: AnalysisDep, user: CurrentUser
) -> AnalysisRead:
    """Cancel an analysis that has not started yet."""
    service.cancel(analysis_id, actor=user)
    return service.to_schema(service.get(analysis_id, actor=user))


@router.get(
    "/exams/{exam_id}/analyses",
    response_model=list[AnalysisRead],
    summary="List an exam's analyses",
    dependencies=[Depends(require_permission(Permission.ANALYSIS_READ))],
)
def list_exam_analyses(
    exam_id: uuid.UUID,
    service: AnalysisDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[AnalysisRead]:
    """List analyses for an exam, newest first."""
    analyses = service.analyses.list_for_exam(exam_id, limit=limit, offset=offset)
    return [service.to_schema(service.get(analysis.id)) for analysis in analyses]
