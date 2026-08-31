"""Analysis, model-run and result repositories."""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database.models import Analysis, Artifact, ModelRun, Prediction, Segmentation
from app.database.repositories.base import BaseRepository


class AnalysisRepository(BaseRepository[Analysis]):
    model = Analysis

    def get_with_results(self, analysis_id: uuid.UUID) -> Analysis | None:
        """Fetch an analysis eagerly loaded with every nested result."""
        stmt = (
            select(Analysis)
            .where(Analysis.id == analysis_id)
            .options(
                selectinload(Analysis.runs).selectinload(ModelRun.predictions),
                selectinload(Analysis.runs).selectinload(ModelRun.segmentations),
                selectinload(Analysis.runs).selectinload(ModelRun.artifacts),
            )
        )
        return self.session.execute(stmt).scalars().first()

    def list_for_exam(
        self, exam_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> Sequence[Analysis]:
        """List analyses of an exam, newest first."""
        stmt = (
            select(Analysis)
            .where(Analysis.exam_id == exam_id)
            .order_by(Analysis.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        return self.session.execute(stmt).scalars().all()


class ModelRunRepository(BaseRepository[ModelRun]):
    model = ModelRun


class PredictionRepository(BaseRepository[Prediction]):
    model = Prediction


class SegmentationRepository(BaseRepository[Segmentation]):
    model = Segmentation


class ArtifactRepository(BaseRepository[Artifact]):
    model = Artifact
