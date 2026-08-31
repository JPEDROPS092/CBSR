"""Analysis orchestration and persistence.

Responsibilities split:

* :class:`~app.ai.inference.InferenceEngine` runs models over pixels and
  returns in-memory results;
* this service owns the *job*: validation, queueing, status transitions,
  writing masks and artifacts to object storage and rows to the database.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.ai.inference import InferenceEngine, ModelSelection, PipelineConfig, PipelineOutcome
from app.ai.registry import ModelRegistry
from app.ai.registry import registry as global_registry
from app.ai.results import ArtifactPayload, MaskPayload, ModelResult
from app.core.config import get_settings
from app.core.disclaimer import DISCLAIMER_EN
from app.core.enums import AnalysisStatus, AuditAction, RunStatus
from app.core.exceptions import ConflictError, NotFoundError
from app.core.logging import get_logger, log_context
from app.database.base import utcnow
from app.database.models import Analysis, Artifact, Image, ModelRun, Prediction, Segmentation, User
from app.database.repositories import (
    AnalysisRepository,
    ExamRepository,
    ImageRepository,
    ModelRecordRepository,
)
from app.schemas.analysis import (
    AnalysisRead,
    ArtifactRead,
    ModelRunRead,
    PredictionRead,
    SegmentationRead,
)
from app.services.audit_service import AuditService
from app.services.exam_service import ExamService
from app.storage import ObjectStorage, build_result_key, get_storage
from app.workers.queue import TaskQueue, get_queue

logger = get_logger(__name__)


class AnalysisService:
    """Creates, executes and reads analyses."""

    def __init__(
        self,
        session: Session,
        *,
        storage: ObjectStorage | None = None,
        model_registry: ModelRegistry | None = None,
        queue: TaskQueue | None = None,
        engine: InferenceEngine | None = None,
    ) -> None:
        self.session = session
        self.analyses = AnalysisRepository(session)
        self.exams = ExamRepository(session)
        self.images = ImageRepository(session)
        self.model_records = ModelRecordRepository(session)
        self.audit = AuditService(session)
        self.storage = storage or get_storage()
        self.registry = global_registry if model_registry is None else model_registry
        self.queue = queue or get_queue()
        self.engine = engine or InferenceEngine(self.registry)
        self.exam_service = ExamService(session, storage=self.storage, model_registry=self.registry)

    # -- creation ---------------------------------------------------------- #
    def create(
        self,
        *,
        exam_id: uuid.UUID,
        image_id: uuid.UUID | None = None,
        models: list[str] | None = None,
        quality_gate: bool | None = None,
        explainability: bool | None = None,
        frame_selection: str = "middle",
        actor: User | None = None,
    ) -> Analysis:
        """Queue an analysis.

        The requested models are resolved against the registry *before*
        queueing, so an unknown model id fails immediately with a 404 instead
        of surfacing minutes later as a failed job.

        Raises:
            NotFoundError: unknown exam, image, or model id.
            ConflictError: the exam has no image to analyse.
        """
        settings = get_settings()
        exam = self.exams.get(exam_id)
        if exam is None:
            raise NotFoundError("Exam not found.")

        image = self._select_image(exam_id, image_id)
        selections = [ModelSelection.parse(item) for item in (models or [])]
        for selection in selections:
            # Raises ModelNotFoundError (404) when the id is unknown.
            self.registry.get(selection.model_id, selection.version)

        analysis = Analysis(
            exam_id=exam_id,
            image_id=image.id,
            status=AnalysisStatus.QUEUED,
            requested_models=[selection.as_string() for selection in selections],
            pipeline_config={
                "quality_gate": quality_gate,
                "explainability": explainability,
                "frame_selection": frame_selection,
            },
            queued_at=utcnow(),
            software_version=settings.SOFTWARE_VERSION,
            created_by_id=actor.id if actor else None,
        )
        self.analyses.add(analysis)
        self.audit.record(
            AuditAction.ANALYSIS_CREATE,
            actor=actor,
            resource_type="analysis",
            resource_id=analysis.id,
            meta={"exam_id": str(exam_id), "models": analysis.requested_models},
        )
        # The worker reads this row from another session, so it must be
        # committed before the job is dispatched.
        self.session.commit()

        analysis.task_id = self.queue.enqueue_analysis(analysis.id)
        if analysis.task_id:
            self.session.commit()
        self.session.refresh(analysis)
        return analysis

    def _select_image(self, exam_id: uuid.UUID, image_id: uuid.UUID | None) -> Image:
        """Resolve which image an analysis runs on."""
        if image_id is not None:
            image = self.images.get(image_id)
            if image is None or image.exam_id != exam_id:
                raise NotFoundError("Image not found for this exam.")
            return image
        images = self.images.list_for_exam(exam_id)
        if not images:
            raise ConflictError("Exam has no uploaded image to analyse.")
        return images[-1]

    # -- execution --------------------------------------------------------- #
    def execute(self, analysis_id: uuid.UUID) -> Analysis:
        """Run an analysis to completion and persist everything it produced.

        Called by the worker (or inline). Failures are recorded on the analysis
        row rather than raised, so the API can always report a terminal state.
        """
        analysis = self.analyses.get(analysis_id)
        if analysis is None:
            raise NotFoundError("Analysis not found.")
        if analysis.status in (AnalysisStatus.COMPLETED, AnalysisStatus.CANCELLED):
            return analysis

        with log_context(analysis_id=str(analysis.id)):
            analysis.status = AnalysisStatus.PROCESSING
            analysis.started_at = utcnow()
            self.session.commit()

            try:
                outcome = self._run_pipeline(analysis)
            except Exception as exc:  # noqa: BLE001 - terminal state must be recorded
                logger.exception("analysis_failed", extra={"analysis_id": str(analysis.id)})
                analysis.status = AnalysisStatus.FAILED
                analysis.error_code = getattr(exc, "code", "internal_error")
                # Error text is written for an operator and carries no patient data.
                analysis.error_message = getattr(exc, "message", "Analysis failed.")
                analysis.finished_at = utcnow()
                self.session.commit()
                return analysis

            analysis.quality_summary = outcome.summary()
            analysis.status = AnalysisStatus.COMPLETED
            analysis.finished_at = utcnow()
            self.session.commit()
            logger.info(
                "analysis_completed",
                extra={
                    "analysis_id": str(analysis.id),
                    "models_completed": len(outcome.completed),
                    "gate_passed": outcome.gate_passed,
                },
            )
        return analysis

    def _run_pipeline(self, analysis: Analysis) -> PipelineOutcome:
        """Load the image, run the engine and persist each model run."""
        exam = self.exams.get(analysis.exam_id)
        if exam is None:
            raise NotFoundError("Exam not found.")
        image = self.images.get(analysis.image_id) if analysis.image_id else None
        if image is None:
            raise ConflictError("Analysis has no image to run on.")

        config_data = analysis.pipeline_config or {}
        config = PipelineConfig.from_request(
            list(analysis.requested_models or []),
            quality_gate=config_data.get("quality_gate"),
            explainability=config_data.get("explainability"),
            frame_selection=config_data.get("frame_selection", "middle"),
        )
        exam_image = self.exam_service.load_exam_image(
            image, exam, frame_selection=config.frame_selection
        )
        outcome = self.engine.run(exam_image, config, modality=exam.modality)

        for result in outcome.results:
            self._persist_result(analysis, image, result)
        return outcome

    def _persist_result(self, analysis: Analysis, image: Image, result: ModelResult) -> None:
        """Write one model result: run row, predictions, masks, artifacts."""
        settings = get_settings()
        record = self.model_records.get_by_model_id(result.model_id, result.model_version)
        run = ModelRun(
            analysis_id=analysis.id,
            image_id=image.id,
            model_record_id=record.id if record else None,
            model_id=result.model_id,
            model_version=result.model_version,
            task=result.task,
            status=result.status,
            device=result.device_info.device,
            device_name=result.device_info.device_name,
            precision=str(result.device_info.precision),
            batch_size=result.device_info.batch_size,
            processing_time_ms=result.processing_time_ms,
            vram_used_mb=result.device_info.vram_used_mb,
            input_hash=result.input_hash,
            software_version=settings.SOFTWARE_VERSION,
            measurements=result.measurements or None,
            quality=result.quality.model_dump() if result.quality else None,
            warnings=result.warnings or None,
            error_message=result.error_message,
        )
        self.session.add(run)
        self.session.flush()

        for prediction in result.predictions:
            self.session.add(
                Prediction(
                    model_run_id=run.id,
                    label=prediction.label,
                    score=prediction.score,
                    rank=prediction.rank,
                    extra=prediction.extra,
                )
            )
        for detection in result.detections:
            self.session.add(
                Prediction(
                    model_run_id=run.id,
                    label=detection.label,
                    score=detection.score,
                    rank=0,
                    extra={"box": detection.box.model_dump()},
                )
            )
        for mask in result.segmentations:
            self._persist_mask(analysis, run, mask)
        for artifact in result.artifacts:
            self._persist_artifact(analysis, run, artifact)
        self.session.flush()

    def _persist_mask(self, analysis: Analysis, run: ModelRun, mask: MaskPayload) -> None:
        """Upload a mask to object storage and register it."""
        key = build_result_key(analysis.id, run.id, "masks", mask.label, mask.content_type)
        stored = self.storage.put(key, mask.data, content_type=mask.content_type)
        self.session.add(
            Segmentation(
                model_run_id=run.id,
                label=mask.label,
                storage_bucket=stored.bucket,
                storage_key=stored.key,
                content_type=mask.content_type,
                area_px=mask.area_px,
                area_ratio=mask.area_ratio,
                measurements=mask.measurements or None,
            )
        )

    def _persist_artifact(
        self, analysis: Analysis, run: ModelRun, artifact: ArtifactPayload
    ) -> None:
        """Upload an explainability artifact and register it."""
        key = build_result_key(
            analysis.id, run.id, artifact.kind, artifact.name, artifact.content_type
        )
        stored = self.storage.put(key, artifact.data, content_type=artifact.content_type)
        self.session.add(
            Artifact(
                model_run_id=run.id,
                kind=artifact.kind,
                storage_bucket=stored.bucket,
                storage_key=stored.key,
                content_type=artifact.content_type,
                meta=artifact.meta or None,
            )
        )

    # -- reading ----------------------------------------------------------- #
    def get(self, analysis_id: uuid.UUID, *, actor: User | None = None) -> Analysis:
        """Fetch an analysis with all nested results."""
        analysis = self.analyses.get_with_results(analysis_id)
        if analysis is None:
            raise NotFoundError("Analysis not found.")
        self.audit.record(
            AuditAction.ANALYSIS_READ,
            actor=actor,
            resource_type="analysis",
            resource_id=analysis.id,
        )
        return analysis

    def cancel(self, analysis_id: uuid.UUID, *, actor: User | None = None) -> Analysis:
        """Cancel a queued analysis.

        Raises:
            ConflictError: when the analysis already started or finished.
        """
        analysis = self.analyses.get(analysis_id)
        if analysis is None:
            raise NotFoundError("Analysis not found.")
        if analysis.status is not AnalysisStatus.QUEUED:
            raise ConflictError(
                "Only a queued analysis can be cancelled.",
                details={"status": str(analysis.status)},
            )
        analysis.status = AnalysisStatus.CANCELLED
        analysis.finished_at = utcnow()
        self.audit.record(
            AuditAction.ANALYSIS_CANCEL,
            actor=actor,
            resource_type="analysis",
            resource_id=analysis.id,
        )
        self.session.commit()
        return analysis

    def to_schema(self, analysis: Analysis) -> AnalysisRead:
        """Serialize an analysis, resolving storage keys into URLs."""
        runs = [self._run_to_schema(run) for run in analysis.runs]
        return AnalysisRead(
            id=analysis.id,
            exam_id=analysis.exam_id,
            image_id=analysis.image_id,
            status=analysis.status,
            requested_models=list(analysis.requested_models or []),
            quality_summary=analysis.quality_summary,
            queued_at=analysis.queued_at,
            started_at=analysis.started_at,
            finished_at=analysis.finished_at,
            error_code=analysis.error_code,
            error_message=analysis.error_message,
            software_version=analysis.software_version,
            created_at=analysis.created_at,
            models=runs,
            disclaimer=DISCLAIMER_EN,
        )

    def _run_to_schema(self, run: ModelRun) -> ModelRunRead:
        return ModelRunRead(
            id=run.id,
            model_id=run.model_id,
            model_version=run.model_version,
            task=run.task,
            status=run.status,
            device=run.device,
            device_name=run.device_name,
            precision=run.precision,
            batch_size=run.batch_size,
            processing_time_ms=run.processing_time_ms,
            input_hash=run.input_hash,
            software_version=run.software_version,
            predictions=[
                PredictionRead.model_validate(prediction)
                for prediction in sorted(run.predictions, key=lambda p: p.rank)
            ],
            segmentations=[
                SegmentationRead(
                    id=segmentation.id,
                    label=segmentation.label,
                    mask_url=self.storage.url_for(segmentation.storage_key),
                    content_type=segmentation.content_type,
                    area_px=segmentation.area_px,
                    area_ratio=segmentation.area_ratio,
                    measurements=segmentation.measurements,
                )
                for segmentation in run.segmentations
            ],
            artifacts=[
                ArtifactRead(
                    id=artifact.id,
                    kind=artifact.kind,
                    artifact_url=self.storage.url_for(artifact.storage_key),
                    content_type=artifact.content_type,
                    meta=artifact.meta,
                )
                for artifact in run.artifacts
            ],
            measurements=run.measurements,
            quality=run.quality,
            warnings=run.warnings,
            error_message=run.error_message,
        )

    def summary_for_report(self, analysis: Analysis) -> dict[str, Any]:
        """Condensed result view used by the report engine."""
        findings: list[dict[str, Any]] = []
        measurements: dict[str, Any] = {}
        for run in analysis.runs:
            if run.status is not RunStatus.COMPLETED:
                continue
            for prediction in sorted(run.predictions, key=lambda p: p.rank)[:3]:
                findings.append(
                    {
                        "name": prediction.label,
                        "score": prediction.score,
                        "model_id": run.model_id,
                        "model_version": run.model_version,
                    }
                )
            for segmentation in run.segmentations:
                findings.append(
                    {
                        "name": segmentation.label,
                        "area_px": segmentation.area_px,
                        "area_ratio": segmentation.area_ratio,
                        "model_id": run.model_id,
                        "model_version": run.model_version,
                    }
                )
            if run.measurements:
                measurements[run.model_id] = run.measurements
        return {"findings": findings, "measurements": measurements}
