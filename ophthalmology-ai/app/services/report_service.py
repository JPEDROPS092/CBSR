"""Report engine.

Takes the stored results of an analysis and renders a report in JSON, HTML or
PDF. The JSON document is the canonical form; HTML and PDF are renderings of
exactly the same payload, so a printed report can never disagree with the API.

Every report states the models and versions that produced it, their evidence
level and their documented limitations, plus the platform disclaimer.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy.orm import Session

from app.ai.registry import ModelRegistry
from app.ai.registry import registry as global_registry
from app.core.config import get_settings
from app.core.disclaimer import DISCLAIMER_VERSION, disclaimer_block
from app.core.enums import AuditAction, ReportFormat, RunStatus
from app.core.exceptions import AppError, NotFoundError
from app.core.logging import get_logger
from app.database.models import Analysis, Report, User
from app.database.repositories import (
    AnalysisRepository,
    ExamRepository,
    PatientRepository,
    ReportRepository,
)
from app.services.audit_service import AuditService
from app.storage import ObjectStorage, build_report_key, get_storage

logger = get_logger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"


class ReportService:
    """Builds and stores analysis reports."""

    def __init__(
        self,
        session: Session,
        *,
        storage: ObjectStorage | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self.session = session
        self.analyses = AnalysisRepository(session)
        self.exams = ExamRepository(session)
        self.patients = PatientRepository(session)
        self.reports = ReportRepository(session)
        self.audit = AuditService(session)
        self.storage = storage or get_storage()
        self.registry = global_registry if model_registry is None else model_registry
        self._jinja = Environment(
            loader=FileSystemLoader(TEMPLATE_DIR),
            autoescape=select_autoescape(["html"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    # -- payload ----------------------------------------------------------- #
    def build_payload(
        self, analysis: Analysis, *, include_explainability: bool = True
    ) -> dict[str, Any]:
        """Assemble the canonical JSON report document."""
        settings = get_settings()
        exam = self.exams.get(analysis.exam_id)
        if exam is None:
            raise NotFoundError("Exam not found for this analysis.")
        patient = self.patients.get(exam.patient_id)

        models: list[dict[str, Any]] = []
        findings: list[dict[str, Any]] = []
        measurements: dict[str, Any] = {}
        artifacts: list[dict[str, Any]] = []

        for run in analysis.runs:
            metadata = self._model_metadata(run.model_id, run.model_version)
            models.append(
                {
                    "model_id": run.model_id,
                    "version": run.model_version,
                    "task": str(run.task),
                    "status": str(run.status),
                    "processing_time_ms": run.processing_time_ms,
                    "device": run.device,
                    "precision": run.precision,
                    "input_hash": run.input_hash,
                    "evidence_level": metadata.get("evidence_level"),
                    "limitations": metadata.get("limitations"),
                    "license": metadata.get("license"),
                    "error_message": run.error_message,
                }
            )
            if run.status is not RunStatus.COMPLETED:
                continue

            for prediction in sorted(run.predictions, key=lambda p: p.rank):
                findings.append(
                    {
                        "name": prediction.label,
                        "score": round(float(prediction.score), 4),
                        "type": "prediction",
                        "model_id": run.model_id,
                        "model_version": run.model_version,
                    }
                )
            for segmentation in run.segmentations:
                findings.append(
                    {
                        "name": segmentation.label,
                        "type": "segmentation",
                        "area_px": segmentation.area_px,
                        "area_ratio": segmentation.area_ratio,
                        "measurements": segmentation.measurements,
                        "mask_url": self.storage.url_for(segmentation.storage_key),
                        "model_id": run.model_id,
                        "model_version": run.model_version,
                    }
                )
            if run.measurements:
                measurements[run.model_id] = run.measurements
            if include_explainability:
                artifacts.extend(
                    {
                        "kind": artifact.kind,
                        "url": self.storage.url_for(artifact.storage_key),
                        "model_id": run.model_id,
                        "meta": artifact.meta,
                    }
                    for artifact in run.artifacts
                )

        return {
            "generated_at": datetime.now(UTC).isoformat(),
            "software_version": settings.SOFTWARE_VERSION,
            "patient": {
                "patient_id": str(exam.patient_id),
                "external_ref": patient.external_ref if patient else None,
                "birth_year": patient.birth_year if patient else None,
                "sex": patient.sex if patient else None,
            },
            "exam": {
                "exam_id": str(exam.id),
                "modality": str(exam.modality),
                "laterality": str(exam.laterality),
                "acquired_at": exam.acquired_at.isoformat() if exam.acquired_at else None,
                "device_manufacturer": exam.device_manufacturer,
                "device_model": exam.device_model,
            },
            "analysis": {
                "analysis_id": str(analysis.id),
                "status": str(analysis.status),
                "requested_models": list(analysis.requested_models or []),
                "started_at": analysis.started_at.isoformat() if analysis.started_at else None,
                "finished_at": analysis.finished_at.isoformat() if analysis.finished_at else None,
            },
            "quality": analysis.quality_summary,
            "findings": findings,
            "measurements": measurements,
            "models": models,
            "artifacts": artifacts,
            "disclaimer": disclaimer_block(),
        }

    def _model_metadata(self, model_id: str, version: str) -> dict[str, Any]:
        """Look up evidence level, limitations and license for a run's model."""
        try:
            model = self.registry.get(model_id, version)
        except AppError:
            try:
                model = self.registry.get(model_id)
            except AppError:
                return {}
        return {
            "evidence_level": str(model.metadata.evidence_level),
            "limitations": model.metadata.limitations,
            "license": model.metadata.license.name,
        }

    # -- rendering --------------------------------------------------------- #
    def render_html(self, payload: dict[str, Any]) -> str:
        """Render the HTML report from the canonical payload."""
        template = self._jinja.get_template("report.html.j2")
        return template.render(**payload)

    def render_pdf(self, payload: dict[str, Any]) -> bytes:
        """Render a PDF report.

        Raises:
            AppError: when the optional PDF dependency is not installed.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
        except ImportError as exc:
            raise AppError(
                "PDF rendering requires the optional 'reports' extra.",
                code="pdf_unavailable",
                details={"install": "pip install -e '.[reports]'"},
            ) from exc

        import io

        buffer = io.BytesIO()
        document = SimpleDocTemplate(buffer, pagesize=A4, title="Ophthalmology AI Report")
        styles = getSampleStyleSheet()
        flow: list[Any] = [Paragraph("Ophthalmology AI Analysis Report", styles["Title"])]

        def section(title: str, lines: list[str]) -> None:
            flow.append(Spacer(1, 12))
            flow.append(Paragraph(title, styles["Heading2"]))
            for line in lines:
                flow.append(Paragraph(line, styles["BodyText"]))

        exam, analysis = payload["exam"], payload["analysis"]
        section(
            "Exam",
            [
                f"Exam: {exam['exam_id']}",
                f"Modality: {exam['modality']} ({exam['laterality']})",
                f"Analysis: {analysis['analysis_id']} - {analysis['status']}",
            ],
        )
        section(
            "Findings",
            [
                f"{item['name']}: "
                + (
                    f"score {item['score']}"
                    if item.get("score") is not None
                    else f"area {item.get('area_px')} px"
                )
                + f" ({item['model_id']} v{item['model_version']})"
                for item in payload["findings"]
            ]
            or ["No findings were produced."],
        )
        section(
            "Measurements",
            [f"{model}: {values}" for model, values in payload["measurements"].items()]
            or ["No quantitative measurements were produced."],
        )
        section("Disclaimer", [payload["disclaimer"]["en"], payload["disclaimer"]["pt"]])
        document.build(flow)
        return buffer.getvalue()

    # -- persistence ------------------------------------------------------- #
    def create(
        self,
        analysis_id: uuid.UUID,
        *,
        report_format: ReportFormat = ReportFormat.JSON,
        include_explainability: bool = True,
        actor: User | None = None,
    ) -> Report:
        """Generate and store a report for an analysis."""
        analysis = self.analyses.get_with_results(analysis_id)
        if analysis is None:
            raise NotFoundError("Analysis not found.")

        payload = self.build_payload(analysis, include_explainability=include_explainability)
        report = Report(
            analysis_id=analysis.id,
            format=report_format,
            payload=payload,
            disclaimer_version=DISCLAIMER_VERSION,
            created_by_id=actor.id if actor else None,
        )
        self.reports.add(report)

        if report_format is not ReportFormat.JSON:
            content_type = "text/html" if report_format is ReportFormat.HTML else "application/pdf"
            body = (
                self.render_html(payload).encode("utf-8")
                if report_format is ReportFormat.HTML
                else self.render_pdf(payload)
            )
            key = build_report_key(analysis.id, report.id, content_type)
            stored = self.storage.put(key, body, content_type=content_type)
            report.storage_bucket = stored.bucket
            report.storage_key = stored.key

        self.audit.record(
            AuditAction.REPORT_CREATE,
            actor=actor,
            resource_type="report",
            resource_id=report.id,
            meta={"format": str(report_format)},
        )
        self.session.commit()
        return report

    def get(self, report_id: uuid.UUID, *, actor: User | None = None) -> Report:
        """Fetch a stored report."""
        report = self.reports.get(report_id)
        if report is None:
            raise NotFoundError("Report not found.")
        self.audit.record(
            AuditAction.REPORT_READ, actor=actor, resource_type="report", resource_id=report.id
        )
        return report

    def document_url(self, report: Report) -> str | None:
        """URL of the rendered document, for HTML/PDF reports."""
        return self.storage.url_for(report.storage_key) if report.storage_key else None
