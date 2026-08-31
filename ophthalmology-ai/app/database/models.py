"""ORM entities.

Relationship overview::

    Patient -> Exam -> Image -> Analysis -> ModelRun -> Prediction
                                                     -> Segmentation
                                                     -> Artifact
                                          Analysis -> Report

Privacy notes (see ``docs/SECURITY.md``):

* ``Patient`` deliberately has no name, national id (CPF) or full date of
  birth. Sites keep the identity map in their own record system and store only
  a pseudonymous ``external_ref`` here.
* Pixel data never lives in PostgreSQL. ``Image.storage_key`` and
  ``Artifact.storage_key`` point at object storage.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.enums import (
    AnalysisStatus,
    EvidenceLevel,
    Framework,
    ImageStatus,
    Laterality,
    Modality,
    ModelStatus,
    ReportFormat,
    RunStatus,
    TaskType,
    UserRole,
)
from app.database.base import Base, JSONType, TimestampMixin, UUIDPrimaryKeyMixin


class User(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A platform user. Authentication subject and audit actor."""

    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(200), nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[UserRole] = mapped_column(nullable=False, default=UserRole.VIEWER)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<User {self.id} role={self.role}>"


class Patient(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A pseudonymous patient record."""

    __tablename__ = "patients"
    __table_args__ = (
        UniqueConstraint("external_ref", name="uq_patients_external_ref"),
        CheckConstraint("birth_year IS NULL OR birth_year > 1850", name="birth_year_plausible"),
    )

    #: Site-assigned pseudonymous identifier (never a name or national id).
    external_ref: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    birth_year: Mapped[int | None] = mapped_column(Integer)
    sex: Mapped[str | None] = mapped_column(String(16))
    consent_research: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    clinical_context: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))

    exams: Mapped[list[Exam]] = relationship(
        back_populates="patient", cascade="all, delete-orphan", passive_deletes=True
    )


class Exam(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One acquisition session for one modality."""

    __tablename__ = "exams"

    patient_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), index=True, nullable=False
    )
    modality: Mapped[Modality] = mapped_column(nullable=False)
    laterality: Mapped[Laterality] = mapped_column(nullable=False, default=Laterality.UNKNOWN)
    acquired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    device_manufacturer: Mapped[str | None] = mapped_column(String(120))
    device_model: Mapped[str | None] = mapped_column(String(120))
    acquisition_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))

    patient: Mapped[Patient] = relationship(back_populates="exams")
    images: Mapped[list[Image]] = relationship(
        back_populates="exam", cascade="all, delete-orphan", passive_deletes=True
    )
    analyses: Mapped[list[Analysis]] = relationship(
        back_populates="exam", cascade="all, delete-orphan", passive_deletes=True
    )


class Image(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Metadata for one stored image (or B-scan series) of an exam."""

    __tablename__ = "images"
    __table_args__ = (Index("ix_images_exam_id_created_at", "exam_id", "created_at"),)

    exam_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("exams.id", ondelete="CASCADE"), index=True, nullable=False
    )
    storage_bucket: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    checksum_sha256: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    num_frames: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    #: Axial/lateral scale when the device reports it; required for real
    #: thickness measurements in micrometres.
    pixel_spacing_um: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    status: Mapped[ImageStatus] = mapped_column(nullable=False, default=ImageStatus.UPLOADED)
    #: Sanitized (extension-only) original name. Never the uploaded filename,
    #: which frequently embeds a patient name.
    original_extension: Mapped[str | None] = mapped_column(String(16))
    uploaded_by_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))

    exam: Mapped[Exam] = relationship(back_populates="images")


class ModelRecord(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Persisted metadata for a registered model version.

    The in-memory registry is the source of truth for *what can run*; this
    table is the durable, versioned record used by audit and reproducibility.
    """

    __tablename__ = "models"
    __table_args__ = (UniqueConstraint("model_id", "version", name="uq_models_model_id_version"),)

    model_id: Mapped[str] = mapped_column(String(128), index=True, nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    modality: Mapped[Modality] = mapped_column(nullable=False)
    task: Mapped[TaskType] = mapped_column(nullable=False)
    framework: Mapped[Framework] = mapped_column(nullable=False)
    evidence_level: Mapped[EvidenceLevel] = mapped_column(nullable=False)
    status: Mapped[ModelStatus] = mapped_column(nullable=False)
    weights_sha256: Mapped[str | None] = mapped_column(String(64))
    license_name: Mapped[str | None] = mapped_column(String(120))
    license_url: Mapped[str | None] = mapped_column(String(512))
    source_url: Mapped[str | None] = mapped_column(String(512))
    commercial_use: Mapped[str | None] = mapped_column(String(32))
    input_spec: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    output_spec: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    #: Metrics published by the model's authors. Never populated by this
    #: platform on its own - unmeasured means absent.
    reported_metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONType)


class Analysis(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """An asynchronous job running one or more models over an exam."""

    __tablename__ = "analyses"
    __table_args__ = (Index("ix_analyses_status_created_at", "status", "created_at"),)

    exam_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("exams.id", ondelete="CASCADE"), index=True, nullable=False
    )
    image_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("images.id", ondelete="SET NULL"))
    status: Mapped[AnalysisStatus] = mapped_column(nullable=False, default=AnalysisStatus.QUEUED)
    requested_models: Mapped[list[str]] = mapped_column(JSONType, nullable=False)
    pipeline_config: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    quality_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    queued_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    software_version: Mapped[str | None] = mapped_column(String(32))
    task_id: Mapped[str | None] = mapped_column(String(128))
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))

    exam: Mapped[Exam] = relationship(back_populates="analyses")
    runs: Mapped[list[ModelRun]] = relationship(
        back_populates="analysis",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ModelRun.created_at",
    )
    reports: Mapped[list[Report]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan", passive_deletes=True
    )


class ModelRun(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """One model executed against one image inside an analysis.

    Every field needed to reproduce and audit an inference lives here:
    model id + version, input hash, device, precision, batch size and latency.
    """

    __tablename__ = "model_runs"
    __table_args__ = (Index("ix_model_runs_model_id_created_at", "model_id", "created_at"),)

    analysis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True, nullable=False
    )
    image_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("images.id", ondelete="SET NULL"))
    model_record_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("models.id"))
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    model_version: Mapped[str] = mapped_column(String(32), nullable=False)
    task: Mapped[TaskType] = mapped_column(nullable=False)
    status: Mapped[RunStatus] = mapped_column(nullable=False)
    device: Mapped[str] = mapped_column(String(32), nullable=False, default="cpu")
    device_name: Mapped[str | None] = mapped_column(String(128))
    precision: Mapped[str] = mapped_column(String(8), nullable=False, default="fp32")
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    processing_time_ms: Mapped[float | None] = mapped_column(Float)
    vram_used_mb: Mapped[float | None] = mapped_column(Float)
    input_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    software_version: Mapped[str | None] = mapped_column(String(32))
    measurements: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    quality: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    warnings: Mapped[list[str] | None] = mapped_column(JSONType)
    error_message: Mapped[str | None] = mapped_column(Text)

    analysis: Mapped[Analysis] = relationship(back_populates="runs")
    predictions: Mapped[list[Prediction]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    segmentations: Mapped[list[Segmentation]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="run", cascade="all, delete-orphan", passive_deletes=True
    )


class Prediction(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A scored label produced by a classification or detection model."""

    __tablename__ = "predictions"

    model_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Optional bounding box / extra structured payload for detections.
    extra: Mapped[dict[str, Any] | None] = mapped_column(JSONType)

    run: Mapped[ModelRun] = relationship(back_populates="predictions")


class Segmentation(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A segmented structure, with its mask stored in object storage."""

    __tablename__ = "segmentations"

    model_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    label: Mapped[str] = mapped_column(String(120), nullable=False)
    storage_bucket: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False, default="image/png")
    area_px: Mapped[float | None] = mapped_column(Float)
    area_ratio: Mapped[float | None] = mapped_column(Float)
    measurements: Mapped[dict[str, Any] | None] = mapped_column(JSONType)

    run: Mapped[ModelRun] = relationship(back_populates="segmentations")


class Artifact(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A derived binary output: Grad-CAM, overlay, probability map."""

    __tablename__ = "artifacts"

    model_run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("model_runs.id", ondelete="CASCADE"), index=True, nullable=False
    )
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    storage_bucket: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(512), nullable=False)
    content_type: Mapped[str] = mapped_column(String(64), nullable=False)
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONType)

    run: Mapped[ModelRun] = relationship(back_populates="artifacts")


class Report(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """A rendered report for an analysis."""

    __tablename__ = "reports"

    analysis_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("analyses.id", ondelete="CASCADE"), index=True, nullable=False
    )
    format: Mapped[ReportFormat] = mapped_column(nullable=False)
    payload: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
    storage_bucket: Mapped[str | None] = mapped_column(String(128))
    storage_key: Mapped[str | None] = mapped_column(String(512))
    disclaimer_version: Mapped[str] = mapped_column(String(16), nullable=False, default="1.0")
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))

    analysis: Mapped[Analysis] = relationship(back_populates="reports")


class AuditLog(UUIDPrimaryKeyMixin, Base):
    """Append-only audit trail.

    Rows are written for every access to patient-linked data. No clinical
    content is stored here - only actor, action, resource identifiers and
    correlation ids.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (Index("ix_audit_logs_action_created_at", "action", "created_at"),)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"), index=True)
    actor_role: Mapped[str | None] = mapped_column(String(32))
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(64))
    resource_id: Mapped[str | None] = mapped_column(String(64), index=True)
    request_id: Mapped[str | None] = mapped_column(String(64), index=True)
    #: Salted hash of the client IP - enough to correlate abuse, not enough to
    #: re-identify a person from the log alone.
    client_ip_hash: Mapped[str | None] = mapped_column(String(64))
    outcome: Mapped[str] = mapped_column(String(16), nullable=False, default="success")
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSONType)
