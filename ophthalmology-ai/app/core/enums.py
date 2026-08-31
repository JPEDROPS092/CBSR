"""Domain enumerations shared by the database, schemas and AI layers."""

from __future__ import annotations

from enum import StrEnum


class UserRole(StrEnum):
    """RBAC profiles, ordered from most to least privileged."""

    ADMIN = "admin"
    DOCTOR = "doctor"
    RESEARCHER = "researcher"
    OPERATOR = "operator"
    VIEWER = "viewer"


class Modality(StrEnum):
    """Supported exam modalities."""

    FUNDUS = "fundus"
    OCT = "oct"
    OTHER = "other"


class Laterality(StrEnum):
    """Eye laterality (OD = right, OS = left, OU = both)."""

    OD = "od"
    OS = "os"
    OU = "ou"
    UNKNOWN = "unknown"


class TaskType(StrEnum):
    """What a model produces."""

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"
    DETECTION = "detection"
    REGRESSION = "regression"
    QUALITY = "quality"


class Framework(StrEnum):
    """Execution backend used by a model adapter."""

    PYTORCH = "pytorch"
    ONNX = "onnx"
    CLASSICAL = "classical"


class EvidenceLevel(StrEnum):
    """How much clinical weight a model's output may be given.

    ``HEURISTIC`` marks deterministic image-processing baselines shipped with
    the platform; they are engineering aids, never clinical evidence.
    ``RESEARCH`` marks published/trained models used outside a regulatory
    clearance. ``CLINICAL_VALIDATED`` is reserved for models with documented
    clinical validation in the deployment context - nothing shipped in this
    repository uses it.
    """

    HEURISTIC = "heuristic"
    RESEARCH = "research"
    CLINICAL_VALIDATED = "clinical_validated"


class ModelStatus(StrEnum):
    """Lifecycle state of a registered model."""

    ACTIVE = "active"
    UNAVAILABLE = "unavailable"
    DEPRECATED = "deprecated"
    DISABLED = "disabled"


class DeviceType(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"


class Precision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"


class AnalysisStatus(StrEnum):
    """Lifecycle of an analysis job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStatus(StrEnum):
    """Outcome of a single model run inside an analysis."""

    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED_QUALITY = "skipped_quality"
    SKIPPED_UNAVAILABLE = "skipped_unavailable"


class ImageStatus(StrEnum):
    UPLOADED = "uploaded"
    VALIDATED = "validated"
    REJECTED = "rejected"


class ReportFormat(StrEnum):
    JSON = "json"
    HTML = "html"
    PDF = "pdf"


class AuditAction(StrEnum):
    """Auditable events. Keep values stable - they end up in the audit trail."""

    LOGIN_SUCCESS = "login.success"
    LOGIN_FAILURE = "login.failure"
    TOKEN_REFRESH = "token.refresh"
    USER_CREATE = "user.create"
    PATIENT_CREATE = "patient.create"
    PATIENT_READ = "patient.read"
    PATIENT_UPDATE = "patient.update"
    PATIENT_DELETE = "patient.delete"
    EXAM_CREATE = "exam.create"
    EXAM_READ = "exam.read"
    IMAGE_UPLOAD = "image.upload"
    IMAGE_DOWNLOAD = "image.download"
    ANALYSIS_CREATE = "analysis.create"
    ANALYSIS_READ = "analysis.read"
    ANALYSIS_CANCEL = "analysis.cancel"
    REPORT_CREATE = "report.create"
    REPORT_READ = "report.read"
