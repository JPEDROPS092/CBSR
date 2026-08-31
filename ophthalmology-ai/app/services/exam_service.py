"""Exam and image-upload service.

Upload path: validate -> hash -> decode -> store bytes in object storage ->
persist metadata -> screen quality (advisory).

Uploaded filenames are discarded on purpose: in practice they routinely embed
patient names. Only the extension is kept.
"""

from __future__ import annotations

import hashlib
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.ai.inference import select_frame
from app.ai.preprocessing import ExamImage, decode_image, decode_series
from app.ai.registry import ModelRegistry
from app.ai.registry import registry as global_registry
from app.core.config import get_settings
from app.core.enums import AuditAction, ImageStatus, Modality, TaskType
from app.core.exceptions import (
    NotFoundError,
    PayloadTooLargeError,
    UnsupportedMediaError,
    ValidationError,
)
from app.core.logging import get_logger
from app.database.models import Exam, Image, User
from app.database.repositories import ExamRepository, ImageRepository, PatientRepository
from app.schemas.exam import ExamCreate
from app.services.audit_service import AuditService
from app.storage import ObjectStorage, build_image_key, get_storage

logger = get_logger(__name__)

#: Magic-number prefixes accepted for upload, mapped to their canonical type.
#: Content type is decided by inspecting the bytes, never by trusting the
#: client's Content-Type header or the file extension.
MAGIC_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
]

MULTI_FRAME_TYPES = {"image/tiff"}


def sniff_content_type(data: bytes) -> str:
    """Identify an upload by its magic number.

    Raises:
        UnsupportedMediaError: when the bytes match no accepted format.
    """
    for signature, content_type in MAGIC_SIGNATURES:
        if data.startswith(signature):
            return content_type
    # DICOM files carry "DICM" at offset 128.
    if len(data) > 132 and data[128:132] == b"DICM":
        return "application/dicom"
    raise UnsupportedMediaError(
        "File format is not supported.",
        details={"accepted": sorted({ct for _, ct in MAGIC_SIGNATURES} | {"application/dicom"})},
    )


class ExamService:
    """Creates exams and ingests their images."""

    def __init__(
        self,
        session: Session,
        *,
        storage: ObjectStorage | None = None,
        model_registry: ModelRegistry | None = None,
    ) -> None:
        self.session = session
        self.exams = ExamRepository(session)
        self.images = ImageRepository(session)
        self.patients = PatientRepository(session)
        self.audit = AuditService(session)
        self.storage = storage or get_storage()
        self.registry = global_registry if model_registry is None else model_registry

    # -- exams ------------------------------------------------------------- #
    def create(self, payload: ExamCreate, *, actor: User | None = None) -> Exam:
        """Create an exam for an existing patient."""
        if self.patients.get(payload.patient_id) is None:
            raise NotFoundError("Patient not found.")
        exam = Exam(
            patient_id=payload.patient_id,
            modality=payload.modality,
            laterality=payload.laterality,
            acquired_at=payload.acquired_at,
            device_manufacturer=payload.device_manufacturer,
            device_model=payload.device_model,
            acquisition_metadata=payload.acquisition_metadata,
            created_by_id=actor.id if actor else None,
        )
        self.exams.add(exam)
        self.audit.record(
            AuditAction.EXAM_CREATE,
            actor=actor,
            resource_type="exam",
            resource_id=exam.id,
            meta={"modality": str(payload.modality)},
        )
        return exam

    def get(self, exam_id: uuid.UUID, *, actor: User | None = None) -> Exam:
        """Fetch an exam."""
        exam = self.exams.get(exam_id)
        if exam is None:
            raise NotFoundError("Exam not found.")
        return exam

    def list_for_patient(
        self, patient_id: uuid.UUID, *, limit: int = 50, offset: int = 0
    ) -> Sequence[Exam]:
        """List a patient's exams."""
        return self.exams.list_for_patient(patient_id, limit=limit, offset=offset)

    def list_images(self, exam_id: uuid.UUID) -> Sequence[Image]:
        """List an exam's images."""
        return self.images.list_for_exam(exam_id)

    # -- uploads ----------------------------------------------------------- #
    def upload_image(
        self,
        exam_id: uuid.UUID,
        data: bytes,
        *,
        filename: str | None = None,
        actor: User | None = None,
        run_quality_check: bool = True,
    ) -> tuple[Image, dict[str, Any] | None]:
        """Validate, store and register one image for an exam.

        Returns:
            The persisted image and, when a quality model exists for the
            modality, its advisory quality report.

        Raises:
            PayloadTooLargeError: file exceeds ``MAX_UPLOAD_BYTES``.
            UnsupportedMediaError: unrecognized or disallowed format.
            ValidationError: bytes are not a decodable image.
        """
        settings = get_settings()
        exam = self.get(exam_id, actor=actor)

        if not data:
            raise ValidationError("Uploaded file is empty.")
        if len(data) > settings.MAX_UPLOAD_BYTES:
            raise PayloadTooLargeError(
                "Uploaded file exceeds the configured size limit.",
                details={"max_bytes": settings.MAX_UPLOAD_BYTES, "received_bytes": len(data)},
            )
        content_type = sniff_content_type(data)
        if content_type not in settings.ALLOWED_IMAGE_MIME_TYPES:
            raise UnsupportedMediaError(
                "This file type is not accepted by this deployment.",
                details={"detected": content_type},
            )
        if content_type == "application/dicom":
            raise UnsupportedMediaError(
                "DICOM ingestion is not implemented yet.",
                details={
                    "remediation": (
                        "Export the pixel data as PNG/TIFF, or add a DICOM reader that "
                        "de-identifies the header before storage."
                    )
                },
            )

        checksum = hashlib.sha256(data).hexdigest()
        existing = self.images.get_by_checksum(exam_id, checksum)
        if existing is not None:
            logger.info("upload_deduplicated", extra={"image_id": str(existing.id)})
            return existing, None

        decoded = decode_image(data, modality=exam.modality, content_type=content_type)

        image_id = uuid.uuid4()
        key = build_image_key(exam_id, image_id, content_type)
        stored = self.storage.put(key, data, content_type=content_type)

        image = Image(
            id=image_id,
            exam_id=exam_id,
            storage_bucket=stored.bucket,
            storage_key=stored.key,
            content_type=content_type,
            byte_size=stored.size,
            checksum_sha256=stored.checksum_sha256,
            width=decoded.width,
            height=decoded.height,
            num_frames=decoded.frame_count,
            pixel_spacing_um=self._pixel_spacing(exam),
            status=ImageStatus.UPLOADED,
            original_extension=(Path(filename).suffix[:16] if filename else None),
            uploaded_by_id=actor.id if actor else None,
        )
        self.images.add(image)
        self.audit.record(
            AuditAction.IMAGE_UPLOAD,
            actor=actor,
            resource_type="image",
            resource_id=image.id,
            meta={"exam_id": str(exam_id), "bytes": stored.size, "content_type": content_type},
        )

        quality: dict[str, Any] | None = None
        if run_quality_check:
            quality = self._screen_quality(decoded, exam.modality)
            if quality is not None:
                image.status = (
                    ImageStatus.VALIDATED if quality["is_valid"] else ImageStatus.REJECTED
                )
        self.session.flush()
        return image, quality

    def _pixel_spacing(self, exam: Exam) -> dict[str, Any] | None:
        """Extract the device pixel scale from the exam's acquisition metadata."""
        metadata = exam.acquisition_metadata or {}
        spacing = metadata.get("pixel_spacing_um")
        return spacing if isinstance(spacing, dict) else None

    def _screen_quality(self, decoded: ExamImage, modality: Modality) -> dict[str, Any] | None:
        """Run the modality's quality model as an advisory check at upload time."""
        candidates = self.registry.list(
            modality=modality, task=TaskType.QUALITY, available_only=True
        )
        if not candidates:
            return None
        try:
            result = candidates[0].run(decoded, explain=False)
        except Exception:  # noqa: BLE001 - screening must never block an upload
            logger.warning("upload_quality_check_failed", exc_info=True)
            return None
        if result.quality is None:
            return None
        return {"model_id": result.model_id, **result.quality.model_dump()}

    # -- reading images back ----------------------------------------------- #
    def load_exam_image(
        self, image: Image, exam: Exam, *, frame_selection: str = "middle"
    ) -> ExamImage:
        """Fetch an image's bytes from storage and decode it for inference."""
        data = self.storage.get(image.storage_key)
        spacing = {
            key: float(value)
            for key, value in (image.pixel_spacing_um or {}).items()
            if isinstance(value, (int, float))
        }
        if image.num_frames > 1 and image.content_type in MULTI_FRAME_TYPES:
            frames = decode_series(
                data,
                modality=exam.modality,
                image_id=str(image.id),
                content_type=image.content_type,
                pixel_spacing_um=spacing,
            )
            return select_frame(frames, frame_selection)
        return decode_image(
            data,
            modality=exam.modality,
            image_id=str(image.id),
            content_type=image.content_type,
            pixel_spacing_um=spacing,
        )
