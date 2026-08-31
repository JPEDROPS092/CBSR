"""Exam and image-upload routes."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile, status

from app.api.dependencies import CurrentUser, get_exam_service, require_permission
from app.core.config import get_settings
from app.core.exceptions import PayloadTooLargeError
from app.core.security import Permission
from app.schemas.exam import ExamCreate, ExamRead, ImageRead, ImageUploadResponse
from app.services.exam_service import ExamService

router = APIRouter(prefix="/exams", tags=["exams"])

ExamDep = Annotated[ExamService, Depends(get_exam_service)]


@router.post(
    "",
    response_model=ExamRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create an exam",
    dependencies=[Depends(require_permission(Permission.EXAM_WRITE))],
)
def create_exam(payload: ExamCreate, service: ExamDep, user: CurrentUser) -> ExamRead:
    """Create an exam for a patient.

    Include ``acquisition_metadata.pixel_spacing_um`` to obtain measurements in
    micrometres rather than pixels.
    """
    exam = service.create(payload, actor=user)
    service.session.commit()
    return ExamRead.model_validate(exam)


@router.get(
    "/{exam_id}",
    response_model=ExamRead,
    summary="Fetch an exam",
    dependencies=[Depends(require_permission(Permission.EXAM_READ))],
)
def get_exam(exam_id: uuid.UUID, service: ExamDep, user: CurrentUser) -> ExamRead:
    """Fetch one exam by id."""
    return ExamRead.model_validate(service.get(exam_id, actor=user))


@router.get(
    "/{exam_id}/images",
    response_model=list[ImageRead],
    summary="List an exam's images",
    dependencies=[Depends(require_permission(Permission.EXAM_READ))],
)
def list_images(exam_id: uuid.UUID, service: ExamDep) -> list[ImageRead]:
    """List the images uploaded for an exam."""
    return [ImageRead.model_validate(image) for image in service.list_images(exam_id)]


@router.post(
    "/{exam_id}/upload",
    response_model=ImageUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload an exam image",
    dependencies=[Depends(require_permission(Permission.IMAGE_UPLOAD))],
)
async def upload_image(
    exam_id: uuid.UUID,
    service: ExamDep,
    user: CurrentUser,
    file: Annotated[UploadFile, File(description="JPEG, PNG or TIFF image.")],
) -> ImageUploadResponse:
    """Store one image for an exam and screen its quality.

    The file's type is determined from its magic number, not from the supplied
    filename or content type. The original filename is discarded - only the
    extension is kept - because uploaded filenames routinely contain patient
    names.
    """
    settings = get_settings()
    data = await file.read()
    if len(data) > settings.MAX_UPLOAD_BYTES:
        raise PayloadTooLargeError(
            "Uploaded file exceeds the configured size limit.",
            details={"max_bytes": settings.MAX_UPLOAD_BYTES},
        )
    image, quality = service.upload_image(exam_id, data, filename=file.filename, actor=user)
    service.session.commit()
    return ImageUploadResponse(image=ImageRead.model_validate(image), quality=quality)
