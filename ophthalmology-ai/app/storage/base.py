"""Object-storage abstraction.

Medical images are never stored as BLOBs in PostgreSQL. The database keeps
metadata and a ``(bucket, key)`` pointer; bytes live in S3, GCS, MinIO or - for
local development and tests - a directory on disk.

Storage keys are derived only from UUIDs and content hashes, never from
patient identifiers or uploaded filenames.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class StoredObject:
    """Result of a successful upload."""

    bucket: str
    key: str
    size: int
    content_type: str
    checksum_sha256: str


EXTENSION_BY_CONTENT_TYPE = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/tiff": ".tif",
    "application/dicom": ".dcm",
    "application/json": ".json",
    "text/html": ".html",
    "application/pdf": ".pdf",
    "application/octet-stream": ".bin",
}


def build_image_key(exam_id: uuid.UUID, image_id: uuid.UUID, content_type: str) -> str:
    """Build a PHI-free storage key for an uploaded exam image."""
    date = datetime.now(UTC).strftime("%Y/%m/%d")
    ext = EXTENSION_BY_CONTENT_TYPE.get(content_type, ".bin")
    return f"exams/{date}/{exam_id}/images/{image_id}{ext}"


def build_result_key(
    analysis_id: uuid.UUID, run_id: uuid.UUID, kind: str, name: str, content_type: str
) -> str:
    """Build a storage key for a derived artifact (mask, Grad-CAM, overlay)."""
    ext = EXTENSION_BY_CONTENT_TYPE.get(content_type, ".bin")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)[:60]
    return f"results/{analysis_id}/{run_id}/{kind}/{safe_name}{ext}"


def build_report_key(analysis_id: uuid.UUID, report_id: uuid.UUID, content_type: str) -> str:
    """Build a storage key for a rendered report."""
    ext = EXTENSION_BY_CONTENT_TYPE.get(content_type, ".bin")
    return f"reports/{analysis_id}/{report_id}{ext}"


class ObjectStorage(ABC):
    """Minimal object-storage interface used across the platform."""

    bucket: str

    @abstractmethod
    def put(self, key: str, data: bytes, *, content_type: str) -> StoredObject:
        """Store ``data`` under ``key``, overwriting any existing object."""

    @abstractmethod
    def get(self, key: str) -> bytes:
        """Return the object's bytes.

        Raises:
            StorageError: if the object does not exist or cannot be read.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether an object exists under ``key``."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete an object. Deleting a missing key is not an error."""

    @abstractmethod
    def url_for(self, key: str, *, expires_in: int | None = None) -> str:
        """Return a URL a client can use to fetch the object.

        Backends that cannot issue presigned URLs return the platform's own
        streaming endpoint path, which enforces authentication and RBAC.
        """
