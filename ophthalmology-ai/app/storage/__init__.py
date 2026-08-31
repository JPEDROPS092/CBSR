"""Object storage package."""

from app.storage.base import (
    ObjectStorage,
    StoredObject,
    build_image_key,
    build_report_key,
    build_result_key,
)
from app.storage.factory import build_storage, get_storage
from app.storage.local import LocalObjectStorage
from app.storage.s3 import S3ObjectStorage

__all__ = [
    "LocalObjectStorage",
    "ObjectStorage",
    "S3ObjectStorage",
    "StoredObject",
    "build_image_key",
    "build_report_key",
    "build_result_key",
    "build_storage",
    "get_storage",
]
