"""Storage backend selection."""

from __future__ import annotations

import functools

from app.core.config import Settings, get_settings
from app.storage.base import ObjectStorage
from app.storage.local import LocalObjectStorage
from app.storage.s3 import S3ObjectStorage


def build_storage(settings: Settings | None = None) -> ObjectStorage:
    """Instantiate the configured object-storage backend."""
    settings = settings or get_settings()
    if settings.STORAGE_BACKEND == "local":
        return LocalObjectStorage(settings.STORAGE_LOCAL_ROOT, settings.OBJECT_STORAGE_BUCKET)
    return S3ObjectStorage(
        bucket=settings.OBJECT_STORAGE_BUCKET,
        endpoint_url=settings.OBJECT_STORAGE_ENDPOINT,
        access_key=settings.OBJECT_STORAGE_ACCESS_KEY,
        secret_key=settings.OBJECT_STORAGE_SECRET_KEY,
        region=settings.OBJECT_STORAGE_REGION,
        presigned_ttl=settings.PRESIGNED_URL_TTL_SECONDS,
        create_bucket=settings.ENVIRONMENT in ("local", "test"),
    )


@functools.lru_cache
def get_storage() -> ObjectStorage:
    """Process-wide storage singleton (FastAPI dependency and worker entrypoint)."""
    return build_storage()
