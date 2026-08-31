"""S3-compatible object storage (AWS S3, MinIO, GCS via interoperability)."""

from __future__ import annotations

import hashlib
from typing import Any

from app.core.exceptions import StorageError
from app.storage.base import ObjectStorage, StoredObject


class S3ObjectStorage(ObjectStorage):
    """Thin wrapper over boto3's S3 client.

    ``endpoint_url`` makes the same class work against MinIO in Docker Compose
    and against AWS S3 in production.
    """

    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
        presigned_ttl: int = 900,
        create_bucket: bool = False,
    ) -> None:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise StorageError(
                "boto3 is required for STORAGE_BACKEND=s3.",
                details={"install": "pip install boto3"},
            ) from exc

        self.bucket = bucket
        self.presigned_ttl = presigned_ttl
        self._client: Any = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        if create_bucket:
            self.ensure_bucket()

    def ensure_bucket(self) -> None:
        """Create the bucket when missing (used by local MinIO bootstrap)."""
        try:
            self._client.head_bucket(Bucket=self.bucket)
        except Exception:  # noqa: BLE001 - boto3 raises ClientError subclasses
            try:
                self._client.create_bucket(Bucket=self.bucket)
            except Exception as exc:  # noqa: BLE001
                raise StorageError("Could not create storage bucket.") from exc

    def put(self, key: str, data: bytes, *, content_type: str) -> StoredObject:
        try:
            self._client.put_object(
                Bucket=self.bucket, Key=key, Body=data, ContentType=content_type
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError("Failed to upload object.") from exc
        return StoredObject(
            bucket=self.bucket,
            key=key,
            size=len(data),
            content_type=content_type,
            checksum_sha256=hashlib.sha256(data).hexdigest(),
        )

    def get(self, key: str) -> bytes:
        try:
            response = self._client.get_object(Bucket=self.bucket, Key=key)
            return bytes(response["Body"].read())
        except Exception as exc:  # noqa: BLE001
            raise StorageError("Object not found in storage.") from exc

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
        except Exception:  # noqa: BLE001
            return False
        return True

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:  # noqa: BLE001
            raise StorageError("Failed to delete object.") from exc

    def url_for(self, key: str, *, expires_in: int | None = None) -> str:
        try:
            return str(
                self._client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self.bucket, "Key": key},
                    ExpiresIn=expires_in or self.presigned_ttl,
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise StorageError("Failed to sign object URL.") from exc
