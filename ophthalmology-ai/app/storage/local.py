"""Filesystem-backed object storage for local development and tests."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from app.core.exceptions import StorageError
from app.storage.base import ObjectStorage, StoredObject


class LocalObjectStorage(ObjectStorage):
    """Stores objects under ``root/bucket/key``.

    Not for production use: it offers no replication, encryption at rest or
    presigned URLs. :class:`Settings` refuses ``STORAGE_BACKEND=local`` outside
    local/test environments.
    """

    def __init__(self, root: Path, bucket: str) -> None:
        self.root = Path(root)
        self.bucket = bucket
        self._base = self.root / bucket
        self._base.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        """Resolve a key inside the bucket, rejecting traversal attempts."""
        target = (self._base / key).resolve()
        base = self._base.resolve()
        if not target.is_relative_to(base):
            raise StorageError("Invalid storage key.", details={"reason": "path_traversal"})
        return target

    def put(self, key: str, data: bytes, *, content_type: str) -> StoredObject:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary file first so a crash cannot leave a partial
        # object visible under the final key.
        tmp = path.with_suffix(path.suffix + ".partial")
        tmp.write_bytes(data)
        tmp.replace(path)
        return StoredObject(
            bucket=self.bucket,
            key=key,
            size=len(data),
            content_type=content_type,
            checksum_sha256=hashlib.sha256(data).hexdigest(),
        )

    def get(self, key: str) -> bytes:
        path = self._resolve(key)
        try:
            return path.read_bytes()
        except OSError as exc:
            raise StorageError("Object not found in storage.") from exc

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.is_file():
            path.unlink()

    def url_for(self, key: str, *, expires_in: int | None = None) -> str:
        """Local storage has no presigning; route through the authenticated API."""
        return f"/api/v1/objects/{key}"

    def clear(self) -> None:
        """Remove every object in the bucket (test helper)."""
        shutil.rmtree(self._base, ignore_errors=True)
        self._base.mkdir(parents=True, exist_ok=True)
