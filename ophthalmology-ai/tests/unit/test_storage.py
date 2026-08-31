"""Local object storage."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from app.core.exceptions import StorageError
from app.storage import LocalObjectStorage, build_image_key, build_report_key, build_result_key


@pytest.fixture
def storage(tmp_path: Path) -> LocalObjectStorage:
    return LocalObjectStorage(tmp_path, "bucket")


def test_put_get_exists_delete(storage: LocalObjectStorage) -> None:
    stored = storage.put("a/b.png", b"payload", content_type="image/png")
    assert stored.size == 7
    assert stored.checksum_sha256
    assert storage.exists("a/b.png")
    assert storage.get("a/b.png") == b"payload"
    storage.delete("a/b.png")
    assert not storage.exists("a/b.png")
    storage.delete("a/b.png")  # deleting twice is not an error


def test_missing_object_raises_storage_error(storage: LocalObjectStorage) -> None:
    with pytest.raises(StorageError):
        storage.get("missing.png")


def test_path_traversal_is_refused(storage: LocalObjectStorage) -> None:
    """A key must never escape its bucket."""
    with pytest.raises(StorageError):
        storage.put("../../escape.png", b"x", content_type="image/png")


def test_keys_contain_no_patient_identifiers() -> None:
    """Storage keys are built from UUIDs only."""
    exam_id, image_id = uuid.uuid4(), uuid.uuid4()
    key = build_image_key(exam_id, image_id, "image/png")
    assert str(exam_id) in key and str(image_id) in key
    assert key.endswith(".png")

    result_key = build_result_key(uuid.uuid4(), uuid.uuid4(), "masks", "retina/../x", "image/png")
    assert ".." not in result_key
    assert result_key.endswith(".png")

    assert build_report_key(exam_id, image_id, "application/pdf").endswith(".pdf")
