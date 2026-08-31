"""Authenticated access to stored objects.

Masks and explainability artifacts are medical-derived data, so they are never
served from a public bucket. With S3-compatible storage the API hands out
short-lived presigned URLs; with local storage it streams the bytes through
this endpoint, which enforces the same authentication and RBAC as every other
route.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Response

from app.api.dependencies import CurrentUser, get_storage_dep, require_permission
from app.core.exceptions import NotFoundError
from app.core.security import Permission
from app.storage import ObjectStorage

router = APIRouter(prefix="/objects", tags=["objects"])

#: Content types this endpoint is willing to return, keyed by extension.
CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tif": "image/tiff",
    ".html": "text/html",
    ".pdf": "application/pdf",
    ".json": "application/json",
}


@router.get(
    "/{key:path}",
    summary="Stream a stored object",
    dependencies=[Depends(require_permission(Permission.ANALYSIS_READ))],
    response_class=Response,
)
def get_object(
    key: str,
    storage: Annotated[ObjectStorage, Depends(get_storage_dep)],
    user: CurrentUser,
) -> Response:
    """Stream one stored object (mask, overlay, rendered report)."""
    if not storage.exists(key):
        raise NotFoundError("Object not found.")
    suffix = key[key.rfind(".") :].lower() if "." in key else ""
    return Response(
        content=storage.get(key),
        media_type=CONTENT_TYPES.get(suffix, "application/octet-stream"),
        headers={"Cache-Control": "private, max-age=60"},
    )
