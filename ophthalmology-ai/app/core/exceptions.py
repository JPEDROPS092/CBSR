"""Application exception hierarchy.

Every error carries a machine-readable ``code`` and an HTTP status. Messages
are written for API consumers and must never embed patient data, file paths or
stack details (see ``docs/SECURITY.md``).
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class for all expected application errors."""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "Internal server error."

    def __init__(
        self,
        message: str | None = None,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or self.message
        self.code = code or self.code
        self.details = details or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the platform's standard error envelope."""
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class NotFoundError(AppError):
    status_code = 404
    code = "not_found"
    message = "Resource not found."


class ValidationError(AppError):
    status_code = 422
    code = "validation_error"
    message = "Request payload is invalid."


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
    message = "Resource conflict."


class AuthenticationError(AppError):
    status_code = 401
    code = "authentication_failed"
    message = "Authentication credentials are missing or invalid."


class PermissionDeniedError(AppError):
    status_code = 403
    code = "permission_denied"
    message = "You do not have permission to perform this action."


class RateLimitError(AppError):
    status_code = 429
    code = "rate_limited"
    message = "Too many requests."


class UnsupportedMediaError(AppError):
    status_code = 415
    code = "unsupported_media_type"
    message = "Unsupported file type."


class PayloadTooLargeError(AppError):
    status_code = 413
    code = "payload_too_large"
    message = "Uploaded file exceeds the configured size limit."


class StorageError(AppError):
    status_code = 502
    code = "storage_error"
    message = "Object storage operation failed."


class ModelNotFoundError(NotFoundError):
    code = "model_not_found"
    message = "Requested model is not registered."


class ModelUnavailableError(AppError):
    """Raised when a registered model cannot run in this deployment.

    The usual cause is missing weights or a missing runtime (torch, ONNX).
    ``details`` states exactly what the operator must provide.
    """

    status_code = 409
    code = "model_unavailable"
    message = "Model is registered but not available in this deployment."


class InferenceError(AppError):
    status_code = 500
    code = "inference_failed"
    message = "Model inference failed."


class QualityGateRejection(AppError):
    """Raised when quality control rejects an image before inference."""

    status_code = 422
    code = "quality_gate_rejected"
    message = "Image quality is insufficient for the requested analysis."
