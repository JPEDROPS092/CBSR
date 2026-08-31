"""Application configuration.

All settings come from the environment (or a local ``.env`` file). Secrets are
never hard-coded: :class:`Settings` refuses to start in production mode with a
default JWT secret.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "staging", "production"]
StorageBackend = Literal["local", "s3"]
QueueBackend = Literal["inline", "celery"]

DEV_JWT_SECRET = "insecure-development-secret-change-me"  # noqa: S105 - explicit placeholder


class Settings(BaseSettings):
    """Runtime configuration for API, workers and inference engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # ``model_`` is a protected namespace in pydantic v2; we do use
        # ``MODEL_DIR``/``MODEL_REGISTRY_*`` names on purpose.
        protected_namespaces=(),
    )

    # --- General -----------------------------------------------------------
    ENVIRONMENT: Environment = "local"
    PROJECT_NAME: str = "Ophthalmology AI Platform"
    API_V1_PREFIX: str = "/api/v1"
    SOFTWARE_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # --- Database ----------------------------------------------------------
    DATABASE_URL: str = "sqlite+pysqlite:///./ophthalmology.db"
    DATABASE_ECHO: bool = False
    DATABASE_POOL_SIZE: int = 5
    DATABASE_MAX_OVERFLOW: int = 10

    # --- Cache / broker ----------------------------------------------------
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str | None = None
    CELERY_RESULT_BACKEND: str | None = None
    TASK_QUEUE_BACKEND: QueueBackend = "inline"

    # --- Object storage ----------------------------------------------------
    STORAGE_BACKEND: StorageBackend = "local"
    STORAGE_LOCAL_ROOT: Path = Path("./storage-data")
    OBJECT_STORAGE_ENDPOINT: str | None = None
    OBJECT_STORAGE_BUCKET: str = "ophthalmology"
    OBJECT_STORAGE_ACCESS_KEY: str | None = None
    OBJECT_STORAGE_SECRET_KEY: str | None = None
    OBJECT_STORAGE_REGION: str = "us-east-1"
    OBJECT_STORAGE_SECURE: bool = True
    PRESIGNED_URL_TTL_SECONDS: int = 900

    # --- Security ----------------------------------------------------------
    JWT_SECRET: str = DEV_JWT_SECRET
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_TTL_MINUTES: int = 30
    REFRESH_TOKEN_TTL_DAYS: int = 14
    PASSWORD_MIN_LENGTH: int = 12
    PASSWORD_HASH_ITERATIONS: int = 600_000
    CORS_ORIGINS: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_REQUESTS: int = 120
    RATE_LIMIT_WINDOW_SECONDS: int = 60

    # --- Uploads -----------------------------------------------------------
    MAX_UPLOAD_BYTES: int = 100 * 1024 * 1024
    ALLOWED_IMAGE_MIME_TYPES: list[str] = Field(
        default_factory=lambda: [
            "image/jpeg",
            "image/png",
            "image/tiff",
            "application/dicom",
        ]
    )

    # --- AI runtime --------------------------------------------------------
    MODEL_DIR: Path = Path("./models")
    DEVICE: Literal["auto", "cpu", "cuda"] = "auto"
    INFERENCE_PRECISION: Literal["fp32", "fp16", "bf16", "int8"] = "fp32"
    INFERENCE_BATCH_SIZE: int = 1
    QUALITY_GATE_ENABLED: bool = True
    QUALITY_GATE_MIN_SCORE: float = 0.35
    EXPLAINABILITY_ENABLED: bool = True

    # --- Observability -----------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: Literal["json", "console"] = "json"

    @field_validator("CORS_ORIGINS", "ALLOWED_IMAGE_MIME_TYPES", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Allow ``A,B`` as well as JSON lists in the environment."""
        if isinstance(value, str) and not value.strip().startswith("["):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @model_validator(mode="after")
    def _check_production_secrets(self) -> Settings:
        if self.ENVIRONMENT in ("staging", "production"):
            if self.JWT_SECRET == DEV_JWT_SECRET or len(self.JWT_SECRET) < 32:
                raise ValueError(
                    "JWT_SECRET must be set to a strong value (>=32 chars) outside local/test."
                )
            if self.STORAGE_BACKEND == "local":
                raise ValueError("STORAGE_BACKEND=local is not supported outside local/test.")
        return self

    @property
    def celery_broker(self) -> str:
        return self.CELERY_BROKER_URL or self.REDIS_URL

    @property
    def celery_backend(self) -> str:
        return self.CELERY_RESULT_BACKEND or self.REDIS_URL

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")


@functools.lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()


settings = get_settings()
