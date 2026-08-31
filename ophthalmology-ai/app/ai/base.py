"""The model plugin interface.

Every model in the platform - a deep network, an ONNX graph or a deterministic
image-processing baseline - implements :class:`BaseOphthalmologyModel`. The API
layer only ever sees this interface and the standard
:class:`~app.ai.results.ModelResult`, which is what lets a new model be added
without touching routes, schemas, the database or the frontend.

Subclasses implement ``load``/``preprocess``/``predict``/``postprocess`` (and
optionally ``explain``); :meth:`BaseOphthalmologyModel.run` is the template
method that sequences them, times the inference and records the execution
context needed for audit and reproducibility.
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any

from app.ai.devices import DeviceManager, get_device_manager
from app.ai.preprocessing import ExamImage
from app.ai.results import (
    ArtifactPayload,
    Availability,
    DeviceInfo,
    ModelMetadata,
    ModelResult,
)
from app.core.config import get_settings
from app.core.enums import ModelStatus, RunStatus
from app.core.exceptions import InferenceError, ModelUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class BaseOphthalmologyModel(ABC):
    """Base class for all ophthalmology models."""

    #: Subclasses must provide their metadata.
    metadata: ModelMetadata

    def __init__(self, device_manager: DeviceManager | None = None) -> None:
        self._device_manager = device_manager or get_device_manager()
        self._loaded = False
        self._load_lock = threading.Lock()

    # -- identity ---------------------------------------------------------- #
    @property
    def model_id(self) -> str:
        return self.metadata.model_id

    @property
    def version(self) -> str:
        return self.metadata.version

    @property
    def modality(self) -> str:
        return str(self.metadata.modality)

    @property
    def task(self) -> str:
        return str(self.metadata.task)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.model_id}@{self.version}>"

    # -- lifecycle --------------------------------------------------------- #
    def availability(self) -> Availability:
        """Whether this model can run here.

        The default implementation reports "available"; adapters that need
        weights or an optional runtime override it so the API can explain what
        is missing instead of failing at inference time.
        """
        if self.metadata.status in (ModelStatus.DISABLED, ModelStatus.DEPRECATED):
            return Availability(
                available=False,
                reason=f"Model is {self.metadata.status}.",
                remediation="Enable the model in the registry or pick another version.",
            )
        return Availability(available=True)

    def load(self) -> None:
        """Load weights into memory. Idempotent and thread-safe.

        Subclasses override :meth:`_load` rather than this method.
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            availability = self.availability()
            if not availability.available:
                raise ModelUnavailableError(
                    availability.reason or "Model is unavailable.",
                    details=availability.model_dump(exclude_none=True),
                )
            started = time.perf_counter()
            self._load()
            self._loaded = True
            logger.info(
                "model_loaded",
                extra={
                    "model_id": self.model_id,
                    "model_version": self.version,
                    "load_time_ms": round((time.perf_counter() - started) * 1000, 2),
                    "device": self._device_manager.profile.torch_device,
                },
            )

    def _load(self) -> None:  # noqa: B027 - optional hook, not an abstract method
        """Hook for subclasses that need to materialize weights.

        Deliberately concrete and empty: models with no checkpoint (the
        classical ones) have nothing to load.
        """

    def unload(self) -> None:
        """Release resources held by the model (used by GPU workers)."""
        self._loaded = False
        self._device_manager.empty_cache()

    # -- inference steps --------------------------------------------------- #
    def preprocess(self, image: ExamImage) -> Any:
        """Convert an exam image into the model's input tensor/array."""
        from app.ai.preprocessing import prepare_for_model

        return prepare_for_model(image, self.metadata.input_spec)

    @abstractmethod
    def predict(self, prepared: Any) -> Any:
        """Run the forward pass on preprocessed input."""

    @abstractmethod
    def postprocess(self, output: Any, image: ExamImage) -> ModelResult:
        """Convert a raw forward-pass output into a standard result."""

    def explain(self, image: ExamImage, output: Any) -> list[ArtifactPayload]:
        """Produce explainability artifacts. Empty when unsupported."""
        return []

    # -- orchestration ----------------------------------------------------- #
    def run(self, image: ExamImage, *, explain: bool | None = None) -> ModelResult:
        """Execute the full inference pipeline for one image.

        Args:
            image: Decoded exam image.
            explain: Force explainability on/off. ``None`` follows
                ``EXPLAINABILITY_ENABLED`` and the model's own capability.

        Returns:
            A populated :class:`~app.ai.results.ModelResult`.

        Raises:
            ModelUnavailableError: weights or runtime missing.
            InferenceError: the model failed while running.
        """
        settings = get_settings()
        self.load()
        started = time.perf_counter()
        try:
            prepared = self.preprocess(image)
            raw_output = self.predict(prepared)
            result = self.postprocess(raw_output, image)
        except ModelUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed error
            logger.exception("model_inference_failed", extra={"model_id": self.model_id})
            raise InferenceError(
                "Model failed while processing the image.",
                details={"model_id": self.model_id, "model_version": self.version},
            ) from exc

        want_explain = settings.EXPLAINABILITY_ENABLED if explain is None else explain
        if want_explain and self.metadata.supports_explainability:
            try:
                result.artifacts.extend(self.explain(image, raw_output))
            except Exception:  # noqa: BLE001 - explainability is best-effort
                logger.warning("explainability_failed", extra={"model_id": self.model_id})
                result.warnings.append("Explainability artifacts could not be generated.")

        elapsed_ms = (time.perf_counter() - started) * 1000
        result.processing_time_ms = round(elapsed_ms, 2)
        result.input_hash = image.input_fingerprint()
        result.device_info = self._device_info()
        result.model_id = self.model_id
        result.model_version = self.version
        if result.status is RunStatus.COMPLETED and self.metadata.limitations:
            result.warnings.append(self.metadata.limitations)
        return result

    def _device_info(self) -> DeviceInfo:
        """Snapshot of the execution context for this run."""
        profile = self._device_manager.profile
        settings = get_settings()
        return DeviceInfo(
            device=str(profile.device),
            device_name=profile.name,
            precision=self._device_manager.resolve_precision(),
            batch_size=settings.INFERENCE_BATCH_SIZE,
            vram_total_mb=profile.total_memory_mb,
            vram_used_mb=self._device_manager.memory_used_mb(),
        )


class ClassicalModel(BaseOphthalmologyModel):
    """Base for deterministic image-processing models.

    These need no checkpoint, so they run everywhere - which is what makes the
    end-to-end pipeline demonstrable without downloading weights. Their outputs
    are engineering signals (``evidence_level = heuristic``), never clinical
    findings.
    """

    def preprocess(self, image: ExamImage) -> ExamImage:
        """Classical models work on the decoded image directly."""
        return image
