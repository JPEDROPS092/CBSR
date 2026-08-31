"""The inference pipeline.

Sequence::

    validation -> quality control -> [gate] -> preprocessing -> inference
               -> postprocessing -> explainability -> results

The engine is deliberately storage- and database-free: it takes decoded images
and returns in-memory results. Persisting masks, artifacts and rows is the
service layer's job, which keeps model execution testable and reusable from a
worker, a script or a notebook.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.ai.preprocessing import ExamImage
from app.ai.registry import ModelRegistry
from app.ai.registry import registry as global_registry
from app.ai.results import ModelResult, QualityReport
from app.core.config import get_settings
from app.core.enums import Modality, RunStatus, TaskType
from app.core.exceptions import AppError, ModelNotFoundError, ModelUnavailableError
from app.core.logging import get_logger, log_context

logger = get_logger(__name__)


class ModelSelection(BaseModel):
    """A requested model, optionally pinned to a version."""

    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    version: str | None = None

    @classmethod
    def parse(cls, value: str | dict[str, Any] | ModelSelection) -> ModelSelection:
        """Accept ``"id"``, ``"id@version"`` or ``{"model_id": ..., "version": ...}``."""
        if isinstance(value, ModelSelection):
            return value
        if isinstance(value, dict):
            return cls.model_validate(value)
        model_id, _, version = value.partition("@")
        return cls(model_id=model_id.strip(), version=version.strip() or None)

    def as_string(self) -> str:
        return f"{self.model_id}@{self.version}" if self.version else self.model_id


class PipelineConfig(BaseModel):
    """Runtime configuration of one analysis."""

    model_config = ConfigDict(protected_namespaces=())

    models: list[ModelSelection] = Field(default_factory=list)
    quality_gate: bool | None = None
    explainability: bool | None = None
    #: Which frames of a multi-frame series to analyse.
    frame_selection: str = "middle"

    @classmethod
    def from_request(
        cls,
        models: list[str] | None,
        *,
        quality_gate: bool | None = None,
        explainability: bool | None = None,
        frame_selection: str = "middle",
    ) -> PipelineConfig:
        """Build a config from the API's simple string list."""
        return cls(
            models=[ModelSelection.parse(item) for item in (models or [])],
            quality_gate=quality_gate,
            explainability=explainability,
            frame_selection=frame_selection,
        )


@dataclass(slots=True)
class PipelineOutcome:
    """Everything one analysis produced."""

    results: list[ModelResult] = field(default_factory=list)
    quality: QualityReport | None = None
    quality_model_id: str | None = None
    gate_passed: bool = True

    @property
    def completed(self) -> list[ModelResult]:
        return [r for r in self.results if r.status is RunStatus.COMPLETED]

    def summary(self) -> dict[str, Any]:
        """Compact summary stored on the analysis row."""
        return {
            "gate_passed": self.gate_passed,
            "quality_model_id": self.quality_model_id,
            "quality": self.quality.model_dump() if self.quality else None,
            "models_run": len(self.results),
            "models_completed": len(self.completed),
        }


class InferenceEngine:
    """Runs a configured set of models over an exam image."""

    def __init__(self, model_registry: ModelRegistry | None = None) -> None:
        # Explicit None check: an empty registry is falsy (it defines __len__).
        self.registry = global_registry if model_registry is None else model_registry

    # -- selection --------------------------------------------------------- #
    def resolve(self, config: PipelineConfig, modality: Modality) -> list[Any]:
        """Resolve requested selections into model instances.

        Quality models are moved to the front so the gate can act before any
        expensive model runs.

        Raises:
            ModelNotFoundError: if a requested model id is not registered.
        """
        if not config.models:
            from app.ai.models import default_models_for

            selections = [ModelSelection(model_id=mid) for mid in default_models_for(modality)]
        else:
            selections = config.models

        models = [self.registry.get(sel.model_id, sel.version) for sel in selections]
        quality = [m for m in models if m.metadata.task is TaskType.QUALITY]
        others = [m for m in models if m.metadata.task is not TaskType.QUALITY]
        return quality + others

    # -- execution --------------------------------------------------------- #
    def run(
        self,
        image: ExamImage,
        config: PipelineConfig,
        *,
        modality: Modality | None = None,
    ) -> PipelineOutcome:
        """Execute the pipeline for a single image.

        A failing model never aborts the analysis: its run is recorded with
        ``status = failed`` and the remaining models still run. A failing
        *quality gate*, in contrast, deliberately stops the downstream models
        and records them as ``skipped_quality``.
        """
        settings = get_settings()
        modality = modality or image.modality
        gate_enabled = (
            settings.QUALITY_GATE_ENABLED if config.quality_gate is None else config.quality_gate
        )
        outcome = PipelineOutcome()
        models = self.resolve(config, modality)

        for model in models:
            with log_context(model_id=model.model_id):
                is_quality = model.metadata.task is TaskType.QUALITY
                if not outcome.gate_passed and not is_quality:
                    outcome.results.append(
                        self._skipped(
                            model,
                            RunStatus.SKIPPED_QUALITY,
                            "Skipped: the image did not pass quality control.",
                        )
                    )
                    continue

                availability = model.availability()
                if not availability.available:
                    outcome.results.append(
                        self._skipped(
                            model,
                            RunStatus.SKIPPED_UNAVAILABLE,
                            availability.remediation or availability.reason or "Model unavailable.",
                        )
                    )
                    continue

                result = self._run_one(model, image, config)
                outcome.results.append(result)

                if is_quality and result.quality is not None:
                    if outcome.quality is None:
                        outcome.quality = result.quality
                        outcome.quality_model_id = model.model_id
                    if gate_enabled and not result.quality.is_valid:
                        outcome.gate_passed = False
                        logger.info(
                            "quality_gate_rejected",
                            extra={
                                "model_id": model.model_id,
                                "quality_score": result.quality.quality_score,
                                "issues": result.quality.issues,
                            },
                        )
        return outcome

    def _run_one(self, model: Any, image: ExamImage, config: PipelineConfig) -> ModelResult:
        """Run one model, converting failures into a failed result."""
        try:
            result = model.run(image, explain=config.explainability)
        except ModelUnavailableError as exc:
            return self._skipped(model, RunStatus.SKIPPED_UNAVAILABLE, exc.message)
        except AppError as exc:
            logger.warning("model_run_failed", extra={"model_id": model.model_id, "code": exc.code})
            return self._failed(model, exc.message)
        except Exception:  # noqa: BLE001 - one broken model must not sink the analysis
            logger.exception("model_run_crashed", extra={"model_id": model.model_id})
            return self._failed(model, "Model raised an unexpected error.")

        logger.info(
            "model_run_completed",
            extra={
                "model_id": model.model_id,
                "model_version": model.version,
                "processing_time_ms": result.processing_time_ms,
                "device": result.device_info.device,
                "predictions": len(result.predictions),
                "segmentations": len(result.segmentations),
            },
        )
        return result

    @staticmethod
    def _skipped(model: Any, status: RunStatus, message: str) -> ModelResult:
        return ModelResult(
            model_id=model.model_id,
            model_version=model.version,
            task=model.metadata.task,
            status=status,
            error_message=message,
        )

    @staticmethod
    def _failed(model: Any, message: str) -> ModelResult:
        return ModelResult(
            model_id=model.model_id,
            model_version=model.version,
            task=model.metadata.task,
            status=RunStatus.FAILED,
            error_message=message,
        )


def select_frame(frames: list[ExamImage], strategy: str = "middle") -> ExamImage:
    """Pick which B-scan of a series to analyse.

    ``middle`` is the default because the central B-scan of a macular volume is
    the one through the fovea in most acquisition patterns.

    Raises:
        ModelNotFoundError: never; raises ``ValueError`` on an empty series.
    """
    if not frames:
        raise ValueError("Frame series is empty.")
    if strategy == "first":
        return frames[0]
    if strategy == "last":
        return frames[-1]
    return frames[len(frames) // 2]


__all__ = [
    "InferenceEngine",
    "ModelNotFoundError",
    "ModelSelection",
    "PipelineConfig",
    "PipelineOutcome",
    "select_frame",
]
