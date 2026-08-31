"""Model registry - the plugin system.

Models register themselves at startup; the rest of the platform looks them up
by id. Registration is the *only* step needed to expose a new model through
``/api/v1/models`` and make it selectable in an analysis request.

Versioning: several versions of the same ``model_id`` may be registered at
once. ``get("dr_grading")`` resolves to the newest available version;
``get("dr_grading", "2.1.0")`` pins one explicitly, which is what stored
analyses reference so results stay reproducible after an upgrade.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator

from app.ai.base import BaseOphthalmologyModel
from app.ai.results import Availability, ModelMetadata
from app.core.enums import Modality, ModelStatus, TaskType
from app.core.exceptions import ConflictError, ModelNotFoundError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _version_key(version: str) -> tuple[int, ...]:
    """Sortable key for a dotted version string; unparsable parts sort last."""
    parts: list[int] = []
    for chunk in version.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


class ModelRegistry:
    """Thread-safe registry of model plugins."""

    def __init__(self) -> None:
        self._models: dict[tuple[str, str], BaseOphthalmologyModel] = {}
        self._lock = threading.RLock()

    # -- registration ------------------------------------------------------ #
    def register(self, model: BaseOphthalmologyModel, *, replace: bool = False) -> None:
        """Add a model to the registry.

        Raises:
            ConflictError: if the same ``(model_id, version)`` is already
                registered and ``replace`` is False.
        """
        key = (model.model_id, model.version)
        with self._lock:
            if key in self._models and not replace:
                raise ConflictError(
                    "A model with this id and version is already registered.",
                    details={"model_id": model.model_id, "version": model.version},
                )
            self._models[key] = model
        logger.info(
            "model_registered",
            extra={
                "model_id": model.model_id,
                "model_version": model.version,
                "task": model.task,
                "modality": model.modality,
                "available": model.availability().available,
            },
        )

    def unregister(self, model_id: str, version: str) -> None:
        """Remove a model version from the registry."""
        with self._lock:
            self._models.pop((model_id, version), None)

    def clear(self) -> None:
        """Drop every registration (used by tests)."""
        with self._lock:
            self._models.clear()

    # -- lookup ------------------------------------------------------------ #
    def get(self, model_id: str, version: str | None = None) -> BaseOphthalmologyModel:
        """Resolve a model, defaulting to its newest registered version.

        Raises:
            ModelNotFoundError: when no matching model is registered.
        """
        with self._lock:
            if version is not None:
                model = self._models.get((model_id, version))
                if model is None:
                    raise ModelNotFoundError(
                        "Model version is not registered.",
                        details={"model_id": model_id, "version": version},
                    )
                return model
            candidates = [m for (mid, _), m in self._models.items() if mid == model_id]
        if not candidates:
            raise ModelNotFoundError("Model is not registered.", details={"model_id": model_id})
        return max(candidates, key=lambda m: _version_key(m.version))

    def has(self, model_id: str, version: str | None = None) -> bool:
        """Whether a model (optionally a specific version) is registered."""
        try:
            self.get(model_id, version)
        except ModelNotFoundError:
            return False
        return True

    def list(
        self,
        *,
        modality: Modality | None = None,
        task: TaskType | None = None,
        status: ModelStatus | None = None,
        available_only: bool = False,
    ) -> list[BaseOphthalmologyModel]:
        """List registered models, optionally filtered."""
        with self._lock:
            models = list(self._models.values())
        if modality is not None:
            models = [m for m in models if m.metadata.modality is modality]
        if task is not None:
            models = [m for m in models if m.metadata.task is task]
        if status is not None:
            models = [m for m in models if m.metadata.status is status]
        if available_only:
            models = [m for m in models if m.availability().available]
        return sorted(models, key=lambda m: (m.model_id, _version_key(m.version)))

    def describe(self, model: BaseOphthalmologyModel) -> tuple[ModelMetadata, Availability]:
        """Metadata plus a live availability probe for one model."""
        return model.metadata, model.availability()

    def __iter__(self) -> Iterator[BaseOphthalmologyModel]:
        return iter(self.list())

    def __len__(self) -> int:
        with self._lock:
            return len(self._models)


#: Process-wide registry. Populated by :func:`app.ai.models.bootstrap_registry`.
registry = ModelRegistry()


def register_all(models: Iterable[BaseOphthalmologyModel], *, replace: bool = True) -> None:
    """Register a collection of models into the global registry."""
    for model in models:
        registry.register(model, replace=replace)
