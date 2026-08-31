"""Registry bootstrap: binds catalogue entries to concrete implementations.

Resolution order for every catalogued task:

1. a **built-in** implementation, if one exists (quality control, classical OCT
   boundaries, classical vessels);
2. an **installed manifest** under ``MODEL_DIR`` matching the catalogue id,
   which supersedes a built-in of the same id;
3. otherwise a **placeholder** that reports what must be installed.

Manifests that do not correspond to any catalogue entry are registered too, so
a site can add a model the catalogue never anticipated without touching code.
"""

from __future__ import annotations

from pathlib import Path

from app.ai.base import BaseOphthalmologyModel
from app.ai.manifest import ExternalModelSpec, discover_manifests
from app.ai.models.external import build_external_model
from app.ai.models.placeholder import PlaceholderModel
from app.ai.registry import ModelRegistry, registry
from app.core.enums import Modality, TaskType
from app.core.exceptions import ValidationError
from app.core.logging import get_logger
from app.ophthalmology.catalog import CATALOG, CATALOG_BY_ID
from app.ophthalmology.fundus.vessels_classical import RetinalVesselClassicalModel
from app.ophthalmology.oct.layers_classical import OCTLayerBoundaryModel
from app.ophthalmology.quality.fundus_quality import FundusQualityModel
from app.ophthalmology.quality.oct_quality import OCTQualityModel

logger = get_logger(__name__)


def built_in_models() -> list[BaseOphthalmologyModel]:
    """Models that ship with the platform and need no checkpoint."""
    return [
        FundusQualityModel(),
        OCTQualityModel(),
        OCTLayerBoundaryModel(),
        RetinalVesselClassicalModel(),
    ]


def _merge_catalog_metadata(model: BaseOphthalmologyModel) -> None:
    """Carry catalogue framing (description, limitations) into a manifest model.

    The manifest owns the technical truth about the checkpoint; the catalogue
    owns the clinical framing. Manifest values win where both are present.
    """
    entry = CATALOG_BY_ID.get(model.model_id)
    if entry is None:
        return
    updates = {}
    if not model.metadata.description:
        updates["description"] = entry.description
    if not model.metadata.limitations:
        updates["limitations"] = entry.limitations
    if updates:
        model.metadata = model.metadata.model_copy(update=updates)


def models_from_manifests(model_dir: Path | None = None) -> list[BaseOphthalmologyModel]:
    """Build adapters for every valid manifest found under ``MODEL_DIR``."""
    models: list[BaseOphthalmologyModel] = []
    spec: ExternalModelSpec
    for spec in discover_manifests(model_dir):
        try:
            model = build_external_model(spec)
        except ValidationError as exc:
            logger.warning(
                "model_manifest_unsupported",
                extra={"model_id": spec.model_id, "reason": exc.message},
            )
            continue
        _merge_catalog_metadata(model)
        models.append(model)
    return models


def bootstrap_registry(
    target: ModelRegistry | None = None, *, model_dir: Path | None = None
) -> ModelRegistry:
    """Populate a registry with built-ins, installed manifests and placeholders.

    Safe to call repeatedly: registrations replace previous ones, so a running
    worker can be told to re-scan ``MODEL_DIR`` after new weights are installed.
    """
    # Explicit None check: an empty registry is falsy (it defines __len__).
    target = registry if target is None else target
    registered: set[str] = set()

    for model in built_in_models():
        target.register(model, replace=True)
        registered.add(model.model_id)

    for model in models_from_manifests(model_dir):
        target.register(model, replace=True)
        registered.add(model.model_id)

    for entry in CATALOG:
        if entry.model_id in registered:
            continue
        target.register(
            PlaceholderModel(
                entry.to_metadata(), subdir=entry.subdir, manifest_name=entry.manifest_name
            ),
            replace=True,
        )

    logger.info(
        "model_registry_ready",
        extra={
            "total": len(target),
            "available": len(target.list(available_only=True)),
        },
    )
    return target


def default_models_for(modality: Modality, *, target: ModelRegistry | None = None) -> list[str]:
    """Model ids used when an analysis request does not name any.

    Quality control runs first, then every other available model for the
    modality. Unavailable (placeholder) models are excluded so a default
    pipeline never fails on a missing checkpoint.
    """
    target = registry if target is None else target
    available = target.list(modality=modality, available_only=True)
    quality = [m.model_id for m in available if m.metadata.task is TaskType.QUALITY]
    others = [m.model_id for m in available if m.metadata.task is not TaskType.QUALITY]
    return quality + others


__all__ = [
    "bootstrap_registry",
    "built_in_models",
    "default_models_for",
    "models_from_manifests",
]
