"""Model-registry routes.

These expose the catalogue: what the platform can run, what it cannot run yet,
and - for the latter - exactly what an operator must install.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from app.ai.models import bootstrap_registry
from app.ai.registry import ModelRegistry
from app.api.dependencies import DbSession, get_registry_dep, require_permission, require_role
from app.core.enums import Modality, ModelStatus, TaskType, UserRole
from app.core.security import Permission
from app.database.models import ModelRecord
from app.database.repositories import ModelRecordRepository
from app.schemas.model import ModelDetail, ModelSummary

router = APIRouter(prefix="/models", tags=["models"])

RegistryDep = Annotated[ModelRegistry, Depends(get_registry_dep)]


def _summary(model: object) -> ModelSummary:
    metadata = model.metadata  # type: ignore[attr-defined]
    return ModelSummary(
        model_id=metadata.model_id,
        name=metadata.name,
        version=metadata.version,
        modality=metadata.modality,
        task=metadata.task,
        framework=metadata.framework,
        evidence_level=metadata.evidence_level,
        status=metadata.status,
        available=model.availability().available,  # type: ignore[attr-defined]
        description=metadata.description,
    )


@router.get(
    "",
    response_model=list[ModelSummary],
    summary="List registered models",
    dependencies=[Depends(require_permission(Permission.MODEL_READ))],
)
def list_models(
    model_registry: RegistryDep,
    modality: Modality | None = None,
    task: TaskType | None = None,
    status: ModelStatus | None = None,
    available_only: Annotated[bool, Query(description="Only models runnable here.")] = False,
) -> list[ModelSummary]:
    """List every model in the registry, with live availability."""
    return [
        _summary(model)
        for model in model_registry.list(
            modality=modality, task=task, status=status, available_only=available_only
        )
    ]


@router.get(
    "/{model_id}",
    response_model=ModelDetail,
    summary="Fetch a model's full metadata",
    dependencies=[Depends(require_permission(Permission.MODEL_READ))],
)
def get_model(
    model_id: str,
    model_registry: RegistryDep,
    version: str | None = Query(default=None, description="Pin a specific version."),
) -> ModelDetail:
    """Return a model's metadata, license and - when unavailable - remediation."""
    model = model_registry.get(model_id, version)
    metadata = model.metadata
    return ModelDetail(
        **_summary(model).model_dump(),
        availability=model.availability(),
        input_spec=metadata.input_spec.model_dump(),
        output_spec=metadata.output_spec.model_dump(),
        license=metadata.license,
        supports_explainability=metadata.supports_explainability,
        reported_metrics=metadata.reported_metrics,
        limitations=metadata.limitations,
    )


@router.post(
    "/refresh",
    response_model=list[ModelSummary],
    summary="Re-scan MODEL_DIR and sync the registry (admin only)",
    dependencies=[Depends(require_role(UserRole.ADMIN))],
)
def refresh_models(session: DbSession, model_registry: RegistryDep) -> list[ModelSummary]:
    """Re-read model manifests from disk and persist the catalogue.

    Call this after installing new weights so a running API picks them up
    without a restart.
    """
    bootstrap_registry(model_registry)
    sync_registry_to_database(session, model_registry)
    session.commit()
    return [_summary(model) for model in model_registry.list()]


def sync_registry_to_database(session: DbSession, model_registry: ModelRegistry) -> None:
    """Persist registry metadata so model runs can reference a durable record.

    The in-memory registry is the source of truth for execution; this table is
    the versioned record used for audit and reproducibility.
    """
    repository = ModelRecordRepository(session)
    for model in model_registry.list():
        metadata = model.metadata
        record = repository.get_by_model_id(metadata.model_id, metadata.version)
        values = {
            "name": metadata.name,
            "description": metadata.description,
            "modality": metadata.modality,
            "task": metadata.task,
            "framework": metadata.framework,
            "evidence_level": metadata.evidence_level,
            "status": metadata.status,
            "weights_sha256": metadata.weights.sha256 if metadata.weights else None,
            "license_name": metadata.license.name,
            "license_url": metadata.license.url,
            "source_url": metadata.license.source_url,
            "commercial_use": metadata.license.commercial_use,
            "input_spec": metadata.input_spec.model_dump(),
            "output_spec": metadata.output_spec.model_dump(),
            "reported_metrics": metadata.reported_metrics or None,
        }
        if record is None:
            repository.add(
                ModelRecord(model_id=metadata.model_id, version=metadata.version, **values)
            )
        else:
            for field, value in values.items():
                setattr(record, field, value)
