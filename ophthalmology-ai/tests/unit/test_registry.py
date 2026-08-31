"""Model registry: registration, versioning and lookup."""

from __future__ import annotations

import pytest

from app.ai.base import ClassicalModel
from app.ai.preprocessing import ExamImage
from app.ai.registry import ModelRegistry
from app.ai.results import ModelMetadata, ModelResult
from app.core.enums import EvidenceLevel, Framework, Modality, TaskType
from app.core.exceptions import ConflictError, ModelNotFoundError


def _metadata(model_id: str, version: str, **overrides: object) -> ModelMetadata:
    base = {
        "model_id": model_id,
        "name": model_id,
        "version": version,
        "modality": Modality.OCT,
        "task": TaskType.CLASSIFICATION,
        "framework": Framework.CLASSICAL,
        "evidence_level": EvidenceLevel.HEURISTIC,
    }
    base.update(overrides)
    return ModelMetadata(**base)  # type: ignore[arg-type]


class DummyModel(ClassicalModel):
    """A trivial model used to exercise registry behaviour."""

    def __init__(
        self, model_id: str = "dummy", version: str = "1.0.0", **overrides: object
    ) -> None:
        self.metadata = _metadata(model_id, version, **overrides)
        super().__init__()

    def predict(self, prepared: ExamImage) -> float:
        return 1.0

    def postprocess(self, output: float, image: ExamImage) -> ModelResult:
        return ModelResult(
            model_id=self.model_id, model_version=self.version, task=self.metadata.task
        )


@pytest.fixture
def empty_registry() -> ModelRegistry:
    return ModelRegistry()


def test_register_and_get(empty_registry: ModelRegistry) -> None:
    model = DummyModel()
    empty_registry.register(model)
    assert empty_registry.get("dummy") is model
    assert len(empty_registry) == 1


def test_duplicate_registration_conflicts(empty_registry: ModelRegistry) -> None:
    empty_registry.register(DummyModel())
    with pytest.raises(ConflictError):
        empty_registry.register(DummyModel())
    empty_registry.register(DummyModel(), replace=True)  # explicit replace is allowed


def test_get_resolves_the_newest_version(empty_registry: ModelRegistry) -> None:
    """An unpinned lookup must return the newest version, not the last registered."""
    empty_registry.register(DummyModel(version="1.9.0"))
    empty_registry.register(DummyModel(version="1.10.0"))
    empty_registry.register(DummyModel(version="1.2.0"))
    assert empty_registry.get("dummy").version == "1.10.0"
    assert empty_registry.get("dummy", "1.2.0").version == "1.2.0"


def test_unknown_model_and_version_raise(empty_registry: ModelRegistry) -> None:
    with pytest.raises(ModelNotFoundError):
        empty_registry.get("nope")
    empty_registry.register(DummyModel())
    with pytest.raises(ModelNotFoundError):
        empty_registry.get("dummy", "9.9.9")


def test_filtering(empty_registry: ModelRegistry) -> None:
    empty_registry.register(DummyModel(model_id="a", modality=Modality.OCT))
    empty_registry.register(
        DummyModel(model_id="b", modality=Modality.FUNDUS, task=TaskType.SEGMENTATION)
    )
    assert [m.model_id for m in empty_registry.list(modality=Modality.FUNDUS)] == ["b"]
    assert [m.model_id for m in empty_registry.list(task=TaskType.CLASSIFICATION)] == ["a"]
    assert len(empty_registry.list()) == 2


def test_bootstrapped_registry_exposes_the_catalog() -> None:
    """Catalogued models are always listed, available or not."""
    from app.ai.registry import registry

    ids = {model.model_id for model in registry.list()}
    assert {"fundus_quality_v1", "oct_quality_v1", "oct_layers_classical_v1"} <= ids
    assert "oct_retinal_layers_v1" in ids  # catalogued, unavailable without weights
    unavailable = registry.get("oct_retinal_layers_v1").availability()
    assert unavailable.available is False
    assert unavailable.missing and unavailable.remediation
