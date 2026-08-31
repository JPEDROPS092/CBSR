"""The plugin system: installing a model without touching any code.

An operator drops a checkpoint and a manifest under ``MODEL_DIR``; the registry
picks it up, it replaces the catalogue placeholder, and its declared
preprocessing - never a guessed one - configures the adapter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ai.manifest import ExternalModelSpec, discover_manifests, load_manifest
from app.ai.models import bootstrap_registry, models_from_manifests
from app.ai.models.external import (
    OnnxClassificationModel,
    OnnxSegmentationModel,
    TorchClassificationModel,
    build_external_model,
)
from app.ai.registry import ModelRegistry
from app.core.enums import Modality, TaskType
from app.core.exceptions import ValidationError

MANIFEST = {
    "model_id": "fundus_dr_grading_v1",
    "name": "Diabetic Retinopathy Grading (site model)",
    "version": "2.1.0",
    "modality": "fundus",
    "task": "classification",
    "framework": "onnx",
    "weights_file": "dr_grading.onnx",
    "input": {
        "image_size": [512, 512],
        "channels": 3,
        "color_space": "rgb",
        "normalization_mean": [0.485, 0.456, 0.406],
        "normalization_std": [0.229, 0.224, 0.225],
    },
    "output": {
        "labels": ["no_dr", "mild_npdr", "moderate_npdr", "severe_npdr", "proliferative_dr"],
        "activation": "softmax",
    },
    "license": {"name": "CC BY-NC 4.0", "commercial_use": "prohibited"},
    "reported_metrics": {"quadratic_weighted_kappa": 0.82},
    "limitations": "45-degree macula-centred images only.",
}


@pytest.fixture
def installed_manifest(model_dir: Path) -> Path:
    """Write a manifest into MODEL_DIR and remove it afterwards."""
    path = model_dir / "fundus" / "fundus_dr_grading_v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(MANIFEST), encoding="utf-8")
    yield path
    path.unlink()


def test_manifest_configures_the_adapter(installed_manifest: Path, model_dir: Path) -> None:
    spec = load_manifest(installed_manifest, model_dir=model_dir)
    assert spec.subdir == "fundus"
    model = build_external_model(spec)
    assert isinstance(model, OnnxClassificationModel)
    assert model.metadata.input_spec.image_size == (512, 512)
    assert model.metadata.input_spec.normalization_mean == (0.485, 0.456, 0.406)
    assert model.metadata.output_spec.labels[0] == "no_dr"
    assert model.metadata.license.commercial_use == "prohibited"
    assert model.metadata.reported_metrics == {"quadratic_weighted_kappa": 0.82}
    assert model.metadata.weights is not None
    assert model.metadata.weights.subdir == "fundus"


def test_installed_model_replaces_the_catalog_placeholder(
    installed_manifest: Path, model_dir: Path
) -> None:
    """Registration is the only step needed to expose a new model."""
    registry = ModelRegistry()
    bootstrap_registry(registry, model_dir=model_dir)
    model = registry.get("fundus_dr_grading_v1")
    assert model.version == "2.1.0"  # the installed manifest, not the placeholder
    assert isinstance(model, OnnxClassificationModel)
    # Clinical framing from the catalogue is preserved where the manifest is silent.
    assert model.metadata.description


def test_model_without_its_runtime_or_weights_reports_what_is_missing(
    installed_manifest: Path, model_dir: Path
) -> None:
    """Availability must be actionable: the operator learns exactly what to add."""
    model = build_external_model(load_manifest(installed_manifest, model_dir=model_dir))
    availability = model.availability()
    assert availability.available is False
    assert availability.missing
    assert availability.remediation


def test_invalid_manifest_is_skipped_not_fatal(model_dir: Path) -> None:
    """One malformed file must not stop the platform from starting."""
    bad = model_dir / "broken.json"
    bad.write_text("{ not json", encoding="utf-8")
    try:
        assert discover_manifests(model_dir) == []
        assert models_from_manifests(model_dir) == []
    finally:
        bad.unlink()


def test_manifest_missing_required_fields_is_rejected(model_dir: Path) -> None:
    path = model_dir / "incomplete.json"
    path.write_text(json.dumps({"model_id": "x"}), encoding="utf-8")
    try:
        with pytest.raises(ValidationError):
            load_manifest(path, model_dir=model_dir)
    finally:
        path.unlink()


def test_unsupported_framework_task_pair_is_reported() -> None:
    """A detection model needs a purpose-written adapter, and says so."""
    spec = ExternalModelSpec(
        model_id="x",
        name="X",
        modality=Modality.FUNDUS,
        task=TaskType.DETECTION,
        framework="pytorch",
        weights_file="x.pt",
    )
    with pytest.raises(ValidationError):
        build_external_model(spec)


def test_segmentation_manifest_builds_a_segmentation_adapter() -> None:
    spec = ExternalModelSpec(
        model_id="oct_fluid_segmentation_v1",
        name="Fluid",
        modality=Modality.OCT,
        task=TaskType.SEGMENTATION,
        framework="onnx",
        weights_file="fluid.onnx",
        output={"segmentation_classes": ["background", "irf", "srf"], "activation": "softmax"},
    )
    assert isinstance(build_external_model(spec), OnnxSegmentationModel)


def test_torch_adapter_refuses_to_guess_an_architecture() -> None:
    """A bare state_dict without a declared architecture is not run."""
    from app.core.exceptions import ModelUnavailableError

    spec = ExternalModelSpec(
        model_id="y",
        name="Y",
        modality=Modality.OCT,
        task=TaskType.CLASSIFICATION,
        framework="pytorch",
        weights_file="y.pt",
        output={"labels": ["a", "b"]},
    )
    model = build_external_model(spec)
    assert isinstance(model, TorchClassificationModel)
    with pytest.raises(ModelUnavailableError):
        model.build_module()
