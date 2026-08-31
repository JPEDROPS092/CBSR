"""Pipeline behaviour: ordering, the quality gate and failure isolation.

Uses tiny stub models rather than real checkpoints - the engine's contract is
what is under test, not any model's accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ai.base import ClassicalModel
from app.ai.inference import InferenceEngine, ModelSelection, PipelineConfig, select_frame
from app.ai.preprocessing import ExamImage
from app.ai.registry import ModelRegistry
from app.ai.results import (
    Availability,
    ModelMetadata,
    ModelResult,
    PredictionItem,
    QualityReport,
)
from app.core.enums import (
    EvidenceLevel,
    Framework,
    Modality,
    ModelStatus,
    RunStatus,
    TaskType,
)
from app.core.exceptions import ModelNotFoundError
from tests.factories import oct_phantom


class StubQualityModel(ClassicalModel):
    """Quality model whose verdict is fixed by the test."""

    def __init__(self, *, valid: bool) -> None:
        self.valid = valid
        self.metadata = ModelMetadata(
            model_id="stub_quality",
            name="Stub Quality",
            version="1.0.0",
            modality=Modality.OCT,
            task=TaskType.QUALITY,
            framework=Framework.CLASSICAL,
            evidence_level=EvidenceLevel.HEURISTIC,
        )
        super().__init__()

    def predict(self, prepared: ExamImage) -> bool:
        return self.valid

    def postprocess(self, output: bool, image: ExamImage) -> ModelResult:
        return ModelResult(
            model_id=self.model_id,
            model_version=self.version,
            task=TaskType.QUALITY,
            quality=QualityReport(
                quality_score=0.9 if output else 0.1,
                is_valid=output,
                issues=[] if output else ["blur"],
            ),
        )


class StubClassifier(ClassicalModel):
    """Classifier that either returns a fixed score or raises."""

    def __init__(self, model_id: str = "stub_classifier", *, fail: bool = False) -> None:
        self.fail = fail
        self.metadata = ModelMetadata(
            model_id=model_id,
            name="Stub Classifier",
            version="2.0.0",
            modality=Modality.OCT,
            task=TaskType.CLASSIFICATION,
            framework=Framework.CLASSICAL,
            evidence_level=EvidenceLevel.RESEARCH,
        )
        super().__init__()

    def predict(self, prepared: ExamImage) -> float:
        if self.fail:
            raise RuntimeError("simulated model crash")
        return 0.75

    def postprocess(self, output: float, image: ExamImage) -> ModelResult:
        return ModelResult(
            model_id=self.model_id,
            model_version=self.version,
            task=TaskType.CLASSIFICATION,
            predictions=[PredictionItem(label="finding", score=output)],
        )


class UnavailableModel(StubClassifier):
    """A model that reports itself unavailable, like a missing checkpoint."""

    def availability(self) -> Availability:
        return Availability(
            available=False,
            reason="Weights are not installed.",
            missing=["/models/oct/x.pt"],
            remediation="Place the checkpoint at /models/oct/x.pt.",
        )


@pytest.fixture
def image() -> ExamImage:
    pixels, _ = oct_phantom()
    return ExamImage(pixels=pixels, modality=Modality.OCT)


def _engine(*models: ClassicalModel) -> InferenceEngine:
    registry = ModelRegistry()
    for model in models:
        registry.register(model)
    return InferenceEngine(registry)


def test_quality_model_runs_first_even_if_requested_last(image: ExamImage) -> None:
    """The gate must be evaluated before any expensive model."""
    engine = _engine(StubQualityModel(valid=True), StubClassifier())
    config = PipelineConfig.from_request(["stub_classifier", "stub_quality"])
    outcome = engine.run(image, config)
    assert [result.model_id for result in outcome.results] == ["stub_quality", "stub_classifier"]


def test_failed_quality_gate_skips_downstream_models(image: ExamImage) -> None:
    engine = _engine(StubQualityModel(valid=False), StubClassifier())
    outcome = engine.run(image, PipelineConfig.from_request(["stub_quality", "stub_classifier"]))
    assert outcome.gate_passed is False
    statuses = {result.model_id: result.status for result in outcome.results}
    assert statuses["stub_classifier"] is RunStatus.SKIPPED_QUALITY
    assert outcome.quality is not None and outcome.quality.issues == ["blur"]


def test_quality_gate_can_be_disabled_per_analysis(image: ExamImage) -> None:
    engine = _engine(StubQualityModel(valid=False), StubClassifier())
    outcome = engine.run(
        image,
        PipelineConfig.from_request(["stub_quality", "stub_classifier"], quality_gate=False),
    )
    assert outcome.gate_passed is True
    statuses = {result.model_id: result.status for result in outcome.results}
    assert statuses["stub_classifier"] is RunStatus.COMPLETED


def test_one_failing_model_does_not_sink_the_analysis(image: ExamImage) -> None:
    engine = _engine(
        StubQualityModel(valid=True),
        StubClassifier("broken", fail=True),
        StubClassifier("working"),
    )
    outcome = engine.run(image, PipelineConfig.from_request(["stub_quality", "broken", "working"]))
    statuses = {result.model_id: result.status for result in outcome.results}
    assert statuses["broken"] is RunStatus.FAILED
    assert statuses["working"] is RunStatus.COMPLETED
    broken = next(r for r in outcome.results if r.model_id == "broken")
    assert broken.error_message
    assert broken.predictions == []  # nothing invented for a failed run


def test_unavailable_model_is_skipped_with_remediation(image: ExamImage) -> None:
    engine = _engine(UnavailableModel("needs_weights"))
    outcome = engine.run(image, PipelineConfig.from_request(["needs_weights"]))
    result = outcome.results[0]
    assert result.status is RunStatus.SKIPPED_UNAVAILABLE
    assert "checkpoint" in result.error_message.lower()


def test_unknown_model_is_rejected_before_running(image: ExamImage) -> None:
    engine = _engine(StubClassifier())
    with pytest.raises(ModelNotFoundError):
        engine.run(image, PipelineConfig.from_request(["does_not_exist"]))


def test_deprecated_models_are_not_runnable(image: ExamImage) -> None:
    model = StubClassifier("legacy")
    model.metadata = model.metadata.model_copy(update={"status": ModelStatus.DEPRECATED})
    outcome = _engine(model).run(image, PipelineConfig.from_request(["legacy"]))
    assert outcome.results[0].status is RunStatus.SKIPPED_UNAVAILABLE


def test_results_record_execution_context(image: ExamImage) -> None:
    """Every completed run must carry what is needed to reproduce it."""
    outcome = _engine(StubClassifier()).run(image, PipelineConfig.from_request(["stub_classifier"]))
    result = outcome.results[0]
    assert result.input_hash == image.input_fingerprint()
    assert result.device_info.device in {"cpu", "cuda"}
    assert result.model_version == "2.0.0"
    assert result.processing_time_ms >= 0


def test_model_selection_parses_version_pins() -> None:
    assert ModelSelection.parse("abc@1.2.3") == ModelSelection(model_id="abc", version="1.2.3")
    assert ModelSelection.parse("abc").version is None
    assert ModelSelection.parse({"model_id": "abc", "version": "2.0"}).version == "2.0"
    assert ModelSelection(model_id="a", version="1.0").as_string() == "a@1.0"


def test_default_pipeline_is_used_when_no_model_is_requested(image: ExamImage) -> None:
    """An empty request runs the modality's available models, quality first."""
    from app.ai.registry import registry

    outcome = InferenceEngine(registry).run(image, PipelineConfig.from_request([]))
    executed = [result.model_id for result in outcome.results]
    assert executed[0] == "oct_quality_v1"
    assert "oct_layers_classical_v1" in executed
    # Catalogued-but-unavailable models are never part of a default pipeline.
    assert "oct_retinal_layers_v1" not in executed


def test_select_frame_strategies() -> None:
    frames = [ExamImage(pixels=np.zeros((2, 2), np.uint8), frame_index=i) for i in range(5)]
    assert select_frame(frames, "first").frame_index == 0
    assert select_frame(frames, "middle").frame_index == 2
    assert select_frame(frames, "last").frame_index == 4
    with pytest.raises(ValueError):
        select_frame([])
