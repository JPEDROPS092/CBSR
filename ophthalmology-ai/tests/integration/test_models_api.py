"""The model-registry API: the catalogue and its installation instructions."""

from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import ModelRecord


def test_list_models_exposes_the_whole_catalogue(
    client: TestClient, viewer_auth: dict[str, str]
) -> None:
    response = client.get("/api/v1/models", headers=viewer_auth)
    assert response.status_code == 200
    models = {model["model_id"]: model for model in response.json()}
    assert "oct_quality_v1" in models
    assert models["oct_quality_v1"]["available"] is True
    assert models["oct_quality_v1"]["evidence_level"] == "heuristic"
    # Catalogued but not installed here.
    assert models["fundus_dr_grading_v1"]["available"] is False


def test_filters(client: TestClient, viewer_auth: dict[str, str]) -> None:
    oct_models = client.get(
        "/api/v1/models", params={"modality": "oct"}, headers=viewer_auth
    ).json()
    assert oct_models and all(model["modality"] == "oct" for model in oct_models)

    runnable = client.get(
        "/api/v1/models", params={"available_only": True}, headers=viewer_auth
    ).json()
    assert all(model["available"] for model in runnable)
    assert len(runnable) < len(client.get("/api/v1/models", headers=viewer_auth).json())


def test_model_detail_of_a_built_in_model(client: TestClient, viewer_auth: dict[str, str]) -> None:
    response = client.get("/api/v1/models/oct_layers_classical_v1", headers=viewer_auth)
    assert response.status_code == 200
    detail = response.json()
    assert detail["availability"]["available"] is True
    assert detail["license"]["name"] == "Apache-2.0"
    assert detail["limitations"]
    assert detail["output_spec"]["segmentation_classes"] == ["background", "retina_ilm_to_rpe"]
    # No metric is claimed for a model nobody has evaluated here.
    assert detail["reported_metrics"] == {}


def test_unavailable_model_detail_says_exactly_what_to_install(
    client: TestClient, viewer_auth: dict[str, str]
) -> None:
    detail = client.get("/api/v1/models/oct_retinal_layers_v1", headers=viewer_auth).json()
    availability = detail["availability"]
    assert availability["available"] is False
    assert availability["missing"][0].endswith("oct_retinal_layers_v1.json")
    assert "manifest" in availability["remediation"]
    assert detail["status"] == "unavailable"


def test_unknown_model_is_404(client: TestClient, viewer_auth: dict[str, str]) -> None:
    response = client.get("/api/v1/models/no_such_model", headers=viewer_auth)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "model_not_found"


def test_refresh_is_admin_only_and_persists_model_records(
    client: TestClient, admin_auth: dict[str, str], viewer_auth: dict[str, str], session: Session
) -> None:
    """Refreshing syncs the registry into the durable model table."""
    assert client.post("/api/v1/models/refresh", headers=viewer_auth).status_code == 403

    response = client.post("/api/v1/models/refresh", headers=admin_auth)
    assert response.status_code == 200

    records = session.execute(select(ModelRecord)).scalars().all()
    by_id = {record.model_id: record for record in records}
    assert "oct_quality_v1" in by_id
    assert by_id["oct_quality_v1"].version == "1.0.0"
    assert by_id["oct_quality_v1"].license_name == "Apache-2.0"
    assert by_id["oct_quality_v1"].input_spec is not None


def test_health_reports_registry_and_device(client: TestClient) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["database"] == "ok"
    assert health["storage"] == "ok"
    assert health["models_registered"] >= health["models_available"] > 0
    assert health["device"]["device"] in {"cpu", "cuda"}
    assert "torch_available" in health["device"]
