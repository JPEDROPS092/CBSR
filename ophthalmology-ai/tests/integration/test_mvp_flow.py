"""The MVP acceptance flow, end to end.

Creates a patient, an exam and an OCT upload, runs an analysis, reads the
results back, fetches the segmentation mask and generates a report - the ten
steps the platform's MVP is defined by.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.factories import encode, fundus_phantom, noise_image, oct_phantom


def test_full_oct_pipeline(client: TestClient, doctor_auth: dict[str, str]) -> None:
    """Patient -> exam -> upload -> analysis -> results -> mask -> report."""
    # 1. Create a patient.
    patient = client.post(
        "/api/v1/patients",
        json={"external_ref": "P-MVP-0001", "birth_year": 1958, "sex": "female"},
        headers=doctor_auth,
    )
    assert patient.status_code == 201, patient.text
    patient_id = patient.json()["id"]

    # 2. Create an OCT exam carrying the device's pixel scale.
    exam = client.post(
        "/api/v1/exams",
        json={
            "patient_id": patient_id,
            "modality": "oct",
            "laterality": "od",
            "device_manufacturer": "Phantom",
            "acquisition_metadata": {"pixel_spacing_um": {"axial": 3.87, "lateral": 11.7}},
        },
        headers=doctor_auth,
    )
    assert exam.status_code == 201, exam.text
    exam_id = exam.json()["id"]

    # 3-4. Upload a B-scan; quality control screens it inline.
    pixels, truth_px = oct_phantom()
    upload = client.post(
        f"/api/v1/exams/{exam_id}/upload",
        files={"file": ("scan.png", encode(pixels), "image/png")},
        headers=doctor_auth,
    )
    assert upload.status_code == 201, upload.text
    body = upload.json()
    assert body["image"]["width"] == 512
    assert body["quality"]["is_valid"] is True
    assert body["image"]["status"] == "validated"

    # 5-6. Run the analysis (inline queue executes it immediately).
    analysis = client.post(
        "/api/v1/analysis",
        json={
            "exam_id": exam_id,
            "models": ["oct_quality_v1", "oct_layers_classical_v1"],
        },
        headers=doctor_auth,
    )
    assert analysis.status_code == 202, analysis.text
    analysis_id = analysis.json()["analysis_id"]

    # 7-8. Read the stored results back.
    fetched = client.get(f"/api/v1/analysis/{analysis_id}", headers=doctor_auth)
    assert fetched.status_code == 200, fetched.text
    result = fetched.json()
    assert result["status"] == "completed"
    assert result["disclaimer"]
    runs = {run["model_id"]: run for run in result["models"]}
    assert set(runs) == {"oct_quality_v1", "oct_layers_classical_v1"}
    assert all(run["status"] == "completed" for run in runs.values())

    quality_run = runs["oct_quality_v1"]
    assert quality_run["quality"]["is_valid"] is True
    assert quality_run["processing_time_ms"] > 0
    assert quality_run["input_hash"]

    layers = runs["oct_layers_classical_v1"]
    thickness = layers["measurements"]["retinal_thickness_um"]["mean"]
    # The phantom's ground truth, converted with the exam's axial scale.
    assert abs(thickness - truth_px * 3.87) < 25.0

    # 9. Fetch the segmentation mask through the authenticated object endpoint.
    segmentation = layers["segmentations"][0]
    assert segmentation["label"] == "retina_ilm_to_rpe"
    mask = client.get(segmentation["mask_url"], headers=doctor_auth)
    assert mask.status_code == 200
    assert mask.headers["content-type"] == "image/png"
    assert mask.content[:8] == b"\x89PNG\r\n\x1a\n"

    # Explainability artifacts were produced and are retrievable.
    assert {artifact["kind"] for artifact in layers["artifacts"]} == {"overlay"}
    overlay = client.get(layers["artifacts"][0]["artifact_url"], headers=doctor_auth)
    assert overlay.status_code == 200

    # 10. Generate a JSON report.
    report = client.post(
        f"/api/v1/reports/{analysis_id}",
        json={"format": "json"},
        headers=doctor_auth,
    )
    assert report.status_code == 201, report.text
    payload = report.json()["payload"]
    assert payload["exam"]["modality"] == "oct"
    assert payload["disclaimer"]["version"]
    assert any(f["name"] == "retina_ilm_to_rpe" for f in payload["findings"])
    assert "oct_layers_classical_v1" in payload["measurements"]


def test_quality_gate_blocks_downstream_models(
    client: TestClient, doctor_auth: dict[str, str]
) -> None:
    """An unusable image is rejected and no downstream model is run on it."""
    patient = client.post(
        "/api/v1/patients", json={"external_ref": "P-MVP-0002"}, headers=doctor_auth
    ).json()
    exam = client.post(
        "/api/v1/exams",
        json={"patient_id": patient["id"], "modality": "oct"},
        headers=doctor_auth,
    ).json()
    upload = client.post(
        f"/api/v1/exams/{exam['id']}/upload",
        files={"file": ("noise.png", encode(noise_image()), "image/png")},
        headers=doctor_auth,
    )
    assert upload.json()["quality"]["is_valid"] is False
    assert upload.json()["image"]["status"] == "rejected"

    analysis = client.post(
        "/api/v1/analysis",
        json={"exam_id": exam["id"], "models": ["oct_quality_v1", "oct_layers_classical_v1"]},
        headers=doctor_auth,
    ).json()
    result = client.get(f"/api/v1/analysis/{analysis['analysis_id']}", headers=doctor_auth).json()

    assert result["status"] == "completed"
    assert result["quality_summary"]["gate_passed"] is False
    runs = {run["model_id"]: run for run in result["models"]}
    assert runs["oct_layers_classical_v1"]["status"] == "skipped_quality"
    assert not runs["oct_layers_classical_v1"]["segmentations"]


def test_fundus_pipeline_runs_vessel_segmentation(
    client: TestClient, doctor_auth: dict[str, str]
) -> None:
    """The default fundus pipeline runs quality control plus vessel segmentation."""
    patient = client.post(
        "/api/v1/patients", json={"external_ref": "P-MVP-0003"}, headers=doctor_auth
    ).json()
    exam = client.post(
        "/api/v1/exams",
        json={"patient_id": patient["id"], "modality": "fundus", "laterality": "os"},
        headers=doctor_auth,
    ).json()
    client.post(
        f"/api/v1/exams/{exam['id']}/upload",
        files={"file": ("fundus.png", encode(fundus_phantom()), "image/png")},
        headers=doctor_auth,
    )

    analysis = client.post(
        "/api/v1/analysis", json={"exam_id": exam["id"]}, headers=doctor_auth
    ).json()
    result = client.get(f"/api/v1/analysis/{analysis['analysis_id']}", headers=doctor_auth).json()

    executed = {run["model_id"] for run in result["models"]}
    assert executed == {"fundus_quality_v1", "fundus_vessels_classical_v1"}
    vessels = next(
        run for run in result["models"] if run["model_id"] == "fundus_vessels_classical_v1"
    )
    assert vessels["status"] == "completed"
    assert vessels["measurements"]["vessel_area_ratio"] >= 0.0


def test_unavailable_model_is_skipped_with_remediation(
    client: TestClient, doctor_auth: dict[str, str]
) -> None:
    """A catalogued model without weights explains what to install."""
    patient = client.post(
        "/api/v1/patients", json={"external_ref": "P-MVP-0004"}, headers=doctor_auth
    ).json()
    exam = client.post(
        "/api/v1/exams",
        json={"patient_id": patient["id"], "modality": "oct"},
        headers=doctor_auth,
    ).json()
    pixels, _ = oct_phantom()
    client.post(
        f"/api/v1/exams/{exam['id']}/upload",
        files={"file": ("scan.png", encode(pixels), "image/png")},
        headers=doctor_auth,
    )

    analysis = client.post(
        "/api/v1/analysis",
        json={"exam_id": exam["id"], "models": ["oct_retinal_layers_v1"]},
        headers=doctor_auth,
    ).json()
    result = client.get(f"/api/v1/analysis/{analysis['analysis_id']}", headers=doctor_auth).json()

    run = result["models"][0]
    assert run["status"] == "skipped_unavailable"
    assert "manifest" in run["error_message"].lower()
    # Nothing was invented in place of a real prediction.
    assert run["predictions"] == []
    assert run["segmentations"] == []


def test_html_report_is_rendered_and_downloadable(
    client: TestClient, doctor_auth: dict[str, str]
) -> None:
    """An HTML report is stored and served with its disclaimer."""
    patient = client.post(
        "/api/v1/patients", json={"external_ref": "P-MVP-0005"}, headers=doctor_auth
    ).json()
    exam = client.post(
        "/api/v1/exams",
        json={"patient_id": patient["id"], "modality": "oct"},
        headers=doctor_auth,
    ).json()
    pixels, _ = oct_phantom()
    client.post(
        f"/api/v1/exams/{exam['id']}/upload",
        files={"file": ("scan.png", encode(pixels), "image/png")},
        headers=doctor_auth,
    )
    analysis = client.post(
        "/api/v1/analysis",
        json={"exam_id": exam["id"], "models": ["oct_layers_classical_v1"]},
        headers=doctor_auth,
    ).json()

    report = client.post(
        f"/api/v1/reports/{analysis['analysis_id']}",
        json={"format": "html"},
        headers=doctor_auth,
    )
    assert report.status_code == 201, report.text
    report_id = report.json()["id"]
    assert report.json()["document_url"]

    document = client.get(f"/api/v1/reports/{report_id}/document", headers=doctor_auth)
    assert document.status_code == 200
    assert document.headers["content-type"].startswith("text/html")
    assert "Disclaimer" in document.text
    assert "not a medical diagnosis" in document.text
