"""Upload validation, deduplication and privacy behaviour."""

from __future__ import annotations

import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image as PILImage
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import Image
from tests.factories import encode, fundus_phantom, oct_phantom


def _exam(client: TestClient, auth: dict[str, str], modality: str = "fundus") -> str:
    patient = client.post(
        "/api/v1/patients", json={"external_ref": f"P-UP-{modality}"}, headers=auth
    ).json()
    return client.post(
        "/api/v1/exams", json={"patient_id": patient["id"], "modality": modality}, headers=auth
    ).json()["id"]


def test_upload_stores_metadata_and_returns_quality(
    client: TestClient, doctor_auth: dict[str, str]
) -> None:
    exam_id = _exam(client, doctor_auth)
    response = client.post(
        f"/api/v1/exams/{exam_id}/upload",
        files={"file": ("photo.png", encode(fundus_phantom()), "image/png")},
        headers=doctor_auth,
    )
    assert response.status_code == 201
    image = response.json()["image"]
    assert image["content_type"] == "image/png"
    assert image["width"] == image["height"] == 384
    assert len(image["checksum_sha256"]) == 64
    assert response.json()["quality"]["model_id"] == "fundus_quality_v1"


def test_non_image_upload_is_rejected(client: TestClient, doctor_auth: dict[str, str]) -> None:
    exam_id = _exam(client, doctor_auth)
    response = client.post(
        f"/api/v1/exams/{exam_id}/upload",
        files={"file": ("report.pdf", b"%PDF-1.7 not an image", "image/png")},
        headers=doctor_auth,
    )
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_media_type"


def test_content_type_is_taken_from_the_bytes_not_the_client(
    client: TestClient, doctor_auth: dict[str, str]
) -> None:
    """A mislabelled upload is classified by its magic number."""
    exam_id = _exam(client, doctor_auth)
    jpeg = io.BytesIO()
    PILImage.fromarray(fundus_phantom(size=64)).save(jpeg, format="JPEG")
    response = client.post(
        f"/api/v1/exams/{exam_id}/upload",
        files={"file": ("lying.png", jpeg.getvalue(), "image/png")},
        headers=doctor_auth,
    )
    assert response.json()["image"]["content_type"] == "image/jpeg"


def test_empty_upload_is_rejected(client: TestClient, doctor_auth: dict[str, str]) -> None:
    exam_id = _exam(client, doctor_auth)
    response = client.post(
        f"/api/v1/exams/{exam_id}/upload",
        files={"file": ("empty.png", b"", "image/png")},
        headers=doctor_auth,
    )
    assert response.status_code == 422


def test_identical_upload_is_deduplicated(client: TestClient, doctor_auth: dict[str, str]) -> None:
    exam_id = _exam(client, doctor_auth)
    payload = encode(fundus_phantom(size=96))
    first = client.post(
        f"/api/v1/exams/{exam_id}/upload",
        files={"file": ("a.png", payload, "image/png")},
        headers=doctor_auth,
    ).json()
    second = client.post(
        f"/api/v1/exams/{exam_id}/upload",
        files={"file": ("b.png", payload, "image/png")},
        headers=doctor_auth,
    ).json()
    assert first["image"]["id"] == second["image"]["id"]
    assert len(client.get(f"/api/v1/exams/{exam_id}/images", headers=doctor_auth).json()) == 1


def test_uploaded_filename_is_not_persisted(
    client: TestClient, doctor_auth: dict[str, str], session: Session
) -> None:
    """Filenames routinely contain patient names, so only the extension is kept."""
    exam_id = _exam(client, doctor_auth)
    client.post(
        f"/api/v1/exams/{exam_id}/upload",
        files={"file": ("Maria_Silva_OD_2024.png", encode(fundus_phantom(size=64)), "image/png")},
        headers=doctor_auth,
    )
    image = session.execute(select(Image)).scalars().one()
    assert image.original_extension == ".png"
    assert "Maria" not in image.storage_key
    assert "Maria" not in str(image.__dict__)


def test_pixel_scale_from_the_exam_reaches_the_image(
    client: TestClient, doctor_auth: dict[str, str]
) -> None:
    patient = client.post(
        "/api/v1/patients", json={"external_ref": "P-UP-SCALE"}, headers=doctor_auth
    ).json()
    exam = client.post(
        "/api/v1/exams",
        json={
            "patient_id": patient["id"],
            "modality": "oct",
            "acquisition_metadata": {"pixel_spacing_um": {"axial": 3.87, "lateral": 11.7}},
        },
        headers=doctor_auth,
    ).json()
    pixels, _ = oct_phantom()
    response = client.post(
        f"/api/v1/exams/{exam['id']}/upload",
        files={"file": ("scan.png", encode(pixels), "image/png")},
        headers=doctor_auth,
    )
    assert response.json()["image"]["pixel_spacing_um"] == {"axial": 3.87, "lateral": 11.7}


def test_multi_frame_series_is_recorded_and_the_middle_frame_analysed(
    client: TestClient, doctor_auth: dict[str, str]
) -> None:
    """A B-scan series records its frame count; the pipeline analyses one frame."""
    exam_id = _exam(client, doctor_auth, modality="oct")
    frames = []
    for index in range(3):
        pixels, _ = oct_phantom(height=200, width=256, retina_thickness_px=60 + index * 10)
        frames.append(PILImage.fromarray(pixels))
    buffer = io.BytesIO()
    frames[0].save(buffer, format="TIFF", save_all=True, append_images=frames[1:])

    upload = client.post(
        f"/api/v1/exams/{exam_id}/upload",
        files={"file": ("series.tif", buffer.getvalue(), "image/tiff")},
        headers=doctor_auth,
    )
    assert upload.status_code == 201
    assert upload.json()["image"]["num_frames"] == 3

    analysis = client.post(
        "/api/v1/analysis",
        json={
            "exam_id": exam_id,
            "models": ["oct_layers_classical_v1"],
            "frame_selection": "middle",
            "quality_gate": False,
        },
        headers=doctor_auth,
    ).json()
    result = client.get(f"/api/v1/analysis/{analysis['analysis_id']}", headers=doctor_auth).json()
    run = result["models"][0]
    assert run["status"] == "completed"
    thickness = run["measurements"]["retinal_thickness_px"]["mean"]
    # The middle frame is the 70-pixel one, not the 60 or 80 pixel neighbours.
    assert 60 < thickness < 75


def test_operator_can_upload_but_not_run_analyses(client: TestClient) -> None:
    from app.core.enums import UserRole
    from tests.conftest import _auth_header

    operator = _auth_header(client, UserRole.OPERATOR)
    exam_id = _exam(client, operator)
    upload = client.post(
        f"/api/v1/exams/{exam_id}/upload",
        files={"file": ("f.png", encode(fundus_phantom(size=64)), "image/png")},
        headers=operator,
    )
    assert upload.status_code == 201
    denied = client.post("/api/v1/analysis", json={"exam_id": exam_id}, headers=operator)
    assert denied.status_code == 403


def test_upload_to_unknown_exam_is_404(client: TestClient, doctor_auth: dict[str, str]) -> None:
    response = client.post(
        "/api/v1/exams/00000000-0000-0000-0000-000000000000/upload",
        files={"file": ("f.png", encode(np.zeros((8, 8), np.uint8)), "image/png")},
        headers=doctor_auth,
    )
    assert response.status_code == 404
