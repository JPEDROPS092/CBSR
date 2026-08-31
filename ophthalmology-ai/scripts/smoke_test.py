#!/usr/bin/env python3
"""End-to-end smoke test against a running API.

Walks the MVP flow - patient, exam, upload, analysis, results, report - using a
synthetic OCT phantom, and prints what each step returned. Useful to verify a
fresh deployment without any real patient data.

Usage::

    python scripts/smoke_test.py --base-url http://localhost:8000 \\
        --email admin@clinic.org --password "..."
"""

from __future__ import annotations

import argparse
import io
import sys

import httpx
import numpy as np
from PIL import Image as PILImage


def build_oct_phantom(height: int = 496, width: int = 512) -> bytes:
    """A synthetic B-scan with a bright ILM and RPE band."""
    rng = np.random.default_rng(0)
    image = np.zeros((height, width), dtype=np.float32)
    for x in range(width):
        top = int(height * 0.36 + 12 * np.sin(x / 60.0))
        image[top : top + 5, x] = 130
        image[top + 5 : top + 71, x] = 55
        image[top + 71 : top + 80, x] = 210
    image = np.clip(image + rng.normal(0, 7, (height, width)), 0, 255).astype(np.uint8)
    buffer = io.BytesIO()
    PILImage.fromarray(image, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> int:
    """Run the smoke test; returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--email", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--external-ref", default="SMOKE-0001")
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=120.0)

    tokens = client.post(
        "/api/v1/auth/login", json={"email": args.email, "password": args.password}
    )
    tokens.raise_for_status()
    headers = {"Authorization": f"Bearer {tokens.json()['access_token']}"}
    print("authenticated")

    patient = client.post(
        "/api/v1/patients",
        json={"external_ref": args.external_ref, "birth_year": 1960},
        headers=headers,
    )
    if patient.status_code == 409:
        print("patient already exists; pass a different --external-ref")
        return 1
    patient.raise_for_status()
    print(f"patient {patient.json()['id']}")

    exam = client.post(
        "/api/v1/exams",
        json={
            "patient_id": patient.json()["id"],
            "modality": "oct",
            "laterality": "od",
            "acquisition_metadata": {"pixel_spacing_um": {"axial": 3.87, "lateral": 11.7}},
        },
        headers=headers,
    )
    exam.raise_for_status()
    exam_id = exam.json()["id"]
    print(f"exam {exam_id}")

    upload = client.post(
        f"/api/v1/exams/{exam_id}/upload",
        files={"file": ("phantom.png", build_oct_phantom(), "image/png")},
        headers=headers,
    )
    upload.raise_for_status()
    quality = upload.json()["quality"]
    print(f"upload quality: score={quality['quality_score']} valid={quality['is_valid']}")

    analysis = client.post(
        "/api/v1/analysis",
        json={"exam_id": exam_id, "models": ["oct_quality_v1", "oct_layers_classical_v1"]},
        headers=headers,
    )
    analysis.raise_for_status()
    analysis_id = analysis.json()["analysis_id"]
    print(f"analysis {analysis_id} status={analysis.json()['status']}")

    for _ in range(60):
        result = client.get(f"/api/v1/analysis/{analysis_id}", headers=headers)
        result.raise_for_status()
        if result.json()["status"] in ("completed", "failed", "cancelled"):
            break
    body = result.json()
    print(f"analysis finished: {body['status']}")
    for run in body["models"]:
        print(
            f"  {run['model_id']:32s} {run['status']:20s} "
            f"{run['processing_time_ms']} ms on {run['device']}"
        )
        for segmentation in run["segmentations"]:
            print(f"    mask {segmentation['label']}: {segmentation['mask_url']}")
        if run["measurements"]:
            print(f"    measurements: {run['measurements']}")

    report = client.post(f"/api/v1/reports/{analysis_id}", json={"format": "json"}, headers=headers)
    report.raise_for_status()
    print(f"report {report.json()['id']} with {len(report.json()['payload']['findings'])} findings")
    return 0 if body["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
