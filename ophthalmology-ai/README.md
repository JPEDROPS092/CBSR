# Ophthalmology AI Platform

A modular backend that **orchestrates deep-learning models** over ophthalmology
exams (colour fundus photography and OCT). It is not built around one model:
models are plugins, and adding one is a deployment action, not a code change.

> **Research and decision-support tool.** Outputs are probabilities, scores,
> measurements and segmentations — never a diagnosis. This software is not a
> cleared or certified medical device. See [`docs/SECURITY.md`](docs/SECURITY.md)
> and the disclaimer attached to every report.
>
> *Ferramenta de pesquisa e apoio à decisão. Os resultados não constituem
> diagnóstico médico e não substituem a interpretação por profissional
> habilitado.*

---

## What it does today

```
upload → validation → quality control → [gate] → preprocessing → inference
       → postprocessing → explainability → results → database → report
```

Runs out of the box, with **no weights to download**:

| Model | Modality | Task | Kind |
|---|---|---|---|
| `fundus_quality_v1` | fundus | quality | heuristic (blur, exposure, FOV, uniformity, modality check) |
| `oct_quality_v1` | OCT | quality | heuristic (signal, SNR, truncation, tilt, modality check) |
| `oct_layers_classical_v1` | OCT | segmentation | ILM/RPE boundaries + retinal thickness (µm) |
| `fundus_vessels_classical_v1` | fundus | segmentation | morphological vessel segmentation |

Catalogued and ready for weights — DR grading, glaucoma suspicion, optic
disc/cup, learned vessels, macular abnormality, OCT layers/fluid/biomarkers/
disease/glaucoma. Each is listed by `GET /api/v1/models` as *unavailable* with
the exact file to install. See [`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md).

**The platform never invents weights, metrics or predictions.** A model without
a checkpoint is reported as unavailable; it does not return a neutral guess.

## Quick start (Docker)

```bash
cp .env.example .env                       # then edit JWT_SECRET and the storage keys
docker compose up --build                  # api, worker, postgres, redis, minio
docker compose exec api python scripts/seed_admin.py --email admin@clinic.org --name Admin
open http://localhost:8000/docs
```

## Quick start (local, no services)

SQLite + local file storage + inline task execution — enough to run the whole
pipeline on a laptop:

```bash
pip install -e ".[dev]"
export JWT_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"
export DATABASE_URL="sqlite+pysqlite:///./ophthalmology.db"
export STORAGE_BACKEND=local TASK_QUEUE_BACKEND=inline
alembic upgrade head
python scripts/seed_admin.py --email admin@example.com --name Admin
uvicorn app.main:app --reload
python scripts/smoke_test.py --email admin@example.com --password '...'
```

## The MVP flow

```bash
POST /api/v1/patients                 {"external_ref": "P-2024-0001"}
POST /api/v1/exams                    {"patient_id": ..., "modality": "oct",
                                       "acquisition_metadata": {"pixel_spacing_um": {"axial": 3.87}}}
POST /api/v1/exams/{exam_id}/upload   (multipart file)
POST /api/v1/analysis                 {"exam_id": ..., "models": ["oct_layers_classical_v1"]}
  → 202 {"analysis_id": "...", "status": "queued"}
GET  /api/v1/analysis/{analysis_id}
  → {"status": "completed", "models": [{... "segmentations": [{"mask_url": ...}],
     "measurements": {"retinal_thickness_um": {"mean": 315.2}}}]}
POST /api/v1/reports/{analysis_id}    {"format": "json" | "html" | "pdf"}
```

## Documentation

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Layers, the plugin rule, request and inference flow |
| [`docs/API.md`](docs/API.md) | Endpoints, payloads, errors, result schema |
| [`docs/MODEL_REGISTRY.md`](docs/MODEL_REGISTRY.md) | Installing a model, the manifest schema, versioning |
| [`docs/MODEL_LICENSES.md`](docs/MODEL_LICENSES.md) | License and dataset tracking for every model |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Docker, GPU workers, scaling, migrations, backups |
| [`docs/SECURITY.md`](docs/SECURITY.md) | AuthN/Z, RBAC, privacy, LGPD posture, audit trail |
| [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) | Layout, tests, adding a model in code, conventions |

Interactive API docs: `/docs` (Swagger UI) and `/redoc`.

## Stack

FastAPI · Pydantic v2 · SQLAlchemy 2 · PostgreSQL · Redis · Celery · PyTorch /
ONNX Runtime (optional) · OpenCV · NumPy · Pillow · Alembic · pytest · Docker.

PyTorch, MONAI and ONNX Runtime are **optional extras** (`pip install -e ".[ml]"`).
The API, the built-in models and the whole test suite run without them, so a
CPU-only deployment stays small.

## Tests

```bash
pytest                    # 121 tests: unit, AI behaviour, API integration
pytest tests/ai -q        # model behaviour against synthetic phantoms
```

AI tests assert on behaviour under known defects and on measurements recovered
from phantoms with known ground truth — they never assert an accuracy the
platform has not measured.

## Status and limitations

* MVP / research platform. Not validated clinically, not a registered medical
  device, no regulatory clearance.
* The built-in models are **heuristics** — engineering signals for triage and
  QA of an acquisition workflow, not clinical evidence.
* DICOM ingestion is not implemented; export pixel data as PNG/TIFF first.
* Thickness and area in physical units require the device's pixel scale; without
  it the platform reports pixels and says why.
