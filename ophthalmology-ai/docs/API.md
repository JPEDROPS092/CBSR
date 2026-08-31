# API

Base path `/api/v1`. Interactive docs at `/docs` (Swagger UI) and `/redoc`;
the OpenAPI document is at `/openapi.json`.

All requests except `/health`, `/api/v1/auth/login` and `/api/v1/auth/refresh`
require `Authorization: Bearer <access token>`.

## Conventions

* Identifiers are UUIDs everywhere. No endpoint accepts or returns a patient
  name or national identifier.
* Timestamps are ISO-8601 UTC.
* Every response carries `X-Request-ID`; send your own to correlate traces.
* Errors share one envelope:

```json
{"error": {"code": "quality_gate_rejected", "message": "...", "details": {}}}
```

| Status | Codes |
|---|---|
| 401 | `authentication_failed` |
| 403 | `permission_denied` |
| 404 | `not_found`, `model_not_found` |
| 409 | `conflict`, `model_unavailable` |
| 413 | `payload_too_large` |
| 415 | `unsupported_media_type` |
| 422 | `validation_error`, `quality_gate_rejected` |
| 429 | `rate_limited` |
| 500/502 | `internal_error`, `inference_failed`, `storage_error` |

Validation errors report the offending field and reason but **never echo the
submitted value** — a rejected payload can contain patient data.

## Auth

| Method | Path | Notes |
|---|---|---|
| POST | `/auth/login` | `{email, password}` → access + refresh tokens |
| POST | `/auth/refresh` | `{refresh_token}` → rotated pair |
| GET | `/auth/me` | current user |
| POST | `/auth/users` | admin only; creates a user with a role |

## Patients

| Method | Path | Permission |
|---|---|---|
| POST | `/patients` | `patient:write` |
| GET | `/patients` | `patient:read` (`limit`, `offset`, `external_ref`) |
| GET | `/patients/{patient_id}` | `patient:read` |
| PATCH | `/patients/{patient_id}` | `patient:write` |
| GET | `/patients/{patient_id}/exams` | `exam:read` |

```json
POST /patients
{"external_ref": "P-2024-0001", "birth_year": 1958, "sex": "female",
 "consent_research": true}
```

`external_ref` is a site-assigned pseudonym. A value shaped like a national
identifier (CPF) is rejected with 422.

## Exams and uploads

| Method | Path | Permission |
|---|---|---|
| POST | `/exams` | `exam:write` |
| GET | `/exams/{exam_id}` | `exam:read` |
| GET | `/exams/{exam_id}/images` | `exam:read` |
| POST | `/exams/{exam_id}/upload` | `image:upload` |

```json
POST /exams
{"patient_id": "…", "modality": "oct", "laterality": "od",
 "device_manufacturer": "…",
 "acquisition_metadata": {"pixel_spacing_um": {"axial": 3.87, "lateral": 11.7}}}
```

`pixel_spacing_um` is what turns pixel measurements into micrometres. Without
it the platform reports pixels and adds a warning saying so.

Upload is `multipart/form-data` with a `file` part (JPEG, PNG or TIFF; a
multi-page TIFF is read as a B-scan series). The type is determined from the
file's magic number, not from the filename or the declared content type. The
filename itself is discarded — only the extension is kept.

```json
201 {"image": {"id": "…", "width": 512, "height": 496, "num_frames": 1,
                "checksum_sha256": "…", "status": "validated"},
     "quality": {"model_id": "oct_quality_v1", "quality_score": 0.96,
                 "is_valid": true, "issues": [], "recommendation": "…"}}
```

The upload-time quality check is advisory: it tells an operator to re-acquire
while the patient is still in the chair, but it never blocks the upload.

## Analysis

| Method | Path | Permission |
|---|---|---|
| POST | `/analysis` | `analysis:run` |
| GET | `/analysis/{analysis_id}` | `analysis:read` |
| POST | `/analysis/{analysis_id}/cancel` | `analysis:run` (queued only) |
| GET | `/exams/{exam_id}/analyses` | `analysis:read` |

```json
POST /analysis
{"exam_id": "…",
 "image_id": null,                       // defaults to the exam's latest image
 "models": ["oct_quality_v1", "oct_layers_classical_v1@1.0.0"],
 "quality_gate": null,                   // null = follow QUALITY_GATE_ENABLED
 "explainability": null,
 "frame_selection": "middle"}            // first | middle | last

202 {"analysis_id": "…", "status": "queued", "task_id": "…"}
```

An empty `models` list runs the default pipeline for the exam's modality:
quality control first, then every other available model. Unavailable models are
never part of a default pipeline. A model id that is not registered is rejected
immediately with 404.

Statuses: `queued → processing → completed | failed | cancelled`.

```json
GET /analysis/{analysis_id}
{
  "id": "…", "status": "completed",
  "quality_summary": {"gate_passed": true, "quality_model_id": "oct_quality_v1",
                      "quality": {...}, "models_run": 2, "models_completed": 2},
  "models": [{
    "model_id": "oct_layers_classical_v1", "model_version": "1.0.0",
    "task": "segmentation", "status": "completed",
    "device": "cpu", "precision": "fp32", "batch_size": 1,
    "processing_time_ms": 412.0, "input_hash": "…", "software_version": "0.1.0",
    "predictions": [],
    "segmentations": [{"label": "retina_ilm_to_rpe",
                       "mask_url": "/api/v1/objects/results/…png",
                       "area_px": 42165.0, "area_ratio": 0.166,
                       "measurements": {"area_mm2": 1.909}}],
    "artifacts": [{"kind": "overlay", "artifact_url": "…"}],
    "measurements": {"retinal_thickness_um": {"mean": 314.8, "min": 286.4,
                                              "max": 356.0, "std": 24.3},
                     "central_subfield_thickness_um": 353.9},
    "warnings": ["Detects only the outer envelope of the retina …"]
  }],
  "disclaimer": "Research and decision-support tool. …"
}
```

Per-model run statuses:

| Status | Meaning |
|---|---|
| `completed` | ran and produced results |
| `failed` | raised; `error_message` explains, and nothing is invented |
| `skipped_quality` | the image failed quality control |
| `skipped_unavailable` | no weights/runtime here; `error_message` says what to install |

## Models

| Method | Path | Permission |
|---|---|---|
| GET | `/models` | `model:read` (`modality`, `task`, `status`, `available_only`) |
| GET | `/models/{model_id}` | `model:read` (`version` to pin) |
| POST | `/models/refresh` | admin — re-scan `MODEL_DIR`, sync the catalogue |

`GET /models/{model_id}` returns the input/output spec, the license, the
author-reported metrics (empty unless supplied — the platform never estimates
one), the documented limitations, and for an unavailable model:

```json
"availability": {"available": false,
                 "reason": "No weights or manifest are installed …",
                 "missing": ["/app/models/oct/oct_retinal_layers_v1.json"],
                 "remediation": "Install a model for this task by placing …"}
```

## Reports

| Method | Path | Permission |
|---|---|---|
| POST | `/reports/{analysis_id}` | `report:write` — `{"format": "json"\|"html"\|"pdf"}` |
| GET | `/reports/{report_id}` | `report:read` |
| GET | `/reports/{report_id}/document` | `report:read` — rendered HTML/PDF |

The JSON payload is canonical; HTML and PDF render exactly that document, so a
printed report can never disagree with the API. Every report embeds the models
and versions used, their evidence level, their limitations and the versioned
disclaimer.

## Objects

`GET /objects/{key}` streams a stored mask, overlay or rendered report. With
local storage this is how derived images are served; with S3-compatible storage
the platform issues short-lived presigned URLs instead. Either way access
requires authentication and `analysis:read`.

## Health

`GET /health` reports database, storage, registry counts and the device
profile. It answers `degraded` rather than failing when a dependency is down,
so a load balancer can tell "process alive" from "fully serving".
