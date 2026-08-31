# Architecture

## The rule everything follows

**Nothing outside `app/ai` may import a specific model.** Routes, services and
the database talk to two things only: the `ModelRegistry` and the standard
`ModelResult`. That is what makes the platform an orchestrator rather than a
wrapper around one network — adding, replacing or version-bumping a model
touches no route, no schema, no migration and no frontend code.

```
                         AI PLATFORM
                              │
              ┌───────────────┼───────────────┐
             OCT            FUNDUS          OTHER
              │               │
    ┌─────────┼─────────┐  ┌──┼─────────┐
  layers    fluid   biomarkers │      glaucoma
    │         │        │      DR         │
    └─────────┴────────┴──────┴──────────┘
                       │
              Inference Engine        ← quality gate, ordering, isolation
                       │
               Result Schema          ← one shape for every model
                       │
                Report Engine         ← JSON / HTML / PDF
```

## Layers

```
app/
  api/         HTTP: routes, dependencies, authN/Z, rate limiting
  services/    use cases: patients, exams, analyses, reports, audit
  ai/          model interface, registry, inference pipeline, explainability
  ophthalmology/  domain models (quality, fundus, OCT) + the task catalogue
  database/    ORM entities and repositories
  storage/     object storage (local / S3-compatible)
  workers/     Celery app and task functions
  core/        settings, logging, security, exceptions, enums, disclaimer
```

Dependencies point inward: `api → services → {ai, database, storage}`. The AI
layer knows nothing about HTTP, the database or object storage — it takes
decoded pixels and returns results in memory. Persisting masks and rows is the
service layer's job, which is why models are testable in isolation and reusable
from a script or a notebook.

## Request flow

```
HTTP request
  │  RequestContextMiddleware      assigns X-Request-ID, times, logs
  │  rate limiter                  per-identity quota
  │  HTTPBearer + get_current_user JWT → User
  │  require_permission(...)       RBAC
  ▼
route (thin)  →  service (use case, owns the transaction)
                   ├── repository → PostgreSQL
                   ├── object storage → S3 / MinIO / disk
                   └── task queue → Celery (or inline)
```

Errors travel as typed `AppError`s and are rendered into one envelope:

```json
{"error": {"code": "model_unavailable", "message": "...", "details": {...}}}
```

## Inference flow

```
POST /analysis
  ├── resolve requested models against the registry   (unknown id → 404 now,
  │                                                    not a failed job later)
  ├── persist Analysis(status=queued) and COMMIT
  └── enqueue → Celery worker (or run inline)
                    │
                    ▼
        AnalysisService.execute
          status=processing
          load image bytes from object storage → decode → ExamImage
          InferenceEngine.run:
            quality models first  ──► gate: fail ⇒ downstream = skipped_quality
            for each model:
              availability?  no  ⇒ skipped_unavailable (+ remediation)
              run()  →  preprocess → predict → postprocess → explain
              raises ⇒ this run is failed; the others still run
          persist: ModelRun, Prediction, Segmentation (mask → storage), Artifact
          status=completed | failed, finished_at
```

Every `ModelRun` row records `model_id`, `model_version`, `input_hash`,
`device`, `precision`, `batch_size`, `processing_time_ms` and
`software_version` — enough to reproduce and audit an inference after the fact.

## Data model

```
User ─┐
      ├─ audit_logs (append-only)
Patient ── Exam ── Image ── Analysis ── ModelRun ── Prediction
                              │            ├─ Segmentation → mask in object storage
                              │            └─ Artifact     → Grad-CAM / overlay
                              └─ Report
ModelRecord (versioned registry snapshot, referenced by ModelRun)
```

Pixel data never enters PostgreSQL; the database holds a `(bucket, key)`
pointer. Storage keys are built from UUIDs and never from patient identifiers
or uploaded filenames.

## The plugin system

A model is a class implementing `BaseOphthalmologyModel`:

```python
class MyModel(BaseOphthalmologyModel):
    metadata = ModelMetadata(...)      # id, version, modality, task, license, limits
    def _load(self): ...               # optional: materialize weights
    def preprocess(self, image): ...
    def predict(self, prepared): ...
    def postprocess(self, output, image) -> ModelResult: ...
    def explain(self, image, output) -> list[ArtifactPayload]: ...
```

`run()` is a template method that sequences those steps, times them, records the
execution context and attaches the model's documented limitations to the result.

Three ways a model reaches the registry:

1. **Built in** — registered by `app.ai.models.bootstrap_registry`.
2. **Installed** — a checkpoint plus a JSON manifest under `MODEL_DIR`; the
   generic Torch/ONNX adapters are configured entirely from that manifest.
3. **Placeholder** — a catalogued task with nothing installed, which reports
   what is missing instead of guessing.

Availability is a live probe (`availability()`), not a stored flag: a missing
checkpoint, a missing runtime or a checksum mismatch is reported with the exact
remediation.

## Asynchronous execution

`TASK_QUEUE_BACKEND` selects the backend behind one interface:

* `inline` — runs in-process. Development and tests; no broker needed.
* `celery` — Redis broker, worker with `-c 1` per GPU.

Both call the same `app.workers.tasks.execute_analysis`, so what runs in tests
is what runs in production.

## Device abstraction

`DeviceManager` resolves `auto | cpu | cuda`, validates the precision request
(downgrading `fp16`/`bf16` where unsupported rather than failing mid-inference)
and reports VRAM. Model adapters never call `torch.cuda` directly, which is why
the same adapter runs under PyTorch on a GPU, ONNX Runtime on a CPU, or not at
all — reporting that its runtime is missing.
