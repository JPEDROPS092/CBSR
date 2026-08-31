# Development

## Setup

```bash
pip install -e ".[dev]"          # add ",ml" for torch/MONAI/ONNX adapters
export JWT_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"
export DATABASE_URL="sqlite+pysqlite:///./ophthalmology.db"
export STORAGE_BACKEND=local TASK_QUEUE_BACKEND=inline
alembic upgrade head
uvicorn app.main:app --reload
```

SQLite + local storage + inline queue runs the entire pipeline with no external
service. Postgres/Redis/MinIO are only needed to exercise the real deployment
path.

## Layout

```
app/api/             routes, dependencies, rate limiting        (HTTP only)
app/services/        use cases; own the database transaction
app/ai/              model interface, registry, pipeline, explainability
app/ophthalmology/   domain models and the task catalogue
app/database/        ORM entities + repositories
app/storage/         object storage backends
app/workers/         Celery app and task functions
app/core/            settings, logging, security, exceptions, enums
tests/{unit,ai,integration}
```

Dependencies point inward. In particular: **nothing outside `app/ai` imports a
specific model** — that rule is what keeps models pluggable.

## Tests

```bash
pytest                          # everything (~30 s)
pytest tests/unit               # pure functions: security, registry, image ops
pytest tests/ai                 # model behaviour on synthetic phantoms
pytest tests/integration        # API flows against SQLite + local storage
pytest --cov=app --cov-report=term-missing
```

Conventions:

* AI tests use **phantoms with known ground truth** (`tests/factories.py`), so
  they can assert that a measured thickness matches a constructed one. They
  never assert an accuracy the platform has not measured, and they never
  require a downloaded checkpoint.
* Behaviour under a *known defect* is the unit of test: a blurred phantom must
  be flagged and must score lower than the sharp one.
* Adapter tests for weight-backed models test the *contract* — that a missing
  checkpoint reports what to install, that a manifest configures preprocessing,
  that a bare state dict is refused rather than guessed at.
* `tests/conftest.py` sets the environment **before** importing any application
  module, because settings are cached in a singleton.

## Adding a model

Two paths.

**Installed (no code):** drop a checkpoint plus a manifest into `MODEL_DIR` and
call `POST /api/v1/models/refresh`. See [`MODEL_REGISTRY.md`](MODEL_REGISTRY.md).

**Built in (code):** subclass `BaseOphthalmologyModel` (or `ClassicalModel` for
a weight-free one):

```python
class MyModel(ClassicalModel):
    metadata = ModelMetadata(
        model_id="fundus_something_v1", name="…", version="1.0.0",
        modality=Modality.FUNDUS, task=TaskType.CLASSIFICATION,
        framework=Framework.CLASSICAL, evidence_level=EvidenceLevel.HEURISTIC,
        license=ModelLicense(name="Apache-2.0", commercial_use="allowed"),
        limitations="What a reader of this result must know.",
    )

    def predict(self, prepared):        # prepared is an ExamImage here
        return measure(prepared.pixels)

    def postprocess(self, output, image) -> ModelResult:
        return ModelResult(model_id=self.model_id, model_version=self.version,
                           task=self.metadata.task,
                           predictions=[PredictionItem(label="…", score=output)])
```

Register it in `app/ai/models/built_in_models()`. Nothing else changes — no
route, no schema, no migration.

Checklist for a new model:

- [ ] `metadata.limitations` written for whoever reads the result
- [ ] `evidence_level` honest (`heuristic` unless it is a trained model)
- [ ] licence and citation recorded
- [ ] `reported_metrics` only if someone actually measured them
- [ ] warnings emitted when inputs are missing (e.g. no pixel scale)
- [ ] a test asserting behaviour on a phantom or a known defect

## Conventions

* PEP 8, 100-column lines, `from __future__ import annotations`, type hints on
  every public function, docstrings that say *why* where the *what* is obvious.
* Errors: raise a typed `AppError` subclass; the API renders it. Never let an
  internal message reach the client.
* Logging: `logger.info("event_name", extra={...})` — event names are snake_case
  and stable, since they are queried in log tooling.
* No patient data in logs, error messages, storage keys or filenames.
* Enum columns go through `StrEnumType` so ORM attributes hold real enum
  members — a bare string breaks `is` comparisons silently.
* Services own the transaction boundary; repositories only query.

```bash
ruff check app tests && ruff format app tests
mypy app
```

## Debugging

| Symptom | Where to look |
|---|---|
| Model reports unavailable | `GET /models/{id}` → `availability.remediation` |
| Analysis stuck in `queued` | worker running? `TASK_QUEUE_BACKEND`? Flower |
| `skipped_quality` everywhere | quality thresholds vs your camera; `quality.metrics` in the run |
| Measurements in pixels | exam has no `acquisition_metadata.pixel_spacing_um` |
| Empty mask | check the run's `warnings` and the quality metrics |

Trace one request end to end by its `X-Request-ID`: the same id is attached to
every log line, including those emitted inside the inference pipeline.
