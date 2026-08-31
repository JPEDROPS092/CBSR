# CPU image for the API and for workers that run the built-in models.
# For GPU inference use Dockerfile.gpu.
FROM python:3.11-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libGL/libglib: OpenCV runtime. curl: container healthcheck.
RUN apt-get update && apt-get install --no-install-recommends -y \
        curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first so application edits do not invalidate the layer.
COPY pyproject.toml README.md ./
RUN pip install --upgrade pip && pip install "."

COPY app ./app
COPY migrations ./migrations
COPY scripts ./scripts
COPY alembic.ini ./

# Run as a non-root user: this process handles medical images.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models /app/storage-data \
    && chown -R appuser:appuser /app
USER appuser

ENV MODEL_DIR=/app/models
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
