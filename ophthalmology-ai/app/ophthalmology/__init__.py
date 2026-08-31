"""Ophthalmology domain modules.

Layout::

    ophthalmology/
        catalog.py   the tasks the platform orchestrates
        quality/     modality-specific quality control (the pipeline gate)
        fundus/      colour fundus photography models
        oct/         optical coherence tomography models

Built-in models here are deterministic and weight-free. Learned models for the
same tasks are installed as checkpoints + manifests under ``MODEL_DIR`` and
picked up automatically by :func:`app.ai.models.bootstrap_registry`.
"""
