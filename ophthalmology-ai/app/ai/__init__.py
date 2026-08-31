"""AI platform core: registry, model interface, inference pipeline.

The rule that keeps this platform an orchestrator rather than a wrapper around
one model: **nothing outside this package may import a specific model.** The
API, services and workers talk to :class:`~app.ai.registry.ModelRegistry` and
consume :class:`~app.ai.results.ModelResult`.
"""
