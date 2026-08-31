"""Asynchronous processing.

``TASK_QUEUE_BACKEND=inline`` runs analyses in-process (development, tests);
``celery`` dispatches them to a worker over Redis. Both call the same functions
in :mod:`app.workers.tasks`.
"""
