"""Colour fundus photography models.

Built in: :class:`~app.ophthalmology.fundus.vessels_classical.RetinalVesselClassicalModel`.

Catalogued, externally supplied (see ``app/ophthalmology/catalog.py``):
diabetic retinopathy grading, glaucoma suspicion, optic disc and cup
segmentation, learned vessel segmentation, macular abnormality detection.
"""

from app.ophthalmology.fundus.vessels_classical import RetinalVesselClassicalModel

__all__ = ["RetinalVesselClassicalModel"]
