"""Optical coherence tomography models.

Built in: :class:`~app.ophthalmology.oct.layers_classical.OCTLayerBoundaryModel`
(ILM/RPE boundaries and retinal thickness).

Catalogued, externally supplied (see ``app/ophthalmology/catalog.py``):
multi-layer segmentation, fluid segmentation, biomarker detection, disease
classification and OCT-based glaucoma analysis.
"""

from app.ophthalmology.oct.layers_classical import OCTLayerBoundaryModel

__all__ = ["OCTLayerBoundaryModel"]
