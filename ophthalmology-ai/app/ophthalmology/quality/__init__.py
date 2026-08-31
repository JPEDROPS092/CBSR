"""Quality control models.

Every modality has one built-in quality model. The pipeline runs it first and,
when ``QUALITY_GATE_ENABLED`` is set, refuses to run downstream models on an
image it rejects - an unreadable image produces a misleading score, not a
missing one, which is the more dangerous failure.
"""

from app.ophthalmology.quality.fundus_quality import FundusQualityModel
from app.ophthalmology.quality.oct_quality import OCTQualityModel

__all__ = ["FundusQualityModel", "OCTQualityModel"]
