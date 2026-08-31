"""The statement attached to every result the platform emits.

Kept in one place and versioned, so a report can record which wording it was
issued under.
"""

from __future__ import annotations

DISCLAIMER_VERSION = "1.0"

DISCLAIMER_EN = (
    "Research and decision-support tool. The outputs are model probabilities, scores, "
    "measurements and segmentations - not a medical diagnosis. They are not a "
    "substitute for examination and interpretation by a qualified eye-care "
    "professional. This software is not a cleared or certified medical device and "
    "must not be used as the sole basis for a clinical decision."
)

DISCLAIMER_PT = (
    "Ferramenta de pesquisa e apoio à decisão. Os resultados são probabilidades, "
    "escores, medidas e segmentações produzidos por modelos - não constituem "
    "diagnóstico médico. Não substituem o exame e a interpretação por profissional "
    "habilitado. Este software não é um dispositivo médico registrado ou certificado "
    "e não deve ser a única base de uma decisão clínica."
)


def disclaimer_block() -> dict[str, str]:
    """Both wordings plus the version, for embedding in a report."""
    return {"version": DISCLAIMER_VERSION, "en": DISCLAIMER_EN, "pt": DISCLAIMER_PT}
