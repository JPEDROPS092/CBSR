# Model and dataset licenses

**No model weights are distributed with this repository.** This file is the
register an operator must fill in for every model they install, and the place
to record the licence conditions that follow those weights into production.

## Why this file exists

Ophthalmology models are usually trained on datasets whose licence restricts
use — frequently to non-commercial research. That restriction travels with the
weights: a model trained on a research-only dataset generally cannot be used in
a commercial service, no matter how the checkpoint itself is labelled. Assume
nothing is permitted until the licence says it is.

## Register (fill in as you install)

| Model | Version | Source | Weights licence | Dataset | Dataset licence | Commercial use | Citation | Restrictions |
|---|---|---|---|---|---|---|---|---|
| _example_ `fundus_dr_grading_v1` | 2.1.0 | `https://…` | CC BY-NC 4.0 | EyePACS | see source | **prohibited** | Author et al., 2021 | research use only |
| | | | | | | | | |

The same fields live in each model's manifest under `license`, and are surfaced
by `GET /api/v1/models/{model_id}` — so the constraint is visible to whoever
reads a result, not only to whoever installed the model.

## Models shipped with the platform

These are deterministic image-processing algorithms written for this project.
They carry no third-party weights and no dataset obligations.

| Model | Licence | Commercial use | Method reference |
|---|---|---|---|
| `fundus_quality_v1` | Apache-2.0 | allowed | Variance of Laplacian (Pech-Pacheco et al., ICPR 2000); colourfulness (Hasler & Süsstrunk, SPIE 2003) |
| `oct_quality_v1` | Apache-2.0 | allowed | Intensity/SNR profile analysis |
| `oct_layers_classical_v1` | Apache-2.0 | allowed | Intensity/gradient boundary detection, in the spirit of Chiu et al., Optics Express 2010 |
| `fundus_vessels_classical_v1` | Apache-2.0 | allowed | Morphological vessel enhancement (Zana & Klein, IEEE TIP 2001) |

They are **heuristics**: engineering signals for acquisition QA and triage, not
clinical evidence, and not validated against any reference standard.

## Checklist before installing a model

1. Record the licence of the **weights** and of the **training dataset** — they
   are often different, and the stricter one governs.
2. Check whether the licence permits your deployment (commercial? clinical?
   redistribution? derivative works?).
3. Copy the required attribution/citation into the manifest.
4. Record the documented limitations (population, camera, field, resolution) in
   `limitations`; they are printed on every report.
5. Copy the author-reported metrics verbatim into `reported_metrics`, with the
   dataset they were measured on. Do not transfer a metric measured elsewhere to
   your own population — publish your own evaluation instead.
6. Keep the checkpoint's SHA-256 in the manifest so a swapped file is detected.

## Commonly used ophthalmology datasets

Verify current terms at the source before use; several have changed licence.

| Dataset | Typical use | Note |
|---|---|---|
| EyePACS / APTOS | DR grading | competition terms; typically research-only |
| Messidor / Messidor-2 | DR grading | registration and terms required |
| DRIVE, STARE, CHASE_DB1, HRF | vessel segmentation | research use, attribution required |
| REFUGE, ORIGA, RIM-ONE | disc/cup, glaucoma | per-dataset terms; several research-only |
| IDRiD | DR, lesions, disc | research use |
| Kermany OCT (CNV/DME/drusen/normal) | OCT classification | CC BY 4.0 at publication; single-device |
| RETOUCH, OCT5K, Duke SD-OCT | OCT fluid / layers | challenge or academic terms |

Nothing in this table is legal advice, and none of it is a substitute for
reading the licence that comes with the files you install.
