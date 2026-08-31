# Model registry

## The contract

The platform ships **no model weights**. It ships adapters, a catalogue of
tasks, and a rule:

> Input size, colour space, normalization, output labels and thresholds are
> properties of a *trained checkpoint*, not of this codebase. The platform
> never guesses them. A model whose preprocessing is not documented must not be
> run here — feeding a checkpoint the wrong normalization produces confident,
> wrong numbers, which is worse than no number at all.

So installing a model means supplying two files: the checkpoint and a **manifest**
that states exactly how it must be fed.

## What is registered at startup

| Kind | Source | Availability |
|---|---|---|
| Built-in | `app/ophthalmology/**` (quality, OCT boundaries, classical vessels) | always available |
| Installed | checkpoint + manifest under `MODEL_DIR` | available once the runtime and file are present |
| Placeholder | a catalogued task with nothing installed | never; reports what to install |

`GET /api/v1/models` lists all three. Placeholders are visible on purpose: the
catalogue is the platform's statement of what it orchestrates, and an operator
should be able to see the gap.

## Installing a model

```
models/
  fundus/
    dr_grading.onnx                 # the checkpoint
    fundus_dr_grading_v1.json       # its manifest
  oct/
    layers_unet.pt
    oct_retinal_layers_v1.json
```

```jsonc
{
  "model_id": "fundus_dr_grading_v1",     // catalogue id, or any new id
  "name": "Diabetic Retinopathy Grading (site model)",
  "version": "2.1.0",
  "modality": "fundus",                   // fundus | oct | other
  "task": "classification",               // classification | segmentation
  "framework": "onnx",                    // onnx | pytorch
  "weights_file": "dr_grading.onnx",      // relative to this manifest
  "weights_sha256": "…",                  // optional; verified before loading
  "evidence_level": "research",           // heuristic | research | clinical_validated
  "input": {
    "image_size": [512, 512],
    "channels": 3,
    "color_space": "rgb",                 // rgb | bgr | gray
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std": [0.229, 0.224, 0.225],
    "scale_to_unit_interval": true
  },
  "output": {
    "labels": ["no_dr", "mild_npdr", "moderate_npdr", "severe_npdr", "proliferative_dr"],
    "activation": "softmax",              // softmax | sigmoid | none
    "threshold": 0.5                      // segmentation only
  },
  "license": {
    "name": "CC BY-NC 4.0",
    "url": "…", "source_url": "…",
    "dataset": "EyePACS", "dataset_license": "…",
    "commercial_use": "prohibited",       // allowed | restricted | prohibited | unknown
    "citation": "…", "restrictions": "…"
  },
  "reported_metrics": {"quadratic_weighted_kappa": 0.82},
  "limitations": "45-degree macula-centred images only.",
  "supports_explainability": false
}
```

For a segmentation model use `"output": {"segmentation_classes": ["background",
"irf", "srf", "ped"], "activation": "softmax"}`; class index 0 is background by
convention and is not emitted as a mask.

Then:

```bash
curl -X POST /api/v1/models/refresh -H "Authorization: Bearer <admin token>"
# or, offline:  python scripts/sync_models.py
```

`reported_metrics` is copied verbatim from the model's own documentation. The
platform never fills it in, and an empty object means nobody has supplied a
measurement — not that the model is untested.

## Checkpoint formats

| Format | Behaviour |
|---|---|
| `.onnx` | **preferred** — graph and weights travel together, runs on CPU and CUDA |
| TorchScript `.pt` | loaded directly with `torch.jit.load`; architecture included |
| state dict `.pt/.pth` | requires an adapter that defines the architecture |

A bare state dict is refused with an explanation rather than loaded into a
guessed architecture. Either export TorchScript:

```python
torch.jit.save(torch.jit.script(model.eval()), "layers_unet.pt")
```

or write a small adapter overriding `build_module()`:

```python
class MyUNetAdapter(TorchSegmentationModel):
    def build_module(self):
        from monai.networks.nets import UNet
        return UNet(spatial_dims=2, in_channels=1, out_channels=8,
                    channels=(32, 64, 128, 256), strides=(2, 2, 2), num_res_units=2)
```

## Versioning and reproducibility

Several versions of one `model_id` can be registered at once.
`get("dr_grading")` resolves to the newest; `"dr_grading@2.1.0"` pins one.
Stored analyses always record the exact version that ran, so upgrading a model
never rewrites the meaning of an old result.

Each `ModelRun` row records `model_id`, `model_version`, `input_hash` (SHA-256
of the exact pixels fed to the model), `device`, `precision`, `batch_size`,
`processing_time_ms` and `software_version`. `ModelRecord` keeps a durable
snapshot of the registry metadata a run referenced.

## Quality gate

A model with `task = quality` runs first for its modality. If it rejects the
image and `QUALITY_GATE_ENABLED` is on, downstream models are recorded as
`skipped_quality` rather than run. Override per analysis with
`{"quality_gate": false}`, or replace the built-in gate by registering another
`task = quality` model for that modality.

The built-in thresholds (`app/ophthalmology/quality/*.py`, `THRESHOLDS`) are
defaults measured on 8-bit images. **Calibrate them against your own cameras**
before relying on them: collect a set of images your graders labelled
acceptable/unacceptable, run `GET /models` metrics on them, and adjust.

## The catalogue

| Model id | Modality | Task |
|---|---|---|
| `fundus_quality_v1` * | fundus | quality |
| `fundus_vessels_classical_v1` * | fundus | segmentation |
| `fundus_dr_grading_v1` | fundus | classification |
| `fundus_glaucoma_v1` | fundus | classification |
| `fundus_optic_disc_segmentation_v1` | fundus | segmentation |
| `fundus_optic_cup_segmentation_v1` | fundus | segmentation |
| `fundus_vessel_segmentation_v1` | fundus | segmentation |
| `fundus_macular_abnormality_v1` | fundus | classification |
| `oct_quality_v1` * | OCT | quality |
| `oct_layers_classical_v1` * | OCT | segmentation |
| `oct_retinal_layers_v1` | OCT | segmentation |
| `oct_fluid_segmentation_v1` | OCT | segmentation |
| `oct_biomarker_detection_v1` | OCT | classification |
| `oct_disease_classification_v1` | OCT | classification |
| `oct_glaucoma_analysis_v1` | OCT | classification |

`*` built in, no weights needed. Entries are defined in
`app/ophthalmology/catalog.py`; adding one there is a few lines and requires no
API change.

## Where to find models

MONAI Model Zoo, Hugging Face, the repositories of published papers, and your
own training runs. Before using one, read [`MODEL_LICENSES.md`](MODEL_LICENSES.md):
many ophthalmology datasets and the weights trained on them are
research-only, and a manifest's `license.commercial_use` must reflect that.
