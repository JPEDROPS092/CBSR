# `MODEL_DIR`

Checkpoints and their manifests live here. **Nothing in this directory is
committed** — weights are large and frequently carry licences that forbid
redistribution.

```
models/
  fundus/
    dr_grading.onnx
    fundus_dr_grading_v1.json
  oct/
    layers_unet.pt
    oct_retinal_layers_v1.json
  segmentation/
```

Every `*.json` file under this tree is read as a model manifest at startup and
on `POST /api/v1/models/refresh`. A file that is not a valid manifest is logged
and skipped — one bad file never stops the platform from starting.

Start from `manifest.json.example` (the `.example` suffix keeps it out of the
scan) and read [`../docs/MODEL_REGISTRY.md`](../docs/MODEL_REGISTRY.md) for the
full schema, the accepted checkpoint formats, and why the manifest is mandatory.
