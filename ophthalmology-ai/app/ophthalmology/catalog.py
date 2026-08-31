"""The catalogue of ophthalmology tasks this platform orchestrates.

Each entry describes a task the platform knows how to *run* - its modality,
what it outputs, which labels are expected and what a reader must know about
its limits. Entries are bound to an implementation in one of two ways:

* **Built-in** - a deterministic model shipped in ``app/ophthalmology`` that
  needs no weights (quality control, OCT boundary detection, classical vessel
  segmentation).
* **Externally supplied** - the operator installs a checkpoint plus a manifest
  under ``MODEL_DIR``; the manifest states the preprocessing and labels of that
  specific checkpoint. Until then the entry is registered as a placeholder that
  reports exactly what is missing.

This is what keeps the platform an *orchestrator*: adding, replacing or
upgrading a model is a catalogue/manifest change, never an API change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.ai.results import InputSpec, ModelLicense, ModelMetadata, OutputSpec
from app.core.enums import EvidenceLevel, Framework, Modality, TaskType


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """A task in the catalogue that expects an externally supplied model."""

    model_id: str
    name: str
    modality: Modality
    task: TaskType
    subdir: str
    description: str
    expected_labels: list[str] = field(default_factory=list)
    expected_segmentation_classes: list[str] = field(default_factory=list)
    units: dict[str, str] = field(default_factory=dict)
    limitations: str = ""
    requires_quality_gate: bool = True

    @property
    def manifest_name(self) -> str:
        """Sidecar manifest filename expected under ``MODEL_DIR/<subdir>``."""
        return f"{self.model_id}.json"

    def to_metadata(self) -> ModelMetadata:
        """Metadata used while no implementation is installed."""
        return ModelMetadata(
            model_id=self.model_id,
            name=self.name,
            version="0.0.0",
            modality=self.modality,
            task=self.task,
            framework=Framework.PYTORCH,
            evidence_level=EvidenceLevel.RESEARCH,
            description=self.description,
            input_spec=InputSpec(
                notes=(
                    "Declared by the installed model's manifest. The platform never "
                    "assumes a preprocessing pipeline for third-party weights."
                )
            ),
            output_spec=OutputSpec(
                labels=self.expected_labels,
                segmentation_classes=self.expected_segmentation_classes,
                units=self.units,
                notes="Expected output vocabulary; the manifest declares the actual one.",
            ),
            license=ModelLicense(name="unknown", commercial_use="unknown"),
            limitations=self.limitations,
        )


#: ICDR severity scale, the vocabulary most public DR datasets are labelled in.
ICDR_GRADES = [
    "no_dr",
    "mild_npdr",
    "moderate_npdr",
    "severe_npdr",
    "proliferative_dr",
]


CATALOG: list[CatalogEntry] = [
    # -- Fundus ----------------------------------------------------------- #
    CatalogEntry(
        model_id="fundus_dr_grading_v1",
        name="Diabetic Retinopathy Grading",
        modality=Modality.FUNDUS,
        task=TaskType.CLASSIFICATION,
        subdir="fundus",
        description=(
            "Severity grading of diabetic retinopathy from a colour fundus photograph, "
            "reported as a probability per severity grade."
        ),
        expected_labels=ICDR_GRADES,
        limitations=(
            "Grading models are sensitive to camera, field definition and population. "
            "A grade is a screening signal for referral triage, never a diagnosis."
        ),
    ),
    CatalogEntry(
        model_id="fundus_glaucoma_v1",
        name="Glaucoma Suspicion",
        modality=Modality.FUNDUS,
        task=TaskType.CLASSIFICATION,
        subdir="fundus",
        description=(
            "Probability that a fundus photograph shows a glaucomatous optic neuropathy pattern."
        ),
        expected_labels=["normal", "glaucoma_suspect"],
        limitations=(
            "Photograph-based glaucoma models capture disc appearance only. They cannot "
            "replace intraocular pressure, visual field and OCT RNFL assessment."
        ),
    ),
    CatalogEntry(
        model_id="fundus_optic_disc_segmentation_v1",
        name="Optic Disc Segmentation",
        modality=Modality.FUNDUS,
        task=TaskType.SEGMENTATION,
        subdir="fundus",
        description="Segmentation of the optic disc boundary on a fundus photograph.",
        expected_segmentation_classes=["background", "optic_disc"],
        units={"disc_area_px": "pixel", "disc_area_mm2": "square millimetre"},
        limitations="Disc area in physical units requires a calibrated camera scale.",
    ),
    CatalogEntry(
        model_id="fundus_optic_cup_segmentation_v1",
        name="Optic Cup Segmentation",
        modality=Modality.FUNDUS,
        task=TaskType.SEGMENTATION,
        subdir="fundus",
        description=(
            "Segmentation of the optic cup, used with the disc mask to derive a "
            "vertical cup-to-disc ratio."
        ),
        expected_segmentation_classes=["background", "optic_cup"],
        units={"vertical_cup_to_disc_ratio": "ratio"},
        limitations=(
            "Cup boundaries are ambiguous even between human graders; cup-to-disc ratio "
            "from a photograph has wide inter-observer variability."
        ),
    ),
    CatalogEntry(
        model_id="fundus_vessel_segmentation_v1",
        name="Retinal Vessel Segmentation (learned)",
        modality=Modality.FUNDUS,
        task=TaskType.SEGMENTATION,
        subdir="fundus",
        description=(
            "Learned retinal vasculature segmentation; replaces the built-in classical "
            "vessel model where a trained checkpoint is available."
        ),
        expected_segmentation_classes=["background", "vessels"],
        limitations="Vessel density metrics are not comparable across models or resolutions.",
    ),
    CatalogEntry(
        model_id="fundus_macular_abnormality_v1",
        name="Macular Abnormality Detection",
        modality=Modality.FUNDUS,
        task=TaskType.CLASSIFICATION,
        subdir="fundus",
        description=(
            "Detection of macular abnormalities (drusen, exudates, haemorrhage, scar) "
            "on a colour fundus photograph."
        ),
        expected_labels=["normal", "drusen", "exudates", "haemorrhage", "scar"],
        limitations="Label vocabulary varies widely between datasets; check the manifest.",
    ),
    # -- OCT --------------------------------------------------------------- #
    CatalogEntry(
        model_id="oct_retinal_layers_v1",
        name="OCT Retinal Layer Segmentation",
        modality=Modality.OCT,
        task=TaskType.SEGMENTATION,
        subdir="oct",
        description=(
            "Multi-class segmentation of retinal layers on an OCT B-scan "
            "(e.g. RNFL, GCL+IPL, INL, OPL, ONL, RPE)."
        ),
        expected_segmentation_classes=[
            "background",
            "rnfl",
            "gcl_ipl",
            "inl",
            "opl",
            "onl",
            "is_os",
            "rpe",
        ],
        units={"layer_thickness_um": "micrometre"},
        limitations=(
            "Layer definitions differ between datasets and devices. Thickness values are "
            "only physically meaningful when the exam carries the device's axial scale."
        ),
    ),
    CatalogEntry(
        model_id="oct_fluid_segmentation_v1",
        name="OCT Fluid Segmentation",
        modality=Modality.OCT,
        task=TaskType.SEGMENTATION,
        subdir="oct",
        description=(
            "Segmentation of intraretinal fluid, subretinal fluid and pigment epithelial "
            "detachment on an OCT B-scan."
        ),
        expected_segmentation_classes=["background", "irf", "srf", "ped"],
        units={"fluid_area_mm2": "square millimetre"},
        limitations=(
            "Fluid volume requires a full B-scan series and the device's slice spacing; "
            "a single B-scan yields an area, not a volume."
        ),
    ),
    CatalogEntry(
        model_id="oct_biomarker_detection_v1",
        name="OCT Biomarker Detection",
        modality=Modality.OCT,
        task=TaskType.CLASSIFICATION,
        subdir="oct",
        description=(
            "Detection of OCT biomarkers such as drusen, hyperreflective foci, "
            "epiretinal membrane and ellipsoid-zone disruption."
        ),
        expected_labels=[
            "drusen",
            "hyperreflective_foci",
            "epiretinal_membrane",
            "ellipsoid_zone_disruption",
            "subretinal_hyperreflective_material",
        ],
        limitations="Multi-label output; each score is independent, they need not sum to 1.",
    ),
    CatalogEntry(
        model_id="oct_disease_classification_v1",
        name="OCT Disease Classification",
        modality=Modality.OCT,
        task=TaskType.CLASSIFICATION,
        subdir="oct",
        description=(
            "Classification of an OCT B-scan into common macular disease categories "
            "(e.g. CNV, DME, drusen, normal)."
        ),
        expected_labels=["normal", "cnv", "dme", "drusen"],
        limitations=(
            "Public OCT classification datasets are single-device and heavily curated; "
            "performance on other devices is typically much lower than reported."
        ),
    ),
    CatalogEntry(
        model_id="oct_glaucoma_analysis_v1",
        name="OCT Glaucoma Analysis",
        modality=Modality.OCT,
        task=TaskType.CLASSIFICATION,
        subdir="oct",
        description="Glaucoma-related assessment from OCT (RNFL/ONH-based).",
        expected_labels=["normal", "glaucoma_suspect"],
        limitations="Requires an RNFL or ONH scan pattern; not applicable to macular cubes.",
    ),
]


CATALOG_BY_ID: dict[str, CatalogEntry] = {entry.model_id: entry for entry in CATALOG}
