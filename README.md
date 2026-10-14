# PET-CT Radiomics Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-available-blue.svg)](https://www.docker.com/)

Automated PET radiomics extraction from integrated PET/CT studies using
TotalSegmentator-derived CT masks.

## What This Repository Implements

The base workflow performs the following steps:

1. Select volumetric CT and PET DICOM series automatically.
2. Convert DICOM to NIfTI with `dicom2nifti`.
3. Convert native-grid PET voxel values to SUVbw using PET DICOM metadata.
4. Generate CT masks with TotalSegmentator.
5. Resample the CT-derived masks to the native PET grid with nearest-neighbor
   interpolation.
6. Extract PET first-order and texture features on the native PET grid, while
   retaining shape features from the original CT-grid masks.
7. Create a separate CT-grid PET SUV image for fusion QC.
8. Optionally use `--adapt-diaphragm` to locally extend or retract the
   diaphragmatic end of lung and adjacent liver/spleen masks in the SI direction,
   retaining original outputs and reporting paired native-PET features.

The evaluated source is preserved at tag `anm-evaluated-2026-10-14`.
Release `anm-release-2026-10-14.1` updates distribution documentation and test
configuration without changing the evaluated processing code.
The adaptation comparison uses seven target labels (liver, spleen and five lung lobes),
independently of the base workflow's eight-organ default.

The adaptation stage is experimental and disabled by default. It does not move
PET intensities, perform a whole-organ translation, or establish anatomical
accuracy. See [adaptation inputs, outputs and safeguards](docs/pet_diaphragm_adaptation.md).

Important clarification:

- In the manuscript, reproducibility refers to standardized computational
  execution with fixed processing parameters, not to test-retest repeatability
  of individual radiomic features.
- This is not de novo PET/CT registration.
- PET is assumed to come from an integrated PET/CT acquisition.
- PET intensities are not upsampled to the CT grid for radiomics extraction.
- No additional image resampling is applied inside PyRadiomics.
- `dicom2nifti` applies the DICOM modality rescale while creating the PET
  NIfTI image. For BQML data, the resulting NIfTI values are therefore treated
  directly as activity concentration; Rescale Slope and Intercept are not
  applied a second time during SUV conversion.

## Scope

- Default output: PET radiomics rows for a representative 8-organ set
- Supported mask source: TotalSegmentator CT segmentation
- Available structures: all labels emitted by the configured TotalSegmentator
  version and task (v2.11.0 ``total`` emitted 117 in the validation run)
- Feature families: first-order, shape, GLCM, GLRLM, GLSZM, GLDM, NGTDM

By default, the repository does not export a separate CT radiomics table.
Shape descriptors are still included because they are derived from the
CT-based segmentation mask geometry.

## Quick Start

```bash
git clone https://github.com/haga-aki/pet-ct-radiomics-pipeline-public.git
cd pet-ct-radiomics-pipeline-public
conda env create -f environment.yml
conda activate pet_radiomics
cp config.yaml.example config.yaml
python run_pipeline.py --input /path/to/PETCT_DICOM --output /path/to/output
```

Main output:

- `/path/to/output/radiomics_results.csv`
- `/path/to/output/qc/<run_id>/nifti_images/`
- `/path/to/output/qc/<run_id>/segmentations/`
- `/path/to/output/qc/<run_id>/visualizations/`

## Representative Default Organs

The default configuration targets the following organs:

- liver
- spleen
- kidney_left
- kidney_right
- adrenal_gland_left
- adrenal_gland_right
- aorta
- vertebrae_L1

This is the base workflow's default subset. To process all
available structures, set:

```yaml
organs:
  - all
```

## PyRadiomics Settings

PET extraction uses [`params.yaml`](params.yaml). The manuscript-aligned
defaults are:

- `binWidth = 0.25`
- `resampledPixelSpacing = null`
- `normalize = false`
- `force2D = false`
- `minimumROISize = 50`

Interpretation:

- CT-derived masks are resampled to the native PET grid with nearest-neighbor
  interpolation for PET intensity and texture extraction.
- `resampledPixelSpacing = null` means PyRadiomics does not resample the PET
  image.
- Shape features are calculated separately from the CT-grid masks and merged
  into the organ-level PET output row.
- Small ROIs below 50 voxels are skipped rather than exported as placeholder
  rows. This is a computational inclusion threshold, not a guarantee of
  texture-feature reliability.

The feature set is described using IBSI terminology, but this repository does
not claim formal independent IBSI certification.

The validated environment used PyRadiomics
`3.1.1.dev111+g8ed579383`, pinned to upstream commit
`8ed579383b44806651c463d5e691f3b2b57522ab` in both requirements files and
the Docker image. The exact revision is retained to reproduce the evaluated
implementation; it should not be replaced by an unpinned release range when
reproducing the manuscript results.

## Configuration

Copy [`config.yaml.example`](config.yaml.example) to `config.yaml` and adjust:

```yaml
modalities:
  - CT
  - PET

organs:
  - liver
  - spleen
  - kidney_left
  - kidney_right
  - adrenal_gland_left
  - adrenal_gland_right
  - aorta
  - vertebrae_L1

radiomics:
  params_file: params.yaml
  extract_ct: false
```

Notes:

- `CT` remains necessary because TotalSegmentator runs on CT.
- `extract_ct: false` preserves the manuscript-style PET-focused output.

Full configuration details are documented in
[`docs/config_reference.md`](docs/config_reference.md).

## Alternative Entry Points

- `python run_full_analysis.py`
  Runs `run_pipeline.py` and then generates QC overlays.
- `python create_final_suv.py`
  Compatibility utility that rebuilds PET SUV images and PET rows from existing
  outputs. This is mainly for older workflows; `run_pipeline.py` already does
  SUV conversion internally.
- `python gui_launcher.py`
  Starts the GUI launcher.

## Technical QC reports

Each execution writes `qc/<run_id>/case_qc.csv`, `organ_qc.csv`, and
`stage_qc.csv` below the output directory. These files are checkpointed as
processing proceeds, including when no radiomics rows are extracted.

- Organ records distinguish missing files, empty masks, masks empty after PET
  transfer, small ROIs, geometry mismatches, transfer/extraction failures,
  non-finite features, and extracted rows. With an explicit organ list,
  upstream failures also produce `not_attempted` records.
- Geometry checks compare size, spacing, origin, and direction for the source
  CT mask and CT reference, then the transferred mask and native PET reference.
  Numeric tolerance is 1e-5 (mm for spacing/origin; unitless for direction).
  These checks do not establish anatomical alignment.
- Stage records include series selection, conversion, segmentation, SUV
  conversion, the CT-grid display copy, and fusion-image generation.
- Case summaries use `checks_completed`, `attention_required`, or `incomplete`.
  Visual review remains `pending` and anatomical acceptance `not_assessed`;
  successful processing never automatically certifies segmentation accuracy.

Empty labels require an applicability decision; no universal success percentage
is computed. With `organs: [all]`, the inventory is the files actually emitted;
the program cannot infer an absent expected label without an explicit list.
Small-ROI screening is a computational threshold, not a texture-reliability test.
Hepatic-dome profile analysis and automatic respiratory-mismatch or segmentation
failure detection are not part of this standard QC implementation.

`qc/<run_id>/radiomics_results.csv` contains only the current execution. When an
aggregate already exists, it is backed up to `previous_radiomics_results.csv`
in this run directory before replacement. Reprocessed cases replace their old
aggregate rows, including organs now skipped or failed; unrelated cases remain.
Fusion records refer only to images generated from this execution's SUV display
copy. Historical image files may remain on disk; consult the current stage report.
An interrupted run retains its latest QC checkpoint with incomplete case status.

## Output Table

`radiomics_results.csv` contains one row per `(PatientID, Modality, Organ)`.

Typical columns include:

- `PatientID`
- `Modality`
- `Organ`
- `original_firstorder_Mean`
- `original_firstorder_Maximum`
- `original_shape_VoxelVolume`
- `original_glcm_*`
- `original_glrlm_*`
- `original_glszm_*`
- `original_gldm_*`
- `original_ngtdm_*`

## Quality Control

QC overlays are generated with [`visualize_mask_verification.py`](visualize_mask_verification.py).
These figures show PET/CT fusion images with representative mask overlays for
visual inspection of mask placement and PET/CT alignment.

## Data Handling

This repository is intended for code and documentation only.

- Raw DICOM data should remain outside version control.
- Anonymized ID mapping files are ignored by `.gitignore`.
- De-identified example data are not distributed in this repository and may be
  requested from the corresponding author.
- If you publish derived CSV files, ensure they do not contain patient-linked
  identifiers or institution-restricted data.

## Repository Layout

```text
pet-ct-radiomics-pipeline/
├── run_pipeline.py
├── run_full_analysis.py
├── create_final_suv.py
├── visualize_mask_verification.py
├── suv_converter.py
├── config.yaml.example
├── params.yaml
├── docs/
├── raw_download/        # ignored
├── nifti_images/        # ignored
├── segmentations/       # ignored
└── visualizations/      # ignored
```

## References

- Wasserthal J, et al. TotalSegmentator: robust segmentation of 104 anatomic
  structures in CT images. Radiology: Artificial Intelligence. 2023.
- van Griethuysen JJM, et al. Computational radiomics system to decode the
  radiographic phenotype. Cancer Research. 2017.
- Zwanenburg A, et al. The Image Biomarker Standardization Initiative. Radiology.
  2020.
- Pfaehler E, et al. Repeatability of [18F]FDG PET radiomic features. Medical
  Physics. 2019.
- Boellaard R. FDG PET/CT: EANM procedure guidelines for tumour imaging.
  Eur J Nucl Med Mol Imaging. 2015.

## License

MIT License.

## Safety and intermediate outputs

Each invocation writes fresh NIfTI images, masks and fusion PNGs under
`qc/<run_id>/`. Historical intermediates are retained, not implicitly reused.
This intentionally reruns segmentation and increases storage/runtime; directory
existence is not evidence of a completed or input-matched computation. The
aggregate CSV retains the existing backup-and-replace policy for current cases.
New result rows include `RunID`, `PipelineSHA256`, `ConfigurationSHA256` and
`ParametersSHA256`; runtime package versions are saved alongside each run.
Legacy aggregate rows without provenance remain unidentified, not revalidated.
The aggregate may mix executions: use `qc/<run_id>/radiomics_results.csv` from
a single full-cohort execution for a study, not the convenience aggregate.

With `organs: [all]`, expected mask names come from the installed
TotalSegmentator task map, restricted by `roi_subset` when configured. Missing
files are reported separately from present but empty masks. Empty structures
require applicability review and are not automatically anatomical failures.
Unreadable, non-binary or geometrically inconsistent masks fail the combined
mask stage rather than being silently omitted.

Standard BQML SUV conversion supports `DecayCorrection=START` (acquisition
reference) and `ADMIN` (no dose decay). Missing/invalid timing, NONE/unknown decay
modes and inconsistent series metadata stop SUV extraction. SeriesTime is not
assumed to be AcquisitionTime. Complete injection/acquisition datetimes support
dated midnight crossings. Injection time without a date is rejected by default.
Only after independently confirming same-day administration for every input,
set `suv: {allow_same_day_injection: true}` in the configuration (recorded by
the configuration hash). Negative intervals still stop processing; StudyDate
alone does not establish the injection date. Series with varying acquisition timing
require a validated vendor-specific adapter; this conservative implementation
does not infer a reference from a random slice. Private-tag scaling of already
rescaled NIfTI is likewise rejected until a vendor adapter is validated.
These are explicit support limits, not claims of universal PET vendor support.

Timing definitions: [DICOM PS3.3 C.8.9](https://dicom.nema.org/medical/dicom/2024a/output/chtml/part03/sect_C.8.9.html).
Technical QC and passing unit tests do not establish clinical accuracy or
anatomical acceptance; visual review remains required.
