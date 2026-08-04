# Exception Handling Specification

This document describes the behavior implemented in the current public
pipeline. It intentionally reflects the code as shipped, not aspirational
future behavior.

## ROI Size Handling

For PET radiomics, `minimumROISize=50` is read from [`params.yaml`](../params.yaml).

Behavior:

- Empty masks are skipped with a warning in the console log.
- Masks with fewer than 50 voxels are skipped with a warning in the console log.
- Skipped organs do not produce output rows.

Rationale:

- The threshold is a computational inclusion rule, not evidence that texture
  estimates above the threshold are reliable.
- Texture results from small eligible ROIs remain exploratory.

## Empty or Missing Masks

Behavior:

- If a requested organ mask file is absent, the organ is skipped.
- If the mask exists but contains no foreground voxels, the organ is skipped.

Common causes:

- Organ is outside the field of view.
- Segmentation was not produced for that structure.
- The organ list requests a label that is not present in the segmentation folder.

## Feature Computation Failures

Behavior:

- If PyRadiomics raises an exception for a given organ, the error is logged.
- The pipeline continues to the next organ instead of aborting the case.
- No placeholder row is emitted for the failed organ.

## PET Alignment and SUV Conversion

Behavior:

- Native-grid PET is converted to SUVbw using metadata from the original PET
  DICOM series.
- CT-derived masks are resampled to the native PET grid with nearest-neighbor
  interpolation for first-order and texture extraction.
- Shape features are calculated from the original CT-grid masks.
- A linearly resampled CT-grid PET SUV copy is generated only for fusion QC.
- If mask transfer fails, the affected organ is skipped and logged.
- If SUV conversion fails because required DICOM metadata are unavailable, PET extraction for that case is skipped.

Notes:

- This is not de novo multimodal registration; it assumes integrated PET/CT data.
- No PET image resampling is applied inside PyRadiomics.

## Output Behavior

- The main output is `radiomics_results.csv`.
- Duplicate rows are replaced using the key `(PatientID, Modality, Organ)`.
- CT radiomics rows are disabled by default in `config.yaml.example`.
- QC figures are generated separately under `visualizations/`.

## Logging

The pipeline currently logs to standard output.

- `INFO`: normal processing messages
- `WARNING`: recoverable issues such as empty masks or small ROIs
- `ERROR`: case-level failures such as failed conversion or failed feature extraction
