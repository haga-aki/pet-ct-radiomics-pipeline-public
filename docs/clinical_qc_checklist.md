# Clinical Quality Control Checklist

This document provides guidance for validating pipeline outputs before using radiomics features in research or clinical applications.

## Pre-processing Checks

### 1. DICOM Integrity

- [ ] All DICOM files are readable
- [ ] Patient ID is correctly anonymized
- [ ] Modality tags are correct (CT, PT)
- [ ] Series descriptions match expected protocol

### 2. SUV Conversion Verification

The pipeline converts native PET to SUVbw only from standard DICOM activity
concentration (`Units = BQML`) and rejects ambiguous timing metadata. Check:

- [ ] `stage_qc.csv` reports `suv_conversion` as completed for every PET series
- [ ] Body weight, injected dose, radionuclide half-life, and injection and
      acquisition times are present and plausible in the source DICOM
- [ ] No negative or non-finite SUV values in the converted image
- [ ] Reference-region values (for example, central liver or blood pool) look
      plausible on visual review for the tracer and protocol in use

This document does not define numerical normal or reference ranges. Unusual
values should prompt review of metadata and images, not automatic exclusion.

---

## Segmentation Quality

### Visual Inspection Points

For each organ, verify in the QC visualization:

| Organ | Common Issues | What to Check |
|-------|---------------|---------------|
| Liver | Partial inclusion, excludes lesions | Complete coverage, no lung/heart overlap |
| Spleen | Confused with kidney/stomach | Correct anatomical location |
| Kidneys | Left/right swapped, includes adrenal | Bilateral symmetry, exclude pelvis |
| Adrenal glands | Too small, merged with kidney | Visible as separate structure |
| Aorta | Includes heart, truncated | Continuous from arch to bifurcation |
| Vertebrae | Wrong level labeled | Check against anatomical landmarks |

### Registration Accuracy

**Signs of registration failure:**
- Organ boundaries don't align between CT and PET
- SUV values appear shifted from anatomy
- Unusual patterns at organ edges

**Common causes:**
- Patient motion between scans
- Respiratory phase mismatch
- Metal artifact distortion

---

## Radiomics Feature Validation

### Plausibility Review

No numerical normal ranges are provided. Feature values depend on tracer,
scanner, reconstruction, uptake time, discretization, and the anatomical
definition of each mask (for example, whole-kidney masks include
collecting-system activity). Review instead:

- [ ] Values are consistent with the visual fusion image for the same organ
- [ ] Organs adjacent to the diaphragm, small organs, and organs with excreted
      tracer are interpreted with their known limitations
- [ ] Values that differ markedly from other cases in the same cohort are
      checked against images and metadata before analysis

### Red Flags

**Immediate investigation required:**
- Negative SUV values
- Volume = 0 or extremely small
- NaN or Inf in any feature (reported as `nonfinite_features` in `organ_qc.csv`)
- Entropy = 0 (uniform region)
- `geometry_mismatch` or `source_geometry_mismatch` in `organ_qc.csv`

**Review recommended:**
- High SUVmax that may reflect a lesion, excreted tracer, or artifact
- Volume or shape inconsistent with the visible anatomy

---

## Batch Processing QC

### Statistical Outlier Detection

For each feature across patients:
1. Calculate median and IQR
2. Flag values outside 1.5 × IQR
3. Review flagged cases visually

### Consistency Checks

- [ ] Same number of features for all patients
- [ ] Same organs extracted for all patients
- [ ] No missing values (unless organ not present)
- [ ] Feature distributions match expected patterns

---

## Common Artifacts

### PET-specific

| Artifact | Appearance | Impact on Radiomics |
|----------|------------|---------------------|
| Metal artifact | Streaks, dark bands | Corrupted texture features |
| Respiratory motion | Blurring at diaphragm | Increased entropy, reduced contrast |
| Attenuation overcorrection | Bright rim at edges | Elevated edge features |
| Reconstruction artifact | Structured noise | Texture feature artifacts |

### CT-specific

| Artifact | Appearance | Impact on Segmentation |
|----------|------------|------------------------|
| Metal implant | Star pattern | Incorrect boundaries |
| Motion blur | Blurred edges | Over/under-segmentation |
| Beam hardening | Dark bands | Incorrect HU values |
| Truncation | Cut-off edges | Incomplete organ |

---

## Documentation Requirements

For each processed case, record:

1. **Patient metadata**
   - Anonymized ID
   - Scan date
   - Scanner vendor/model

2. **Processing parameters**
   - Pipeline version
   - Configuration used
   - Processing timestamp

3. **QC results**
   - Visual inspection passed/failed
   - Any organs excluded and why
   - Notes on artifacts or issues

4. **Output validation**
   - Number of features extracted
   - Any missing organs
   - Flagged outlier values

---

## Failure Modes and Solutions

| Issue | Likely Cause | Solution |
|-------|--------------|----------|
| No output CSV | Pipeline crash | Check error logs, verify input format |
| Missing organs | Segmentation failure | Review CT quality, try non-fast mode |
| Implausible SUV values | Conversion or metadata error | Check DICOM Units (BQML), dose, weight, and timing tags |
| Unrealistic features | Registration failure | Review CT-PET alignment |
| Empty mask warning | Organ too small in PET space | Expected for small structures, exclude from analysis |

---

## Minimum QC Protocol

For routine processing, at minimum:

1. **Spot-check** 10% of cases visually
2. **Statistical review** of all feature distributions
3. **Automatic flagging** of outliers (>3 SD from mean)
4. **Documentation** of any excluded cases

For publication or clinical use:

1. **100% visual review** of segmentation
2. **Expert validation** of SUV plausibility
3. **Cross-reference** with radiology reports
4. **Reproducibility check** (re-run random subset)
