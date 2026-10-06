# Pipeline Overview

Detailed technical documentation of the PET-CT Radiomics Pipeline.

## Architecture

```
Paired PET/CT DICOM
        │
        ▼
Volumetric series selection and NIfTI conversion
        │
        ▼
Native PET SUVbw conversion and CT segmentation
        │
        ▼
Transfer CT-derived masks to native PET coordinates
        │
        ▼
Calculate 93 native-PET intensity/texture features
and 14 original CT-grid shape features
        │
        ▼
Create fusion/geometry QC artifacts and radiomics.csv
```

This single-stream view describes the execution order. CT supplies anatomy and
PET supplies activity values within the same workflow; the two modalities are
not presented as independent analysis branches that merge only at the end.

---

## Step-by-Step Processing

### Step 1: DICOM to NIfTI Conversion

**Input:** DICOM directory structure
**Output:** NIfTI files (.nii.gz)

```python
# Using dicom2nifti
dicom2nifti.convert_directory(dicom_dir, output_dir)
```

**Key considerations:**
- Automatically detects modality from DICOM tags
- Handles multi-series DICOM folders
- Preserves spatial orientation and metadata

---

### Step 2: PET/CT Spatial Relationship and Fusion-QC Copy

**Input:** CT and native PET NIfTI images
**Output:** CT-grid PET copy for fusion QC only

**Important:** This step is not de novo multimodal registration. For integrated PET/CT scanner data, PET and CT images are already spatially related at acquisition. The CT-grid copy is used only for fusion visualization; radiomics remains on the native PET image.

```python
# Using nibabel affine-based resampling for the QC image only
pet_qc = resample_from_to(native_pet_suv, ct_img, order=1)
```

**Method:**
- Uses NIfTI world coordinates (affine transformation matrices)
- Uses linear interpolation only for the fusion-QC copy
- Preserves native PET voxel values for first-order and texture extraction

---

### Step 3: SUV Conversion

**Input:** Modality-rescaled PET NIfTI values + PET DICOM metadata
**Output:** SUVbw values

On the pipeline path, SUVbw is calculated only from standard DICOM activity
concentration (`Units = BQML`); pre-scaled or private-tag SUV encodings are
rejected until a validated vendor-specific adapter is available. A legacy
raw-DICOM API in `suv_converter.py` can still apply private scale factors; it
is not used by the pipeline and was not evaluated.

#### Activity concentration (BQML)

```python
# Standard SUV formula with decay correction
activity = pixel_value  # Bq/ml
weight = patient_weight  # g
dose = injected_dose  # Bq
decay = np.exp(-np.log(2) * time_elapsed / half_life)
suv = activity * weight / (dose * decay)
```

`dicom2nifti` has already applied DICOM Rescale Slope and Intercept before this
step. The BQML path must not apply those tags again; otherwise each series is
multiplied by its case-specific rescale slope.

Missing, malformed, or ambiguous timing metadata (for example, an unresolved
midnight rollover) is rejected rather than inferred.

---

### Step 4: CT Segmentation

**Input:** CT NIfTI file
**Output:** organ masks on the CT grid (NIfTI)

```bash
TotalSegmentator -i ct.nii.gz -o segmentations/ --fast
```

**Output structure:**
```
segmentations/
├── liver.nii.gz
├── spleen.nii.gz
├── kidney_left.nii.gz
├── kidney_right.nii.gz
├── ...
└── combined_all_organs.nii.gz
```

---

### Step 5: Mask Application

**Input:** CT-grid masks and native-grid PET SUV
**Output:** Native-PET-grid image-mask pairs plus CT-grid shape pairs

The CT-derived masks are transferred to the native PET grid with nearest-neighbor interpolation. Shape features are calculated separately from the original CT-grid masks:

```python
# Transfer labels, not PET intensities
mask_native_pet = sitk.Resample(
    ct_mask,
    native_pet_suv,
    sitk.Transform(),
    sitk.sitkNearestNeighbor,
    0,
    sitk.sitkUInt8,
)
```

**Note:** The 14 shape features come from the original CT-grid mask; the 93 first-order and texture features come from the native-PET-grid pair.

---

### Step 6: Radiomic Feature Extraction

**Input:** Image (CT or PET) + mask
**Output:** 107 features per organ

```python
from radiomics import featureextractor

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllFeatures()

features = extractor.execute(image, mask)
```

**Feature categories:**
| Category | Features | Description |
|----------|----------|-------------|
| First-order | 18 | Intensity statistics |
| Shape | 14 | 3D morphology |
| GLCM | 24 | Texture co-occurrence |
| GLRLM | 16 | Run-length patterns |
| GLSZM | 16 | Size-zone patterns |
| GLDM | 14 | Dependence patterns |
| NGTDM | 5 | Neighborhood patterns |

---

## Data Flow

The workflow is presented as one ordered sequence. CT supplies the anatomical
mask geometry at the mask-application step; it is not a separate downstream
analysis branch.

```
Paired PET/CT DICOM
        │
        ▼
Series selection and NIfTI conversion
        │
        ▼
Native PET SUV conversion + CT segmentation
        │
        ▼
Apply CT masks to native PET coordinates
        │
        ▼
PET first-order/texture + CT-mask shape extraction
        │
        ▼
QC artifacts and radiomics.csv
```

---

## Performance Characteristics

Runtime depends on image dimensions, the number and size of requested labels,
the TotalSegmentator mode, hardware, and whether model weights are already
cached. Benchmark the complete configured workflow on the deployment system;
the repository does not claim fixed per-step timings.

---

## Error Handling

### Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| "Cannot find DICOM files" | Wrong path or format | Verify directory structure |
| "CUDA out of memory" | GPU memory exceeded | Use `--fast` or CPU mode |
| "Empty mask" | Label not applicable (e.g., sex-specific or pathology-dependent), structure outside the imaged range, segmentation miss, or loss after PET-grid transfer | Check `organ_qc.csv` status and the images before classifying as failure |
| "SUV values negative" | Conversion or metadata error | Check DICOM Units (BQML) and timing tags |
| "Alignment mismatch" | Non-integrated scanner or patient motion | May require additional registration |

### Logging

Pipeline generates logs at multiple levels:
```
logs/
├── pipeline.log        # Main workflow
├── segmentation.log    # TotalSegmentator output
├── extraction.log      # PyRadiomics details
└── errors.log          # Error messages only
```
