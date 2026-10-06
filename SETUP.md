# PET/CT Radiomics Pipeline - Setup Guide

## Overview

This pipeline automates the following processes:
- DICOM to NIfTI conversion
- Native-grid PET SUVbw conversion from standard DICOM activity concentration (BQML)
- TotalSegmentator-based CT organ segmentation (`total` task; 117 mask files with v2.11.0)
- Transfer of CT-derived masks to the native PET grid (nearest-neighbor)
- PyRadiomics feature extraction
- Result visualization

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10+ | 3.10+ |
| RAM | 8GB | 16GB+ |
| GPU | - | NVIDIA (8GB+ VRAM) |
| Storage | 5GB | 10GB+ |

### Supported Operating Systems

- **Linux**: Ubuntu 20.04+, tested on Ubuntu 24.04
- **macOS**: Intel/Apple Silicon (M1/M2/M3)
- **Windows**: Windows 10/11

---

## Installation

### Option 1: Conda (Recommended)

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/pet-ct-radiomics-pipeline.git
cd pet-ct-radiomics-pipeline

# 2. Create conda environment
# Linux/macOS
conda env create -f environment.yml
conda activate pet_radiomics

# Windows
conda env create -f environment_windows.yml
conda activate pet_radiomics

# 3. Create configuration file
cp config.yaml.example config.yaml
```

### Option 2: pip

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/pet-ct-radiomics-pipeline.git
cd pet-ct-radiomics-pipeline

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install CUDA-enabled PyTorch
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 5. Create configuration file
cp config.yaml.example config.yaml
```

### Windows Quick Setup

```cmd
REM Run the setup script
setup_windows.bat
```

This script will:
- Create conda environment (`pet_radiomics`)
- Install all dependencies
- Optionally install CUDA-enabled PyTorch
- Create config.yaml

---

## Directory Structure

```
pet-ct-radiomics-pipeline/
├── raw_download/           # Place DICOM data here
│   ├── Patient_001/
│   │   ├── CT/            # CT DICOM series
│   │   └── PET/           # PET DICOM series
│   └── Patient_002/
│       └── ...
├── nifti_images/           # Converted NIfTI (auto-generated)
├── segmentations/          # Segmentation results (auto-generated)
├── config.yaml             # Configuration file
├── config.yaml.example     # Configuration template
├── run_pipeline.py         # Main pipeline script
└── radiomics_results.csv   # Output (auto-generated)
```

---

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
# Modalities to process
modalities:
  - CT
  - PET

# Target organs (representative 8-organ set recommended for validation)
# Covers different sizes, physiologic uptake patterns, and PV susceptibility
organs:
  - liver              # Large, stable
  - spleen             # Large, stable
  - kidney_left        # Medium, physiologic uptake
  - kidney_right       # Medium, physiologic uptake
  - adrenal_gland_left   # Small, PV-susceptible
  - adrenal_gland_right  # Small, PV-susceptible
  - aorta              # Blood pool reference
  - vertebrae_L1       # Bone marrow representative

# Or use all labels of the configured TotalSegmentator task:
# organs:
#   - all

# Segmentation settings
segmentation:
  fast: true              # Fast mode (lower resolution)
  tasks:
    CT: total             # Full organ segmentation
    MR: total_mr          # MR-specific task
    PET: use_ct_mask      # Use CT mask for PET
    SPECT: use_ct_mask    # Use CT mask for SPECT

# Output settings
output:
  csv_file: radiomics_results.csv
  include_diagnostics: false
```

---

## Usage

### Basic Usage

```bash
# Activate environment
conda activate pet_radiomics

# Run pipeline
python run_pipeline.py
```

### GUI Application

```bash
# Launch GUI (macOS/Windows)
python gui_launcher.py
```

### Batch Scripts

```bash
# macOS
./run_analysis.command

# Windows
run_analysis.bat
```

---

## Processing Steps

1. **Data Preparation**: Place DICOM folders in `raw_download/`
2. **ID Anonymization**: Folder names are converted to anonymous IDs
3. **DICOM to NIfTI**: Automatic conversion with modality detection
4. **SUV Conversion**: Native PET converted to SUVbw from BQML data; ambiguous timing metadata is rejected
5. **Segmentation**: TotalSegmentator processes CT images (`total` task)
6. **Mask Transfer**: CT-derived masks transferred to the native PET grid in world coordinates (nearest-neighbor; NOT de novo registration)
7. **Feature Extraction**: 93 first-order/texture features on native PET plus 14 shape features from the CT-grid mask (107 PyRadiomics features, IBSI terminology)
8. **QC and Output**: A CT-grid PET copy is created for fusion images only; results and run-scoped QC reports are saved as CSV

---

## SUV Conversion Support

SUVbw is calculated from standard DICOM activity concentration
(`Units = BQML`) with `DecayCorrection` START or ADMIN. Missing or ambiguous
timing metadata is rejected rather than inferred. Pre-scaled or private-tag
SUV encodings are not converted and require a validated vendor-specific
adapter. This is an explicit support limit, not a claim of universal vendor
support.

---

## Output Files

| File | Description |
|------|-------------|
| `radiomics_results.csv` | Radiomics features (107 features × organs × modalities) |
| `qc/<run_id>/nifti_images/*.nii.gz` | Converted NIfTI images for the run |
| `qc/<run_id>/segmentations/*/` | Organ segmentation masks for the run |
| `qc/<run_id>/case_qc.csv`, `organ_qc.csv`, `stage_qc.csv` | Run-scoped technical QC reports |
| `qc/<run_id>/visualizations/` | Fusion images for visual review |

---

## Troubleshooting

### GPU Issues with TotalSegmentator

```bash
# Run in CPU mode
export TOTALSEG_DISABLE_GPU=1  # Linux/Mac
set TOTALSEG_DISABLE_GPU=1     # Windows
python run_pipeline.py
```

### Out of Memory

- Set `fast: true` in config.yaml (uses lower resolution)
- Reduce `batch_size` in segmentation settings

### DICOM Not Detected

- Verify folder structure matches expected format
- DICOM files can be with or without `.dcm` extension
- Ensure modality-specific subfolders (CT/, PET/) exist

### TotalSegmentator Model Download

Models (~1.5GB) are downloaded automatically on first run.
If download fails, check internet connection and try again.

---

## Apple Silicon (M1/M2/M3) Notes

- TotalSegmentator runs in CPU mode (MPS not supported)
- Processing is slower but stable
- Use `fast: true` for reasonable performance

---

## License

This pipeline is released under the MIT License.

Note: TotalSegmentator is free for academic use only. Commercial use requires a license from the developers.

---

## References

- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) - Wasserthal et al.
- [PyRadiomics](https://pyradiomics.readthedocs.io/) - van Griethuysen et al.
- [IBSI](https://theibsi.github.io/) - Image Biomarker Standardisation Initiative
