# Hepatic-dome PET transition analysis (study-specific)

Study-specific scripts used for the hepatic-dome analysis in the JJR technical
note (Figs. 4-6). They are **not** part of the standard pipeline entry point
(`run_pipeline.py`) and are provided for transparency and reproduction.

- `measure_liver_pet_edge.py`: case-level and patch-level PET transition
  profiles across the CT-derived liver boundary (Figs. 5 and 6, and the
  displacement and transition-width values reported in the manuscript).
- `create_minimal_liver_dome_fusion_pngs.py`: coronal CT/PET/liver-mask fusion
  images used for minimal visual QC (Fig. 4).

No patient data are included in this repository.

## Inputs

All images are on the **CT grid**:

```
<inputs>/<case>/<case>_CT.nii.gz          # CT
<inputs>/<case>/<case>_liver.nii.gz       # TotalSegmentator liver mask
<inputs>/<case>/<case>_lung_whole.nii.gz  # union of the five lung-lobe masks
<pet>/<case>/<case>_PET_SUV_corrected.nii.gz
    # PET SUVbw resampled to the CT grid (affine, linear interpolation)
```

The PET input is the CT-grid copy of native PET. It is used here and for fusion
images only; radiomic features are extracted on the native PET grid by the
pipeline.

## Usage

```bash
pip install numpy scipy matplotlib nibabel
python measure_liver_pet_edge.py --inputs <inputs> --pet <pet> --out <out_dir>
python create_minimal_liver_dome_fusion_pngs.py --inputs <inputs> --pet <pet> --out <out_dir>
```

`--cases` selects case identifiers (default `Case_001` ... `Case_005`). The
fusion script loads all selected cases before drawing the contact sheet; about
1.7 GB of memory per case was used during verification.

## Analysis definitions (measure_liver_pet_edge.py)

For each in-plane column, `k` is the index of the most superior liver voxel.
Slice offsets are relative to the **center of that voxel**; the CT-derived
boundary is placed half a voxel above it (`k + 0.5`).

- Eligible column: at least 7 liver voxels; `k` at least 10 slices from both
  image edges; at least 2 lung voxels in slices `k+1` to `k+7`; finite profile;
  intrahepatic reference (median of slices `k-7` to `k-3`) at least 1.0 SUV and
  at least 0.35 SUV above the supradiaphragmatic reference (median of slices
  `k+2` to `k+7`).
- Profile: slices `k-9` to `k+9`, normalized as
  (value - supradiaphragmatic) / (intrahepatic - supradiaphragmatic).
- Case and patch profiles: pointwise median of eligible column profiles,
  smoothed with a 1-D Gaussian filter (sigma = 0.7 slice) before edge detection.
- Crossings: linear interpolation; if several descending crossings exist, the
  one nearest the CT-derived boundary is used. Displacement = PET 50% crossing
  minus CT boundary (positive = cranial PET edge); width = 20% minus 80% crossing.
- Patches: 10 x 10 mm by projected in-plane position; patches with fewer than
  20 columns or without finite, ordered 80/50/20% crossings are excluded.

## Verification

Compared with the copies used for the manuscript, these versions differ only in
command-line input/output paths (and the fusion manifest lists PNG file names
instead of paths relative to a project folder). On the manuscript inputs
(Python 3.10.12, numpy 2.2.6, scipy 1.15.3, matplotlib 3.10.9, nibabel 5.4.2):

- the four CSV outputs were byte-identical to those of the original script on
  the same machine and matched the manuscript source CSVs to within 1.4e-14;
- the five per-case fusion PNGs were byte-identical to those of the original
  script on the same machine. The five-case contact sheet was not regenerated
  in this check because of memory limits on the verification machine.
