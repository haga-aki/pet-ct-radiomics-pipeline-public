# Local PET-guided diaphragm mask adaptation

This experimental, opt-in stage locally extends or retracts the diaphragmatic
end of CT-derived lung masks and adjusts adjacent liver or spleen masks. It does
not translate entire organs. Left–right and anterior–posterior coordinates are
fixed; displacement tapers to zero away from the diaphragm, retaining remote
lung apices. PET intensities and attenuation maps are not changed.

## Inputs and execution

The regular pipeline option is `--adapt-diaphragm`. It is off by default and runs
after original PET feature extraction. Original feature rows remain unchanged;
the adaptation stage writes its own paired results.

For existing inputs, run from the repository root:

```bash
python run_adaptation_pipeline.py \
  --pet /path/to/native_pet_suv.nii.gz \
  --ct /path/to/ct.nii.gz \
  --masks /path/to/ct_masks \
  --output /path/to/new_output_directory \
  --case-id ExampleCase
```

`--pet` must contain native-grid SUV values, not unconverted activity or counts.
The mask directory must contain the complete 117-label TotalSegmentator total-task
inventory. Missing, unexpected, uncompressed, nonbinary, nonfinite, or geometrically
inconsistent masks are rejected. Original target–target or target–other-organ
overlap stops the stage, even if that overlap predates adaptation.
Inputs must be in the same physical scanner coordinate system. The correction
currently supports only axis-aligned CT grids with increasing superior z.
Oblique or reversed-z inputs are rejected rather than silently treated as SI.
Existing output directories are rejected to avoid replacing earlier results.

The stage makes a linear CT-grid PET copy for transition analysis. Original and
adapted masks are then independently transferred by nearest-neighbor sampling
to the unchanged native PET for intensity and texture extraction. Shape features
are calculated from each variant on the CT grid.

## Estimation

The superior liver/spleen surface and adjacent inferior lung surface define an
interval. A PET 50% crossing within that interval has zero discrepancy. Eligible
columns are grouped into 10-mm projected patches; the median profile is smoothed
with a Gaussian sigma of 0.7 slices. Alternating patches are used for fitting and
held-out description. Both sets come from the same scan and are not independent
clinical validation data.

Fitting patches require at least 75% of columns with a single descending 50%
crossing and a crossing IQR no greater than 6 mm. Reliability equals that fraction
divided by one plus squared IQR in units of SI slice spacing. Targets with absolute
magnitude no greater than 1.5 mm are zeroed; magnitudes exceeding 9 mm are rejected. An organ needs
at least five accepted fitting patches, with weighted q25 > +1.5 mm or weighted
q75 < -1.5 mm. Otherwise no field is estimated for that organ.

The weighted median and quartiles control activation only. A normalized Gaussian
radial-basis model (20-mm lattice, sigma 20 mm) represents the total proposed
displacement, with no global baseline term or separate 3-mm residual cap. Huber
fitting (delta 1.5 mm) uses normalized reliability weights, coefficient ridge
0.01, and adjacent-control difference penalty 0.1. Both penalties are normalized
by their respective counts. Coefficients are bounded to ±9 mm. IRLS stops at
maximum coefficient change below 1e-6 mm or 40 iterations.

The PET-fitting gap limit remains 6 mm. Field support instead uses original CT
columns with at least two lung voxels among the next seven superior slices. This
CT-only footprint is Gaussian-smoothed with sigma 10 mm; amplitude is
min(smoothed footprint / 0.5, 1), multiplied by a 15-mm outward cosine taper.
Original superior-face heights use normalized Gaussian smoothing over that
footprint. This changes displacement support, not segmentation holes.

The two proposed organ fields are blended with their support weights. Scales
1.00, 0.95, ..., 0.00 are tested in descending order on eligible fitting patches
only. The guard uses the actual inverse nearest-neighbor column sampler,
protected-label veto, and profile recrossing. The largest passing scale limits
each fitting patch's absolute interval-error increase to 1.5 mm and does not
increase the pooled reliability-weighted mean absolute error. No held-out values
select the scale. Saved-mask fitting errors must reproduce these predictions.
This guard does not bound individual-column or held-out worsening.

The final field tapers linearly to zero 36 mm above and below the smoothed surface.
All target labels share one inverse SI transformation; internal lobe interfaces
inside that band can move. No supersampling, mask smoothing or hole filling is
performed. Cached-input and DICOM opt-in workflows select this guarded model.
The low-level core retains legacy model functions for reproducibility; they are
not the default adaptation workflow.
These engineering choices were developed on five cases, including activation-rule
refinement; held-out patches are not an untouched validation set. Regularization
limits local departures but does not guarantee exact shape preservation.

## Outputs and review

- `pet_suv_ct_grid.nii.gz`: interpolated analysis copy, not used for radiomics.
- `adaptation/masks/`: seven adapted CT-grid target masks.
- `adaptation/paired_patches.csv` and `paired_profiles.csv`: original/adapted
  correspondence, with fitting and held-out patch identifiers.
- `adaptation/volume_qc.csv`, `mask_audit.json`, `displacement_field.npz`, and
  `manifest.json`: structural checks, applied field, parameters and input hashes.
- `adaptation/guard_report.json`: tested scales and exact fitting predictions.
- `adaptation/overlap_check.json`: saved-mask overlap gate and source hashes.
- `radiomics/`: independent native-grid mask transfers, QC and paired extraction.
- `paired_radiomics.csv`: separate aggregate with an explicit mask-variant column.

Review both improvement and worsening, finite-pair counts, volumes, connected
components, and matched fusion images. A smaller PET discrepancy is not proof of
anatomical correctness or more accurate SUV. Focal uptake, low contrast, surgical
anatomy and attenuation-related artifacts can invalidate the interpretation of an
uptake transition. Do not replace original masks without image review.

`shape_qc.py CASE_DIRECTORY` records descriptive surface-height finite differences.
`render_multiplane_qc.py CASE_DIRECTORY CT_IMAGE` makes fixed-anatomy-based coronal
and sagittal QC views. Discrete contour roughness may increase after nearest-neighbor
warping even with a smooth displacement field; do not treat a smooth field as proof
of an unchanged contour shape.

Manifests and feature diagnostics can contain local input paths and study data.
Keep these output directories outside a public code repository.

Outputs are computed in a hidden pending directory and moved to the accepted
adaptation path only after saved-mask overlap and source-preservation checks
pass. Failed candidates remain explicitly rejected for diagnosis. The overlap
guarantee covers the seven targets against one another and against the 110
supplied other-organ masks; it does not validate other-organ pairs or anatomy.

## Verification

Run `python -m unittest discover -s tests -v`. For an existing approved guarded
result, optional `tests/verify_guarded_reference.py --reference REFERENCE_DIR
--output NEW_DIRECTORY` checks all seven masks, fields, and patch/profile rows.
It does not modify the reference or source masks.

## Plotting paired outputs

`render_adaptation_qc.py --ct CT --pet CT_GRID_SUV --original ORIGINAL_MASKS
--adapted ADAPTED_MASKS --output fusion.png --tiff` creates matched coronal
views. Supply `adaptation/masks` for the adapted directory. The optional TIFF
has the same content as the PNG. CT/PET must use the same grid.

`build_interval_figure.py RESULTS_ROOT` summarizes five case folders named
`Case_001` through `Case_005`, each containing the adaptation CSVs and manifest.
It writes profiles, signed interval distances and organ-face offsets with
within-case interquartile ranges. This five-case plotting utility is separate
from the general correction entry point.
