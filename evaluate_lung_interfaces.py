"""Evaluate adjacent lung inferior boundaries on the same fixed organ columns."""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from pet_diaphragm_adaptation import LABELS, Parameters, crossings, get_patches, write_csv


def aligned_crossing(profiles, shifts, spacing, sigma):
    grid = (np.arange(-9, 10) - .5)*spacing
    shifted = np.array([np.interp(grid+d, grid, p, left=np.nan, right=np.nan)
                        for p, d in zip(profiles, shifts)])
    common = np.all(np.isfinite(shifted), axis=0)
    indices = np.flatnonzero(common)
    if len(indices) < 9:
        return np.nan
    median = np.median(shifted[:, common], axis=0)
    filled = np.interp(np.arange(19), indices, median)
    return crossings(filled, spacing, sigma)[.5]


def run(case):
    manifest = json.loads((case / "manifest.json").read_text())
    parameters = Parameters(**manifest["parameters"])
    pet_image = nib.load(manifest["inputs"]["pet"])
    pet = pet_image.get_fdata(dtype=np.float32)
    spacing = np.asarray(pet_image.header.get_zooms())
    original_lung = np.zeros(pet.shape, bool)
    adapted_lung = original_lung.copy()
    for name in LABELS[2:]:
        original_lung |= np.asarray(nib.load(manifest["inputs"]["masks"][name]["path"]).dataobj) > 0
        adapted_lung |= np.asarray(nib.load(str(case / "masks" / f"{name}.nii.gz")).dataobj) > 0
    rows = []
    for organ in LABELS[:2]:
        mask = np.asarray(nib.load(manifest["inputs"]["masks"][organ]["path"]).dataobj) > 0
        patches = get_patches(pet, mask, original_lung, spacing, parameters)
        for patch in patches:
            profiles, before_offsets, after_offsets = [], [], []
            missing = 0
            for x, y, top, profile in patch["columns"]:
                original_floor = top+1+np.flatnonzero(original_lung[x, y, top+1:top+8])[0]
                start = max(0, original_floor-int(np.ceil(parameters.max_shift_mm/spacing[2]))-1)
                stop = min(mask.shape[2], original_floor+int(np.ceil(parameters.max_shift_mm/spacing[2]))+2)
                occupied = np.flatnonzero(adapted_lung[x, y, start:stop])
                if len(occupied) == 0:
                    missing += 1
                    continue
                adapted_floor = start+occupied[0]
                profiles.append(profile)
                before_offsets.append((original_floor-top-1)*spacing[2])
                after_offsets.append((adapted_floor-top-1)*spacing[2])
            if missing:
                before, after = np.nan, np.nan  # Do not silently change paired columns.
            else:
                before = aligned_crossing(profiles, before_offsets, spacing[2], parameters.profile_sigma_voxels)
                after = aligned_crossing(profiles, after_offsets, spacing[2], parameters.profile_sigma_voxels)
            rows.append(dict(interface=f"lung_above_{organ}", patch=f"{organ}_{patch['key'][0]}_{patch['key'][1]}",
                             split="training" if patch["training"] else "held_out",
                             original_columns=len(patch["columns"]), missing_adapted_lung_columns=missing,
                             before_mm=before, after_mm=after))
    write_csv(case / "paired_lung_patches.csv", rows)
    print(f"{case.name}: {len(rows)} lung-interface patches written", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    run(parser.parse_args().case)
