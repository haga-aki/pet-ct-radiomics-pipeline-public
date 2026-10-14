"""Post-run checks from saved NIfTI masks; no success inferred from summaries."""
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import label

from pet_diaphragm_adaptation import LABELS, sha256


def components(mask):
    bounds = []
    for axis in range(3):
        present = np.flatnonzero(np.any(mask, axis=tuple(a for a in range(3) if a != axis)))
        if not present.size:
            return dict(components=0, largest_fraction=0.)
        bounds.append(slice(present[0], present[-1]+1))
    labelled, count = label(mask[tuple(bounds)], structure=np.ones((3, 3, 3)))
    sizes = np.bincount(labelled.ravel())[1:]
    return dict(components=int(count), largest_fraction=float(sizes.max()/sizes.sum()))


def audit(case):
    manifest = json.loads((case / "manifest.json").read_text())
    field = np.load(case / "displacement_field.npz")
    rows, occupied, failures = [], None, []
    for name in LABELS:
        source = manifest["inputs"]["masks"][name]
        if sha256(source["path"]) != source["sha256"]:
            failures.append(f"Original changed: {name}")
        old_image = nib.load(source["path"])
        new_image = nib.load(str(case / "masks" / f"{name}.nii.gz"))
        old, new = np.asarray(old_image.dataobj) > 0, np.asarray(new_image.dataobj) > 0
        if new.shape != old.shape or not np.array_equal(new_image.affine, old_image.affine):
            failures.append(f"Geometry mismatch: {name}")
        if occupied is None:
            occupied = np.zeros(old.shape, bool)
        overlaps = int(np.count_nonzero(occupied & new))
        occupied |= new
        if overlaps:
            failures.append(f"Adapted target overlap: {name}")
        outside_changes = 0
        spacing_z = old_image.header.get_zooms()[2]
        for z in range(old.shape[2]):
            outside = ((np.abs(z*spacing_z - field["center_mm"]) > manifest["parameters"]["band_mm"])
                       | (field["shift_mm"] == 0))
            outside_changes += np.count_nonzero((old[:, :, z] != new[:, :, z]) & outside)
        if outside_changes:
            failures.append(f"Changes outside deformation support: {name}")
        before, after = components(old), components(new)
        rows.append(dict(organ=name, original=before, adapted=after,
                         changed_voxels=int(np.count_nonzero(old != new)),
                         new_components=after["components"]-before["components"],
                         outside_support_changes=int(outside_changes), target_overlaps=overlaps))
    report = dict(case=case.name, failures=failures, masks=rows,
                  note="Component changes require inspection; passing checks does not prove anatomical accuracy")
    (case / "mask_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if failures:
        raise RuntimeError(failures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    audit(parser.parse_args().case)
