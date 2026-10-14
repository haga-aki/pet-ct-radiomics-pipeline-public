"""Matched coronal CT/PET crops with original and adapted mask contours."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def main(ct_path, pet_path, original_dir, adapted_dir, output, tiff=False):
    reference = nib.load(str(ct_path))
    spacing = reference.header.get_zooms()
    ct_image, pet_image = reference, nib.load(str(pet_path))
    lung_original = np.zeros(reference.shape, bool)
    lung_adapted = np.zeros(reference.shape, bool)
    for name in ("lung_lower_lobe_left", "lung_lower_lobe_right", "lung_upper_lobe_left",
                 "lung_upper_lobe_right", "lung_middle_lobe_right"):
        lung_original |= np.asarray(nib.load(str(original_dir/f"{name}.nii.gz")).dataobj) > 0
        lung_adapted |= np.asarray(nib.load(str(adapted_dir/f"{name}.nii.gz")).dataobj) > 0
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    figure.subplots_adjust(left=.025, right=.89, bottom=.09, top=.94, hspace=.20, wspace=.14)
    for column, organ in enumerate(("liver", "spleen")):
        original = np.asarray(nib.load(str(original_dir / f"{organ}.nii.gz")).dataobj) > 0
        adapted = np.asarray(nib.load(str(adapted_dir / f"{organ}.nii.gz")).dataobj) > 0
        # Slice selection uses the original mask alone, not correction magnitude.
        footprint = original.any(axis=2)
        y = int(np.median(np.argwhere(footprint)[:, 1]))
        original_slice, adapted_slice = original[:, y], adapted[:, y]
        locations = np.argwhere(original_slice)
        x0 = max(0, int(locations[:, 0].min() - 20 / spacing[0]))
        x1 = min(original.shape[0], int(locations[:, 0].max() + 20 / spacing[0]))
        top = int(locations[:, 1].max())
        z0, z1 = max(0, top - 25), min(original.shape[2], top + 15)
        ct = np.asarray(ct_image.dataobj[x0:x1, y, z0:z1], dtype=float).T
        pet = np.asarray(pet_image.dataobj[x0:x1, y, z0:z1], dtype=float).T
        bounds = (0, (x1-x0)*spacing[0], 0, (z1-z0)*spacing[2])
        xx = (np.arange(x1-x0)+.5)*spacing[0]
        zz = (np.arange(z1-z0)+.5)*spacing[2]
        for row in range(2):
            axis = axes[row, column]
            axis.imshow(ct, cmap="gray", vmin=-160, vmax=240, origin="lower", extent=bounds)
            im = axis.imshow(pet, cmap="inferno", vmin=0, vmax=5, alpha=.5, origin="lower", extent=bounds)
            axis.contour(xx, zz, original_slice[x0:x1, z0:z1].T.astype(float), [.5], colors=["cyan"], linewidths=1.0)
            axis.contour(xx, zz, lung_original[x0:x1, y, z0:z1].T.astype(float), [.5], colors=["cyan"], linewidths=1.0, linestyles="dashed")
            if row:
                axis.contour(xx, zz, adapted_slice[x0:x1, z0:z1].T.astype(float), [.5], colors=["lime"], linewidths=1.0)
                axis.contour(xx, zz, lung_adapted[x0:x1, y, z0:z1].T.astype(float), [.5], colors=["lime"], linewidths=1.0, linestyles="dashed")
            panel = chr(65 + row*2 + column)
            axis.set_title(f"{panel}  {organ.capitalize()}: " + ("original + adapted" if row else "original"), fontsize=11)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
        del original, adapted
    color_axis = figure.add_axes([.92, .27, .018, .48])
    figure.colorbar(im, cax=color_axis, label="SUVbw")
    figure.text(.46, .025, "Original: cyan; adapted: green; organ: solid; lung: dashed", ha="center", fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300)
    if tiff:
        figure.savefig(output.with_suffix('.tiff'), dpi=300, pil_kwargs={'compression':'tiff_lzw'})
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ct", "pet", "original", "adapted", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument('--tiff', action='store_true')
    a = parser.parse_args()
    main(a.ct, a.pet, a.original, a.adapted, a.output, a.tiff)
