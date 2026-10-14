"""Fixed, original-anatomy-based multi-plane views for reviewing contour changes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def render(case, ct_path):
    manifest = json.loads((case/'manifest.json').read_text())
    reference = nib.load(str(ct_path))
    spacing = reference.header.get_zooms()
    ct = reference.get_fdata(dtype=np.float32)
    pet = nib.load(manifest['inputs']['pet']).get_fdata(dtype=np.float32)
    for organ in ('liver', 'spleen'):
        old = np.asarray(nib.load(manifest['inputs']['masks'][organ]['path']).dataobj)>0
        new = np.asarray(nib.load(str(case/'masks'/f'{organ}.nii.gz')).dataobj)>0
        footprint = np.argwhere(old.any(axis=2))
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        indices = []
        for row, fixed_axis in enumerate((1, 0)):
            transverse_axis = 1-fixed_axis
            for column, quantile in enumerate((.25, .5, .75)):
                index = int(np.quantile(footprint[:, fixed_axis], quantile))
                selection = (slice(None), index, slice(None)) if fixed_axis==1 else (index, slice(None), slice(None))
                a, b = old[selection], new[selection]
                occupied = np.argwhere(a)
                low = max(0, int(occupied[:, 0].min()-15/spacing[transverse_axis]))
                high = min(a.shape[0], int(occupied[:, 0].max()+15/spacing[transverse_axis])+1)
                top = int(occupied[:, 1].max())
                bottom, upper = max(0, top-20), min(a.shape[1], top+10)
                crop = (slice(low, high), slice(bottom, upper))
                xx = (np.arange(high-low)+.5)*spacing[transverse_axis]
                zz = (np.arange(upper-bottom)+.5)*spacing[2]
                bounds = (0, (high-low)*spacing[transverse_axis], 0, (upper-bottom)*spacing[2])
                ax = axes[row, column]
                ax.imshow(ct[selection][crop].T, origin='lower', extent=bounds, cmap='gray', vmin=-160, vmax=240)
                ax.imshow(pet[selection][crop].T, origin='lower', extent=bounds, cmap='inferno', vmin=0, vmax=5, alpha=.5)
                ax.contour(xx, zz, a[crop].T, [.5], colors='cyan', linewidths=1)
                ax.contour(xx, zz, b[crop].T, [.5], colors='lime', linewidths=1)
                plane = 'Coronal' if row==0 else 'Sagittal'
                ax.set_title(f'{plane} {column+1}', fontsize=11)
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                indices.append(dict(plane=plane, quantile=quantile, index=index))
        fig.suptitle(f'{case.name.replace("Case_00", "Case ")}  {organ}: original cyan / adapted green', fontsize=12)
        fig.savefig(case/f'{organ}_multiplane.png', dpi=180)
        plt.close(fig)
        (case/f'{organ}_multiplane_slices.json').write_text(json.dumps(indices, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('ct', type=Path)
    args = parser.parse_args()
    render(args.case, args.ct)
