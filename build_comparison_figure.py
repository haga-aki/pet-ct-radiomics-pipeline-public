"""Original versus adapted hepatic and lung contours in two fixed planes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


def build(case, ct_path, output):
    manifest = json.loads((case/'manifest.json').read_text())
    ct_image = nib.load(str(ct_path))
    spacing = ct_image.header.get_zooms()
    ct = ct_image.get_fdata(dtype=np.float32)
    pet = nib.load(manifest['inputs']['pet']).get_fdata(dtype=np.float32)
    old = np.asarray(nib.load(manifest['inputs']['masks']['liver']['path']).dataobj)>0
    new = np.asarray(nib.load(str(case/'masks/liver.nii.gz')).dataobj)>0
    footprint = np.argwhere(old.any(axis=2))
    old_lung, new_lung = np.zeros(old.shape, bool), np.zeros(old.shape, bool)
    for name, record in manifest['inputs']['masks'].items():
        if name.startswith('lung_'):
            old_lung |= np.asarray(nib.load(record['path']).dataobj)>0
            new_lung |= np.asarray(nib.load(str(case/'masks'/f'{name}.nii.gz')).dataobj)>0
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.subplots_adjust(left=.025, right=.89, bottom=.09, top=.94, hspace=.18, wspace=.10)
    for col, fixed_axis in enumerate((1, 0)):
        index = int(np.median(footprint[:, fixed_axis]))
        transverse = 1-fixed_axis
        selection = (slice(None), index, slice(None)) if fixed_axis==1 else (index, slice(None), slice(None))
        locations = np.argwhere(old[selection])
        low = max(0, int(locations[:, 0].min()-20/spacing[transverse]))
        high = min(old[selection].shape[0], int(locations[:, 0].max()+20/spacing[transverse])+1)
        top = locations[:, 1].max()
        lower, upper = max(0, top-25), min(old.shape[2], top+15)
        crop = (slice(low, high), slice(lower, upper))
        xx, zz = (np.arange(high-low)+.5)*spacing[transverse], (np.arange(upper-lower)+.5)*spacing[2]
        bounds = (0, (high-low)*spacing[transverse], 0, (upper-lower)*spacing[2])
        for row in range(2):
            ax = axes[row, col]
            ax.imshow(ct[selection][crop].T, cmap='gray', vmin=-160, vmax=240, origin='lower', extent=bounds)
            im = ax.imshow(pet[selection][crop].T, cmap='inferno', vmin=0, vmax=5, alpha=.5, origin='lower', extent=bounds)
            variants = [(old, old_lung, 'cyan')] + ([(new, new_lung, 'lime')] if row else [])
            for organ, lung, color in variants:
                ax.contour(xx, zz, organ[selection][crop].T, [.5], colors=color, linewidths=1)
                ax.contour(xx, zz, lung[selection][crop].T, [.5], colors=color, linewidths=1, linestyles='dashed')
            ax.set_title(f'{chr(65+row*2+col)}  '+('Coronal' if col==0 else 'Sagittal')+
                         (': original' if row==0 else ': original + adapted'), fontsize=11)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
    color = fig.add_axes([.92, .27, .018, .48])
    fig.colorbar(im, cax=color, label='SUVbw')
    fig.text(.46, .025, 'Original: cyan; adapted: green; liver: solid; lung: dashed', ha='center', fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    fig.savefig(output.with_suffix('.tiff'), dpi=300, pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('case', 'ct', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    build(args.case, args.ct, args.output)
