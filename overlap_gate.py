"""Fail-closed export gate for mutually exclusive organ masks.

All individual NIfTI masks in the supplied directory are checked. Combined
label maps are not independent organs. This cannot detect an unprovided organ
or establish anatomical correctness. Historical analysis outputs stay intact.
"""
import json
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np

from pet_diaphragm_adaptation import LABELS, geometry, sha256


def binary_values(values):
    if values.dtype.kind in 'bu':
        return values.max() <= 1
    if values.dtype.kind == 'i':
        return values.min() >= 0 and values.max() <= 1
    return np.isfinite(values).all() and not np.any((values != 0) & (values != 1))


def check_masks(target_dir, reference_dir, expected_names=None):
    target_dir, reference_dir = Path(target_dir), Path(reference_dir)
    if list(reference_dir.glob('*.nii')) or list(target_dir.glob('*.nii')):
        raise ValueError('Unsupported uncompressed NIfTI masks; supply the complete .nii.gz inventory')
    inventory = {p.name.removesuffix('.nii.gz') for p in reference_dir.glob('*.nii.gz')
                 if not p.name.startswith('combined')}
    if expected_names is not None and inventory != set(expected_names):
        raise ValueError(f'Mask inventory mismatch: missing={sorted(set(expected_names)-inventory)}, '
                         f'unexpected={sorted(inventory-set(expected_names))}')
    if target_dir != reference_dir:
        actual = {p.name.removesuffix('.nii.gz') for p in target_dir.glob('*.nii.gz')}
        if actual != set(LABELS):
            raise ValueError('Adapted target inventory differs from the seven expected labels')
    reference = nib.load(str(reference_dir / 'liver.nii.gz'))
    occupancy = np.zeros(reference.shape, np.uint8)
    conflicts, checked = [], {}
    for index, name in enumerate(LABELS, 1):
        path = target_dir / f'{name}.nii.gz'
        image = nib.load(str(path)); geometry(image, reference)
        values = np.asarray(image.dataobj)
        if not binary_values(values):
            raise ValueError(f'Nonbinary or nonfinite target mask: {name}')
        mask = values > 0
        for other in np.unique(occupancy[mask]):
            if other:
                conflicts.append(dict(first=LABELS[int(other)-1], second=name,
                                      voxels=int(np.count_nonzero(mask & (occupancy == other)))))
        occupancy[mask] = index
        checked[name] = sha256(path)
    protected_hashes = {}
    for path in sorted(reference_dir.glob('*.nii.gz')):
        name = path.name.removesuffix('.nii.gz')
        if name in LABELS or name.startswith('combined'):
            continue
        image = nib.load(str(path)); geometry(image, reference)
        values = np.asarray(image.dataobj)
        if not binary_values(values):
            raise ValueError(f'Nonbinary or nonfinite protection mask: {name}')
        mask = values > 0
        for index in np.unique(occupancy[mask]):
            if index:
                conflicts.append(dict(first=LABELS[int(index)-1], second=name,
                                      voxels=int(np.count_nonzero(mask & (occupancy == index)))))
        protected_hashes[name] = sha256(path)
    if not protected_hashes:
        raise ValueError('No other-organ masks supplied; overlap protection is unverified')
    return dict(passed=not conflicts, conflicts=conflicts,
                target_sha256=checked, protected_sha256=protected_hashes,
                scope='Targets against each other and all supplied individual other-organ masks')


def require_clear(report):
    if not report['passed']:
        raise ValueError('Organ overlap: ' + json.dumps(report['conflicts']))


def run_checked(delegate, pet_path, mask_dir, output_dir, parameters, *, expected_names):
    """Publish output only after source and saved-mask checks both pass.

    Rejected staged output is retained for diagnosis, never exported as accepted.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not set(LABELS) < set(expected_names):
        raise ValueError('An explicit complete source mask inventory is required')
    before = check_masks(mask_dir, mask_dir, expected_names)
    require_clear(before)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output_dir.name + '.pending-',
                                    dir=output_dir.parent))
    candidate = staging / 'candidate'
    try:
        result = delegate(pet_path, mask_dir, candidate, parameters)
        after = check_masks(candidate / 'masks', mask_dir, expected_names)
        (staging / 'overlap_check.json').write_text(json.dumps(after, indent=2))
        require_clear(after)
        # Detect changes to any source/protection masks while the job was running.
        current = check_masks(mask_dir, mask_dir, expected_names)
        if current != before:
            raise ValueError('Source mask inventory or content changed during adaptation')
        (candidate / 'overlap_check.json').write_text(json.dumps(after, indent=2))
        if output_dir.exists():
            raise FileExistsError(output_dir)
        candidate.rename(output_dir)
        return result
    except Exception as exc:
        (staging / 'REJECTED.json').write_text(json.dumps(dict(error=str(exc)), indent=2))
        raise
