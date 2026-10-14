"""Optional local reproduction check against an existing approved guarded run.

Accepts explicit paths; contains no study-specific paths or identifiers. Writes
only to a new output directory and never modifies the reference or inputs.
"""
import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pet_diaphragm_adaptation as core
from overlap_gate import run_checked
from run_adaptation_pipeline import adaptation_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.reference / 'manifest.json').read_text())
    inputs = manifest['inputs']
    pet = Path(inputs['pet'])
    masks = Path(inputs['masks']['liver']['path']).parent
    assert core.sha256(pet) == inputs['pet_sha256']
    for source in inputs['masks'].values():
        assert core.sha256(source['path']) == source['sha256']
    parameters = replace(core.Parameters(**manifest['parameters']), field_model='guarded_smooth')
    print('Running guarded implementation through complete-inventory overlap gate', flush=True)
    run_checked(core.run, pet, masks, args.output, parameters, expected_names=adaptation_inventory())
    for name in core.LABELS:
        old = nib.load(str(args.reference / 'masks' / f'{name}.nii.gz'))
        new = nib.load(str(args.output / 'masks' / f'{name}.nii.gz'))
        np.testing.assert_array_equal(np.asarray(old.dataobj), np.asarray(new.dataobj))
        np.testing.assert_array_equal(old.affine, new.affine)
    original_field = np.load(args.reference / 'displacement_field.npz')
    new_field = np.load(args.output / 'displacement_field.npz')
    for name in ('shift_mm', 'center_mm'):
        np.testing.assert_allclose(original_field[name], new_field[name], rtol=0, atol=1e-12)
    def read(path):
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))
    assert read(args.reference / 'paired_patches.csv') == read(args.output / 'paired_patches.csv')
    assert read(args.reference / 'paired_profiles.csv') == read(args.output / 'paired_profiles.csv')
    report = dict(result='PASS', masks_equal=7, fields_absolute_tolerance=1e-12,
                  paired_patch_and_profile_rows='exact equality',
                  reference_manifest_sha256=core.sha256(args.reference / 'manifest.json'),
                  implementation_sha256=core.sha256(Path(core.__file__)),
                  output_manifest_sha256=core.sha256(args.output / 'manifest.json'))
    (args.output / 'reference_reproduction.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
