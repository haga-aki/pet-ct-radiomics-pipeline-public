"""Opt-in SI-only diaphragm adaptation and paired native-PET radiomics.

For cached CT/PET/masks use this entry point. For DICOM input, run_pipeline.py
--adapt-diaphragm adds the same stage after the unmodified original extraction.
Original and adapted results and native-grid mask transfers remain separate.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from pet_diaphragm_adaptation import LABELS, Parameters, run, sha256
from overlap_gate import check_masks, require_clear, run_checked
from mask_inventory import expected_masks


def adaptation_inventory():
    """The evaluated method requires a complete total-task inventory, no subset."""
    names = set(expected_masks({'segmentation': {'tasks': {'CT': 'total'}}}, 'CT'))
    if len(names) != 117 or not set(LABELS) < names:
        raise ValueError('Guarded adaptation requires the evaluated 117-label total-task inventory')
    return names


def verify_overlap_inputs(original_masks, adaptation_dir=None):
    names = adaptation_inventory()
    require_clear(check_masks(original_masks, original_masks, names))
    if adaptation_dir is not None:
        require_clear(check_masks(Path(adaptation_dir) / 'masks', original_masks, names))
    return names


def verify_inputs(native_pet, ct_path, original_masks, adaptation_dir):
    from nibabel.processing import resample_from_to
    from pet_diaphragm_adaptation import geometry
    manifest = json.loads((adaptation_dir / 'manifest.json').read_text())
    ct = nib.load(str(ct_path))
    recorded_pet = nib.load(manifest['inputs']['pet'])
    geometry(recorded_pet, ct)
    for name in LABELS:
        if sha256(original_masks / f'{name}.nii.gz') != manifest['inputs']['masks'][name]['sha256']:
            raise ValueError(f'Adaptation belongs to different original masks: {name}')
    if sha256(manifest['inputs']['pet']) != manifest['inputs']['pet_sha256']:
        raise ValueError('Adaptation PET input changed')
    expected = resample_from_to(nib.load(str(native_pet)), ct, order=1)
    actual_data = recorded_pet.get_fdata(dtype=np.float32)
    expected_data = np.asarray(expected.dataobj, dtype=np.float32)
    maximum = 0.
    for x in range(ct.shape[0]):
        a, b = actual_data[x], expected_data[x]
        if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a, b, atol=1e-6, rtol=1e-6):
            raise ValueError('Native SUV PET does not reproduce the PET copy used for adaptation')
        maximum = max(maximum, float(np.max(np.abs(a-b))))
    return dict(ct_grid_pet_max_absolute_difference=maximum, atol=1e-6, rtol=1e-6)


def extract_pair(native_pet, ct_path, original_masks, adaptation_dir, output_dir,
                 case_id, extraction_config=None):
    import pandas as pd
    import run_pipeline as pipeline
    from qc_reporting import QCReport

    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    verify_overlap_inputs(original_masks, adaptation_dir)
    consistency = verify_inputs(native_pet, ct_path, original_masks, adaptation_dir)
    output_dir.mkdir(parents=True)
    config = copy.deepcopy(extraction_config or pipeline.load_config())
    params = Path(config.get('radiomics', {}).get('params_file', 'params.yaml'))
    if not params.is_absolute():
        params = Path(__file__).parent / params
    config.setdefault('radiomics', {})['params_file'] = str(params.resolve())
    config['organs'] = list(LABELS)
    config['output']['include_diagnostics'] = False
    previous_config = pipeline.CONFIG
    pipeline.CONFIG = config
    rows = []
    try:
        for variant, masks in [('original', original_masks), ('adapted', adaptation_dir / 'masks')]:
            qc = QCReport(output_dir / variant)
            qc.start_case(case_id)
            provenance = qc.provenance(config, Path(__file__).parent)
            result = pipeline.step3_radiomics(
                native_pet, masks, case_id, 'PET', shape_image_path=ct_path, qc=qc,
                organs_override=LABELS, native_mask_directory=output_dir / variant / 'native_pet_masks')
            for row in result:
                row.update(provenance)
                row['MaskVariant'] = variant
                rows.append(row)
            qc.stage(case_id, 'case', 'finished')
            pd.DataFrame(rows).to_csv(output_dir / 'paired_radiomics.csv', index=False)
            if len(result) != len(LABELS):
                raise RuntimeError(f"{variant}: extracted {len(result)}/{len(LABELS)} requested targets; inspect QC")
    finally:
        pipeline.CONFIG = previous_config
    columns = [k for k in rows[0] if k.startswith('original_')]
    array = pd.DataFrame(rows)[columns].to_numpy(dtype=float)
    report = dict(case=case_id, rows=len(rows), features_per_row=len(columns),
                  input_consistency=consistency,
                  all_finite=bool(np.isfinite(array).all()),
                  native_pet_sha256=sha256(native_pet), ct_sha256=sha256(ct_path),
                  adaptation_manifest_sha256=sha256(adaptation_dir / 'manifest.json'),
                  extractor_sha256=sha256(Path(pipeline.__file__)),
                  workflow_sha256=sha256(Path(__file__)), params_sha256=sha256(params),
                  feature_grid='native PET, nearest-neighbour mask transfer',
                  shape_grid='CT grid, separately for each mask variant')
    (output_dir / 'extraction_manifest.json').write_text(json.dumps(report, indent=2))
    return rows


def process_case(native_pet, ct_path, masks, output_dir, case_id, extraction_config=None):
    from nibabel.processing import resample_from_to
    from audit_adaptation import audit

    native_pet, ct_path, masks, output_dir = map(Path, (native_pet, ct_path, masks, output_dir))
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    names = verify_overlap_inputs(masks)
    output_dir.mkdir(parents=True)
    ct = nib.load(str(ct_path))
    pet_copy = output_dir / 'pet_suv_ct_grid.nii.gz'
    nib.save(resample_from_to(nib.load(str(native_pet)), ct, order=1), str(pet_copy))
    adaptation = output_dir / 'adaptation'
    parameters = Parameters(field_model='guarded_smooth')
    run_checked(run, pet_copy, masks, adaptation, parameters, expected_names=names)
    audit(adaptation)
    rows = extract_pair(native_pet, ct_path, masks, adaptation, output_dir / 'radiomics',
                        case_id, extraction_config)
    # Separate aggregate for this opt-in stage; never merge over original rows.
    import pandas as pd
    pd.DataFrame(rows).to_csv(output_dir / 'paired_radiomics.csv', index=False)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pet', 'ct', 'masks', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--case-id', required=True, help='Deidentified output identifier')
    args = parser.parse_args()
    process_case(args.pet, args.ct, args.masks, args.output, args.case_id)
