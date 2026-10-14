import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

import run_adaptation_pipeline as workflow
from pet_diaphragm_adaptation import LABELS, sha256


class WorkflowTests(unittest.TestCase):
    def test_cached_input_identity_and_wrong_pet_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            masks, adaptation = root/'masks', root/'adaptation'
            masks.mkdir()
            adaptation.mkdir()
            pet, ct = root/'pet.nii.gz', root/'ct.nii.gz'
            affine = np.diag([1., 1., 3., 1.])
            nib.save(nib.Nifti1Image(np.ones((8, 8, 8), np.float32), affine), pet)
            nib.save(nib.Nifti1Image(np.zeros((8, 8, 8), np.float32), affine), ct)
            record = dict(inputs=dict(pet=str(pet), pet_sha256=sha256(pet), masks={}))
            for name in LABELS:
                target = masks/f'{name}.nii.gz'
                nib.save(nib.Nifti1Image(np.ones((8, 8, 8), np.uint8), affine), target)
                record['inputs']['masks'][name] = dict(sha256=sha256(target))
            (adaptation/'manifest.json').write_text(json.dumps(record))
            result = workflow.verify_inputs(pet, ct, masks, adaptation)
            self.assertEqual(result['ct_grid_pet_max_absolute_difference'], 0.)
            wrong = root/'wrong_pet.nii.gz'
            nib.save(nib.Nifti1Image(np.ones((8, 8, 8), np.float32)*2, affine), wrong)
            with self.assertRaisesRegex(ValueError, 'does not reproduce'):
                workflow.verify_inputs(wrong, ct, masks, adaptation)

    def test_paired_rows_masks_and_qc_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adaptation = root/'adaptation'
            adaptation.mkdir()
            (adaptation/'manifest.json').write_text('{}')
            for name in ('pet', 'ct'):
                (root/name).write_text('fixture')
            original_config = {'untouched': True}
            fake = types.ModuleType('run_pipeline')
            fake.CONFIG = original_config
            fake.__file__ = str(Path(workflow.__file__).with_name('run_pipeline.py'))
            calls = []
            def extraction(*args, **kwargs):
                calls.append((args, kwargs))
                return [dict(PatientID='Case_test', Modality='PET', Organ=n,
                             original_firstorder_Mean=1.) for n in LABELS]
            fake.step3_radiomics = extraction
            config = {'output': {}, 'radiomics': {'params_file': 'params.yaml'}}
            with patch.dict('sys.modules', {'run_pipeline': fake}), patch.object(workflow, 'verify_inputs', return_value={}), patch.object(workflow, 'verify_overlap_inputs', return_value=set()):
                rows = workflow.extract_pair(root/'pet', root/'ct', root/'original', adaptation,
                                             root/'results', 'Case_test', config)
            self.assertEqual(len(rows), 14)
            self.assertEqual({r['MaskVariant'] for r in rows}, {'original', 'adapted'})
            self.assertNotEqual(calls[0][1]['native_mask_directory'], calls[1][1]['native_mask_directory'])
            self.assertNotEqual(calls[0][1]['qc'].directory, calls[1][1]['qc'].directory)
            self.assertIs(fake.CONFIG, original_config)
            self.assertTrue((root/'results/extraction_manifest.json').exists())

    def test_existing_workflow_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root/'sentinel.txt'
            sentinel.write_text('preserve')
            with self.assertRaises(FileExistsError):
                workflow.process_case(root/'pet', root/'ct', root/'mask', root, 'test')
            self.assertEqual(sentinel.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
