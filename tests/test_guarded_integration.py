"""Public workflow/core-dispatch tests with isolated synthetic inputs."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

import pet_diaphragm_adaptation as core
import run_adaptation_pipeline as workflow
from guarded_shape import verify_predictions
from overlap_gate import run_checked


class GuardedIntegrationTests(unittest.TestCase):
    def test_real_guarded_dispatch_and_saved_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            masks = root / 'masks'
            masks.mkdir()
            shape = (50, 50, 60)
            affine = np.diag([1., 1., 3., 1.])
            names = workflow.adaptation_inventory()
            for name in sorted(names):
                values = np.zeros(shape, np.uint8)
                if name == 'liver': values[:, :, 5:21] = 1
                if name == 'lung_lower_lobe_left': values[:, :, 23:38] = 1
                if name == 'lung_upper_lobe_left': values[:, :, 38:50] = 1
                nib.save(nib.Nifti1Image(values, affine), masks / f'{name}.nii.gz')
            z = np.arange(shape[2])
            pet_values = np.broadcast_to(.2 + 2 / (1 + np.exp((z - 18.5) / 1.3)), shape).copy()
            pet = root / 'pet.nii.gz'
            nib.save(nib.Nifti1Image(pet_values.astype(np.float32), affine), pet)
            hashes = {name: core.sha256(masks / f'{name}.nii.gz') for name in names}
            output = root / 'adaptation'
            rows, _ = run_checked(core.run, pet, masks, output,
                                  core.Parameters(field_model='guarded_smooth'), expected_names=names)
            report = json.loads((output / 'guard_report.json').read_text())
            self.assertGreater(report['scale'], 0.)
            verify_predictions(rows, report)
            self.assertTrue(json.loads((output / 'overlap_check.json').read_text())['passed'])
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['parameters']['field_model'],
                             'guarded_smooth')
            self.assertEqual(hashes, {name: core.sha256(masks / f'{name}.nii.gz') for name in names})

    def test_cached_workflow_rejects_overlap_before_output(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'output'
            with patch.object(workflow, 'verify_overlap_inputs', side_effect=ValueError('Organ overlap')):
                with self.assertRaisesRegex(ValueError, 'Organ overlap'):
                    workflow.process_case('pet', 'ct', 'masks', output, 'case')
            self.assertFalse(output.exists())

    def test_prediction_comparison_rejects_nonfinite_and_missing(self):
        guard = {'fitting_predictions': [{'patch': 'liver_1_1', 'after_mm': 0.}]}
        with self.assertRaises(AssertionError):
            verify_predictions([{'patch': 'liver_1_1', 'after_interval_error_mm': np.nan}], guard)
        with self.assertRaises(KeyError):
            verify_predictions([], guard)


if __name__ == '__main__':
    unittest.main()
