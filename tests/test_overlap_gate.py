import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np

from overlap_gate import LABELS, check_masks, run_checked


class OverlapGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'; self.source.mkdir()
        self.expected = (*LABELS, 'kidney_right')
        for i, name in enumerate(LABELS):
            a = np.zeros((10, 3, 4), np.uint8); a[i, 0, 0] = 1
            self.save(self.source, name, a)
        a = np.zeros((10, 3, 4), np.uint8); a[9, 0, 0] = 1
        self.save(self.source, 'kidney_right', a)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, folder, name, a):
        nib.save(nib.Nifti1Image(a, np.diag([1., 1., 3., 1.])), folder / (name + '.nii.gz'))

    def test_clear(self):
        self.assertTrue(check_masks(self.source, self.source)['passed'])

    def test_original_overlap_stops_before_delegate(self):
        a = np.zeros((10, 3, 4), np.uint8); a[0, 0, 0] = 1
        self.save(self.source, 'kidney_right', a)
        with self.assertRaisesRegex(ValueError, 'Organ overlap'):
            run_checked(lambda *args: self.fail('Must not run'), None,
                        self.source, self.root / 'output', None, expected_names=self.expected)
        self.assertFalse((self.root / 'output').exists())

    def test_saved_overlap_never_exported(self):
        def delegate(pet, src, dest, params):
            (dest / 'masks').mkdir(parents=True)
            for name in LABELS:
                a = np.asarray(nib.load(src / (name + '.nii.gz')).dataobj).copy()
                if name == 'liver': a[9, 0, 0] = 1
                self.save(dest / 'masks', name, a)
        with self.assertRaisesRegex(ValueError, 'Organ overlap'):
            run_checked(delegate, None, self.source, self.root / 'output', None,
                        expected_names=self.expected)
        self.assertFalse((self.root / 'output').exists())
        self.assertEqual(len(list(self.root.glob('.*.pending-*/REJECTED.json'))), 1)

    def test_target_overlap_rejected(self):
        a = np.zeros((10, 3, 4), np.uint8); a[0, 0, 0] = 1
        self.save(self.source, 'spleen', a)
        self.assertFalse(check_masks(self.source, self.source)['passed'])

    def test_missing_protection_rejected(self):
        (self.source / 'kidney_right.nii.gz').unlink()
        with self.assertRaisesRegex(ValueError, 'No other-organ'):
            check_masks(self.source, self.source)

    def test_accepted_export_has_report(self):
        def delegate(pet, src, dest, params):
            (dest / 'masks').mkdir(parents=True)
            for name in LABELS:
                a = np.asarray(nib.load(src / (name + '.nii.gz')).dataobj)
                self.save(dest / 'masks', name, a)
            return 'accepted'
        output = self.root / 'output'
        self.assertEqual(run_checked(delegate, None, self.source, output, None,
                                    expected_names=self.expected), 'accepted')
        self.assertTrue((output / 'overlap_check.json').exists())

    def test_geometry_mismatch_rejected(self):
        nib.save(nib.Nifti1Image(np.zeros((10, 3, 4), np.uint8), np.eye(4)),
                 self.source / 'kidney_right.nii.gz')
        with self.assertRaisesRegex(ValueError, 'share shape and affine'):
            check_masks(self.source, self.source)

    def test_nonfinite_protection_rejected(self):
        a = np.zeros((10, 3, 4), np.float32); a[9, 0, 0] = np.nan
        self.save(self.source, 'kidney_right', a)
        with self.assertRaisesRegex(ValueError, 'Nonbinary or nonfinite'):
            check_masks(self.source, self.source)

    def test_partial_inventory_rejected(self):
        with self.assertRaisesRegex(ValueError, 'inventory mismatch'):
            check_masks(self.source, self.source, (*self.expected, 'heart'))

    def test_uncompressed_mask_rejected(self):
        nib.save(nib.Nifti1Image(np.zeros((10, 3, 4), np.uint8), np.eye(4)),
                 self.source / 'heart.nii')
        with self.assertRaisesRegex(ValueError, 'Unsupported uncompressed'):
            check_masks(self.source, self.source, self.expected)

    def test_source_change_during_run_rejects_export(self):
        def delegate(pet, src, dest, params):
            (dest / 'masks').mkdir(parents=True)
            for name in LABELS:
                self.save(dest / 'masks', name, np.asarray(nib.load(src / (name+'.nii.gz')).dataobj))
            a = np.zeros((10, 3, 4), np.uint8); a[9, 0, 1] = 1
            self.save(src, 'kidney_right', a)
        with self.assertRaisesRegex(ValueError, 'changed during adaptation'):
            run_checked(delegate, None, self.source, self.root/'output', None,
                        expected_names=self.expected)
        self.assertFalse((self.root/'output').exists())

    def test_delegate_failure_rejects_export(self):
        def delegate(*args):
            raise RuntimeError('Synthetic failure')
        with self.assertRaisesRegex(RuntimeError, 'Synthetic failure'):
            run_checked(delegate, None, self.source, self.root/'output', None,
                        expected_names=self.expected)
        self.assertFalse((self.root/'output').exists())


if __name__ == '__main__':
    unittest.main()
