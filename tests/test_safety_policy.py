import ast
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

from mask_inventory import expected_masks
from suv_timing import decay_seconds, positive


def dataset(**changes):
    rp = NS(RadiopharmaceuticalStartTime='090000')
    ds = NS(DecayCorrection='START', AcquisitionTime='100000', RadiopharmaceuticalInformationSequence=[rp])
    for name, value in changes.items():
        setattr(ds, name, value)
    return ds


class SafetyTests(unittest.TestCase):
    def test_timing_modes(self):
        with self.assertRaises(ValueError):
            decay_seconds(dataset())
        self.assertEqual(decay_seconds(dataset(), allow_same_day_injection=True), 3600)
        self.assertEqual(decay_seconds(dataset(DecayCorrection='ADMIN', AcquisitionTime=None)), 0)
        for mode in ('NONE', '', 'UNKNOWN'):
            with self.assertRaises(ValueError):
                decay_seconds(dataset(DecayCorrection=mode))

    def test_missing_invalid_and_ambiguous_times(self):
        for time in (None, '', '10', '250000', '085959'):
            with self.assertRaises(ValueError):
                decay_seconds(dataset(AcquisitionTime=time, SeriesTime='100000'), allow_same_day_injection=True)
        ds = dataset()
        ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartTime = None
        with self.assertRaises(ValueError):
            decay_seconds(ds)

    def test_dated_midnight_and_fractional_time(self):
        ds = dataset(AcquisitionDateTime='20250102003000.5')
        ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartDateTime = '20250101233000.5'
        self.assertEqual(decay_seconds(ds), 3600)
        ds.AcquisitionDateTime = '20250101223000'
        with self.assertRaises(ValueError):
            decay_seconds(ds)
        for number in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                positive(number, 'synthetic')

    def test_expected_inventory_does_not_depend_on_files(self):
        maps = {'total': {1: 'liver', 2: 'spleen'}}
        self.assertEqual(expected_masks({}, 'PET', maps), ['liver', 'spleen'])
        self.assertEqual(expected_masks({'segmentation': {'roi_subset': ['spleen']}}, 'CT', maps), ['spleen'])
        with self.assertRaises(ValueError):
            expected_masks({'segmentation': {'roi_subset': ['typo']}}, 'CT', maps)

    def test_mixed_timezones_rejected(self):
        ds = dataset(AcquisitionDateTime='20250101100000+0900')
        ds.RadiopharmaceuticalInformationSequence[0].RadiopharmaceuticalStartDateTime = '20250101090000'
        with self.assertRaises(ValueError):
            decay_seconds(ds)

    def test_existing_conversion_is_never_reused(self):
        source = Path(__file__).resolve().parents[1] / 'run_pipeline.py'
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'step1_convert_dicom')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'synthetic_CT.nii.gz').touch()
            converter = Mock()
            env = dict(NIFTI_DIR=root, log_progress=Mock(), dicom2nifti=converter)
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), env)
            with self.assertRaises(FileExistsError):
                env['step1_convert_dicom'](root, 'synthetic', 'CT')
            converter.dicom_series_to_nifti.assert_not_called()
