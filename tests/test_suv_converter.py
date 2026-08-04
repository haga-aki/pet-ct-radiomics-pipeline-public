import tempfile
import unittest
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset

from suv_converter import SUVConverter


class SUVConverterRescaleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        path = Path(self.temp_dir.name) / "00001.dcm"

        file_meta = Dataset()
        file_meta.MediaStorageSOPClassUID = pydicom.uid.PositronEmissionTomographyImageStorage
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Manufacturer = "GE MEDICAL SYSTEMS"
        ds.Units = "BQML"
        ds.RescaleSlope = "2.5"
        ds.RescaleIntercept = "0"
        ds.PatientWeight = "70"
        ds.AcquisitionTime = "100000"
        ds.DecayCorrection = "START"

        radiopharmaceutical = Dataset()
        radiopharmaceutical.RadionuclideTotalDose = "200000000"
        radiopharmaceutical.RadionuclideHalfLife = "6586.2"
        radiopharmaceutical.RadiopharmaceuticalStartTime = "090000"
        ds.RadiopharmaceuticalInformationSequence = [radiopharmaceutical]
        ds.save_as(path, enforce_file_format=True)

        self.converter = SUVConverter(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_raw_dicom_pixels_receive_rescale_once(self):
        raw = np.array([40.0])
        expected_bqml = raw * 2.5
        expected = self.converter.calculate_suv_from_bqml(expected_bqml)

        actual = self.converter.convert_to_suv(raw)

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_dicom2nifti_values_are_not_rescaled_twice(self):
        already_rescaled_bqml = np.array([40.0 * 2.5])
        expected = self.converter.calculate_suv_from_bqml(already_rescaled_bqml)

        actual = self.converter.convert_to_suv(
            already_rescaled_bqml, input_is_rescaled=True
        )

        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_double_rescale_regression_factor(self):
        already_rescaled_bqml = np.array([100.0])
        corrected = self.converter.convert_to_suv(
            already_rescaled_bqml, input_is_rescaled=True
        )
        incorrectly_rescaled = self.converter.convert_to_suv(already_rescaled_bqml)

        np.testing.assert_allclose(
            incorrectly_rescaled / corrected, np.array([2.5]), rtol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
