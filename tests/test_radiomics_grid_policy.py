import unittest

from radiomics_grid_policy import merge_pet_and_shape_results


class RadiomicsGridPolicyTests(unittest.TestCase):
    def test_native_pet_nonshape_and_ct_shape_are_merged(self):
        pet_result = {
            "diagnostics_Image-original_Spacing": "native-pet-spacing",
            "original_firstorder_Mean": 2.1,
            "original_glcm_Contrast": 0.4,
            "original_shape_VoxelVolume": 999.0,
        }
        shape_result = {
            "diagnostics_Image-original_Spacing": "ct-spacing",
            "original_shape_VoxelVolume": 123.0,
            "original_shape_Sphericity": 0.8,
            "original_firstorder_Mean": -1.0,
        }

        merged = merge_pet_and_shape_results(pet_result, shape_result)

        self.assertEqual(merged["diagnostics_Image-original_Spacing"], "native-pet-spacing")
        self.assertEqual(merged["original_firstorder_Mean"], 2.1)
        self.assertEqual(merged["original_glcm_Contrast"], 0.4)
        self.assertEqual(merged["original_shape_VoxelVolume"], 123.0)
        self.assertEqual(merged["original_shape_Sphericity"], 0.8)


if __name__ == "__main__":
    unittest.main()
