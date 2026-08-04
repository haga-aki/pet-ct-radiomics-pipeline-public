import unittest

import numpy as np

from visualize_mask_verification import find_representative_slices


class RepresentativeSliceTests(unittest.TestCase):
    def test_duplicate_anatomical_centers_still_fill_three_panels(self):
        shape = (32, 32, 32)
        masks = {}
        for organ in ("aorta", "kidney_left", "kidney_right"):
            mask = np.zeros(shape, dtype=np.uint8)
            mask[12:20, 16, 12:20] = 1
            masks[organ] = mask

        positions = find_representative_slices(masks, shape)

        self.assertEqual(len(positions["coronal"]), 3)
        self.assertEqual(len(set(positions["coronal"])), 3)


if __name__ == "__main__":
    unittest.main()
